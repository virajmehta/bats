from graph_tool import Graph, load_graph, ungroup_vector_property
from graph_tool.spectral import adjacency
import time
import numpy as np
from scipy.sparse import diags
import pickle
from subprocess import Popen
from tqdm import trange, tqdm
from scipy.sparse import save_npz, load_npz
from modelling.dynamics_construction import train_ensemble
from modelling.policy_construction import load_policy, behavior_clone
from modelling.utils.graph_util import make_boltzmann_policy_dataset
from sklearn.neighbors import radius_neighbors_graph
from examples.mazes.maze_util import get_starts_from_graph
from ipdb import set_trace as db  # NOQA


class BATSTrainer:
    def __init__(self, dataset, env, output_dir, env_name, **kwargs):
        self.dataset = dataset
        self.env = env
        self.env_name = env_name
        self.obs_dim = self.env.observation_space.high.shape[0]
        self.action_dim = self.env.action_space.high.shape[0]
        self.output_dir = output_dir
        self.gamma = kwargs.get('gamma', 0.99)
        self.tqdm = kwargs.get('tqdm', True)
        self.n_val_iterations = kwargs.get('n_val_iterations', 10)
        self.n_val_iterations_end = kwargs.get('n_val_iterations_end', 50)
        all_obs = np.concatenate((dataset['observations'], dataset['next_observations']))
        self.unique_obs = np.unique(all_obs, axis=0)
        self.graph_size = self.unique_obs.shape[0]
        self.dataset_size = self.dataset['observations'].shape[0]

        # set up the parameters for the dynamics model training
        self.dynamics_ensemble_path = None

        self.dynamics_train_params = {}
        self.dynamics_train_params['n_members'] = kwargs.get('dynamics_n_members', 7)
        self.dynamics_train_params['n_elites'] = kwargs.get('dynamics_n_elites', 5)
        self.dynamics_train_params['save_dir'] = str(output_dir)
        self.dynamics_train_params['epochs'] = kwargs.get('dynamics_epochs', 100)
        self.dynamics_train_params['cuda_device'] = kwargs.get('cuda_device', '')

        # set up the parameters for behavior cloning
        self.policy = None
        self.bc_params = {}
        self.bc_params['save_dir'] = str(output_dir)
        self.bc_params['epochs'] = kwargs.get('bc_epochs', 100)
        self.bc_params['cuda_device'] = kwargs.get('cuda_device', '')
        self.bc_params['hidden_sizes'] = kwargs.get('policy_hidden_sizes', '256,256')
        self.temperature = kwargs.get('temperature', 0.25)
        # self.bc_params['hidden_sizes'] = kwargs.get('policy_hidden_sizes', '256, 256')

        # could do it this way or with knn, this is simpler to implement for now
        self.epsilon_neighbors = kwargs.get('epsilon_neighbors', 0.05)  # no idea what this should be
        self.neighbors = None
        # self.possible_stitch_priorities = None

        # set up graph
        self.G = Graph()
        self.value_iteration_done = False
        self.graph_stitching_done = False
        # add a vertex for each state
        self.G.add_vertex(self.graph_size)
        # make sure we keep the mapping from states to vertices
        self.vertices = {obs.tobytes(): i for i, obs in enumerate(self.unique_obs)}
        # the values associated with each node
        self.G.vp.value = self.G.new_vertex_property("float")
        self.G.vp.value.get_array()[:] = 0
        self.G.vp.best_neighbor = self.G.new_vertex_property("int")
        self.G.vp.obs = self.G.new_vertex_property('vector<float>')
        self.G.vp.obs.set_2d_array(self.unique_obs.copy().T)
        self.G.vp.start_node = self.G.new_vertex_property('bool')
        self.G.vp.occupancy = self.G.new_vertex_property('float')
        self.G.vp.terminal = self.G.new_vertex_property('bool')
        self.start_states = None
        self.start_state_path = self.output_dir / 'starts.npy'

        # the actions are gonna be associated with each edge
        self.G.ep.action = self.G.new_edge_property("vector<float>")
        # we also associate the rewards with each edge
        self.G.ep.reward = self.G.new_edge_property("float")
        # whether an edge is real or imagined
        self.G.ep.imagined = self.G.new_edge_property('bool')

        self.action_props = ungroup_vector_property(self.G.ep.action, range(self.action_dim))
        self.state_props = ungroup_vector_property(self.G.vp.obs, range(self.obs_dim))

        # parameters for planning
        self.epsilon_planning = kwargs.get('epsilon_planning', 0.05)  # also no idea what this should be
        self.planning_quantile = kwargs.get('planning_quantile', 0.8)
        self.num_cpus = kwargs.get('num_cpus', 1)
        self.plan_cpus = 2  # self.num_cpus //  10
        self.stitching_chunk_size = kwargs['stitching_chunk_size']
        self.rollout_chunk_size = kwargs['stitching_chunk_size']
        self.max_stitches = kwargs['max_stitches']
        self.stitches_tried = set()
        # this saves an empty file so the child processes can see it
        self.remove_neighbors([])

        # parameters for evaluation
        self.num_eval_episodes = kwargs.get("num_eval_episodes", 20)

        # parameters for interleaving
        self.num_stitching_iters = kwargs.get('num_stitching_iters', 50)

        # printing parameters
        self.neighbor_print_period = 1000

        # normalizing before neighbors
        if kwargs['normalize_obs']:
            self.mean = all_obs.mean(axis=0, keepdims=True)
            self.std = all_obs.std(axis=0, keepdims=True)
            self.neighbor_obs = (self.unique_obs  - self.mean) / self.std
            self.mean_file = self.output_dir / 'mean.npy'
            np.save(self.mean_file, self.mean)
            self.std_file = self.output_dir / 'std.npy'
            np.save(self.std_file, self.std)
        else:
            self.mean = None
            self.std = None
            self.mean_file = None
            self.std_file = None
            self.neighbor_obs = self.unique_obs


        self.use_occupancy = kwargs.get('use_occupancy', False)
        # check for all loads
        if kwargs['load_policy'] is not None:
            self.policy = load_policy(str(kwargs['load_policy']), self.obs_dim, self.action_dim,
                                      cuda_device=self.bc_params['cuda_device'])
            return
        if kwargs['load_value_iteration'] is not None:
            graph_path = kwargs['load_value_iteration'] / 'vi.gt'
            self.G = load_graph(str(graph_path))
            self.value_iteration_done = True
            self.graph_stitching_done = True
            return
        if kwargs['load_graph'] is not None:
            graph_path = kwargs['load_graph'] / 'mdp.gt'
            self.G = load_graph(str(graph_path))
            self.graph_stitching_done = True
            return
        self.neighbor_name = 'neighbors.npz'
        if kwargs['load_neighbors'] is not None:
            neighbors_path = kwargs['load_neighbors'] / self.neighbor_name
            self.neighbors = load_npz(neighbors_path)
            our_neighbor_path = self.output_dir / self.neighbor_name
            save_npz(our_neighbor_path, self.neighbors)
        if kwargs['load_model'] is not None:
            self.dynamics_ensemble_path = str(kwargs['load_model'])

    def get_vertex(self, obs):
        return self.G.vertex(self.vertices[obs.tobytes()])

    def get_iterator(self, n):
        if self.tqdm:
            return trange(n)
        else:
            return range(n)

    def train(self):
        if self.graph_stitching_done:
            self.train_bc()
        # if this order is changed loading behavior might break
        self.add_dataset_edges()
        # get a list of the start states for the graph
        if 'maze' in self.env_name:
            self.start_states = get_starts_from_graph(self.G, self.env)
        else:
            self.start_states = np.argwhere(self.G.vp.start_node.get_array()).flatten()
        print(f"Found {len(self.start_states)} start nodes")
        np.save(self.start_state_path, self.start_states)
        self.find_possible_stitches()
        self.train_dynamics()
        self.G.save(str(self.output_dir / 'dataset.gt'))
        self.G.save(str(self.output_dir / 'mdp.gt'))
        processes = None
        self.value_iteration(self.n_val_iterations_end)
        for i in trange(self.num_stitching_iters):
            self.value_iteration(self.n_val_iterations)
            stitch_start_time = time.time()
            stitches_to_try = self.get_rollout_stitch_chunk()
            print(f"Time to find good stitches: {time.time() - stitch_start_time:.2f}s")
            # edges_to_add should be an asynchronous result object, we'll run value iteration and
            # all other computations needed to prioritize the next round of stitches while this is running
            if stitches_to_try.shape[0] == 0:
                break
            plan_start_time = time.time()
            processes = self.test_neighbor_edges(stitches_to_try)
            self.block_add_edges(processes)
            print(f"Time to test edges: {time.time() - plan_start_time:.2f}s")
            self.G.save(str(self.output_dir / 'mdp.gt'))
        self.G.save(str(self.output_dir / 'mdp.gt'))
        self.value_iteration(self.n_val_iterations_end)
        self.G.save(str(self.output_dir / 'vi.gt'))
        self.graph_stitching_done = True
        self.value_iteration_done = True
        self.train_bc()

    def train_dynamics(self):
        if self.dynamics_ensemble_path or self.graph_stitching_done:
            print('skipping dynamics ensemble training')
            return
        print("training ensemble of dynamics models")
        train_ensemble(self.dataset, **self.dynamics_train_params)
        self.dynamics_ensemble_path = str(self.output_dir)

    def test_neighbor_edges(self, possible_stitches):
        '''
        This function currently assumes whatever prioritization there is exists outside the function and that it is
        supposed to try all possible stitches.
        I tested this and it made things slower:
        # for model in self.dynamics_ensemble:
            # model.share_memory()
        '''
        if self.graph_stitching_done:
            print('skipping graph stitching')
            return
        print(f'testing {possible_stitches.shape[0]} possible stitches')
        chunksize = possible_stitches.shape[0] // self.plan_cpus
        input_path = self.output_dir / 'input'
        output_path = self.output_dir / 'output'
        processes = []
        for i in range(self.plan_cpus):
            cpu_chunk = possible_stitches[i * chunksize:(i + 1) * chunksize, :]
            fn = input_path / f"{i}.npy"
            np.save(fn, cpu_chunk)
            output_file = output_path / f"{i}.npy"
            args = ['python',
                    'plan.py',
                    str(fn),
                    str(output_file),
                    str(self.dynamics_ensemble_path),
                    str(self.obs_dim),
                    str(self.action_dim),
                    str(self.epsilon_planning),
                    str(self.planning_quantile)]
            if self.std_file:
                args += [self.mean_file, self.std_file]
            process = Popen(args)
            processes.append(process)
        return processes

    def block_add_edges(self, processes):
        edges_added = 0
        output_path = self.output_dir / 'output'
        for i, process in enumerate(processes):
            process.wait()
            output_file = output_path / f"{i}.npy"
            edges_to_add = np.load(output_file)
            if len(edges_to_add) == 0:
                continue
            edges_added += self.add_edges(edges_to_add)
        print(f"adding {edges_added} edges")

    def add_edges(self, edges_to_add):
        starts = edges_to_add[:, 0].astype(int)
        ends = edges_to_add[:, 1].astype(int)
        actions = edges_to_add[:, 2:self.action_dim + 2]
        rewards = edges_to_add[:, -1]
        added = 0
        for start, end, action, reward in zip(starts, ends, actions, rewards):
            if self.G.vp.terminal[start]:
                # we don't want to add edges originating from terminal states
                continue
            e = self.G.add_edge(start, end)
            self.G.ep.action[e] = action
            self.G.ep.reward[e] = reward
            if not np.isfinite(reward):
                db()
            self.G.ep.imagined[e] = True
            added += 1
        return added

    def add_dataset_edges(self):
        if self.graph_stitching_done:
            print("skipping adding dataset edges")
            return
        print(f"Adding {self.dataset_size} initial edges to our graph")
        iterator = self.get_iterator(self.dataset_size)
        last_obs = None
        start_nodes = np.zeros((len(self.vertices),))
        for i in iterator:
            obs = self.dataset['observations'][i, :]
            next_obs = self.dataset['next_observations'][i, :]
            if (last_obs != obs).any():
                vnum = self.vertices[obs.tobytes()]
                start_nodes[vnum] = 1
            action = self.dataset['actions'][i, :]
            reward = self.dataset['rewards'][i]
            terminal = self.dataset['terminals'][i]
            v_from = self.get_vertex(obs)
            v_to = self.get_vertex(next_obs)
            self.stitches_tried.add((v_from, v_to))
            e = self.G.add_edge(v_from, v_to)
            self.G.ep.action[e] = action.tolist()  # not sure if the tolist is needed
            self.G.ep.reward[e] = reward
            self.G.ep.imagined[e] = False
            self.G.vp.terminal[v_to] = terminal
            last_obs = next_obs
        self.G.vp.start_node.get_array()[:] = start_nodes

    def find_possible_stitches(self):
        '''
        saves a list in the form of an ndarray with N x D
        where N is the number of possible stitches and D is 2 + (2 * obs_dim) + action_dim
        where each row is start_vertex, end_vertex, start_obs, end_obs, initial_action
        '''
        if self.neighbors is not None or self.graph_stitching_done:
            print(f'skipping the nearest neighbors step, loaded {self.neighbors.nnz} neighbors')
            return
        print("finding possible neighbors")
        # this is the only step with quadratic time complexity, watch out for how long it takes
        start = time.time()
        self.neighbors = radius_neighbors_graph(self.neighbor_obs, self.epsilon_neighbors, n_jobs=-1).astype(bool)
        print(f"Time to find possible neighbors: {time.time() - start:.2f}s")
        print(f"Found {self.neighbors.nnz // 2} neighbor pairs")
        save_npz(self.output_dir / self.neighbor_name, self.neighbors)

    def compute_stitch_priorities(self):
        print(f"Computing updated priorities for stitches")
        # for now the priority will just be the -(approximate advantage)
        priorities = np.empty(self.possible_stitches.shape[0])
        for i, stitch in enumerate(tqdm(self.possible_stitches)):
            start_vertex = stitch[0]
            end_vertex = stitch[0]
            end_vertex_value = self.G.vp.value[end_vertex]
            start_vertex_value = self.G.vp.value[start_vertex]
            start_vertex_occupancy = self.G.vp.occupancy[start_vertex]
            advantage = end_vertex_value - start_vertex_value
            priorities[i] = -advantage
            if self.use_occupancy:
                priorities[i] *= start_vertex_occupancy
        self.possible_stitch_priorities = priorities

    def get_prioritized_stitch_chunk(self):
        if self.stitching_chunk_size >= len(self.possible_stitch_priorities):
            indices = np.arange(len(self.possible_stitch_priorities)).astype(int)
        else:
            indices = np.argpartition(self.possible_stitch_priorities,
                                      self.stitching_chunk_size)[:self.stitching_chunk_size]
        stitch_chunk = self.possible_stitches[indices]
        self.possible_stitches = np.delete(self.possible_stitches, indices, axis=0)
        return stitch_chunk

    def value_iteration(self, n_iters):
        '''
        This value iteration step actually computes two things:
        the value function and the occupancy distribution. Since the value function is pretty
        straightforward I'll give an explanation of the procedure for the value function in the comments.
        '''
        if self.value_iteration_done:
            print("skipping value iteration")
            return
        print("performing value iteration")
        for i in trange(n_iters):
            db()
            # first we initialize the occupancies with the first nodes as 1
            if self.use_occupancy:
                raise NotImplementedError('Deprecating for now, not sure if we '
                                          'are still using or not.')
            # Construct sparse adjacency, reward, sparse value matrices.
            reward_mat = adjacency(self.G, weight=self.G.ep.reward)
            target_val = diags(
                (self.gamma * self.G.vp.value.get_array()
                            * (1 - self.G.vp.terminal.get_array())),
                format='csr',
            )
            # WARNING: graph-tool returns transpose of standard adjacency matrix
            #          hence the line target_val * adjmat (instead of reverse).
            adjmat = adjacency(self.G)
            out_degrees = np.array(adjmat.sum(axis=0))[0, ...]
            is_dead_end = out_degrees == 0
            target_mat = target_val * adjmat
            qs = reward_mat + target_mat
            # HACKINESS ALERT: To ignore zero entries in mat, add large value to
            #                  the non-zero entries.
            bst_childs = np.asarray((qs + adjmat * 1e4).argmax(axis=0)).flatten()
            values = np.asarray(
                    # TODO: I hate how I have to make arange here, how do I not?
                    qs[bst_childs, np.arange(self.G.num_vertices())]).flatten()
            values[is_dead_end] = 0
            bst_childs[is_dead_end] = -1
            self.G.vp.best_neighbor.a = bst_childs
            self.G.vp.value.a = values
            if not np.isfinite(values).all():
                db()

    def train_bc(self):
        print("cloning a policy")
        data, val_data = make_boltzmann_policy_dataset(
                graph=self.G,
                n_collects=2 * self.G.num_vertices(),  # not sure what this should be
                temperature=self.temperature,
                max_ep_len=self.env._max_episode_steps,
                n_val_collects=0,
                val_start_prop=0,
                silent=True,
                starts=self.start_states)
        self.policy = behavior_clone(dataset=data,
                                     env=self.env,
                                     max_ep_len=self.env._max_episode_steps,
                                     **self.bc_params)

    def get_rollout_stitch_chunk(self):
        # need to be less than rollout_chunk_size

        chunksize = self.rollout_chunk_size // self.num_cpus
        output_path = self.output_dir / 'rollout_output'
        output_path.mkdir(exist_ok=True)
        processes = []
        print("Getting possible stitches by rolling out best Boltzmann policy")
        for i in range(self.num_cpus):
            output_file = output_path / f"{i}.npy"
            args = ['python',
                    'rollout_stitches.py',
                    str(self.output_dir / 'mdp.gt'),
                    str(self.output_dir / self.neighbor_name),
                    str(self.output_dir / 'stitches_tried.pkl'),
                    str(self.start_state_path),
                    str(output_file),
                    str(self.action_dim),
                    str(self.obs_dim),
                    str(chunksize),
                    str(self.temperature),
                    str(self.gamma),
                    str(self.max_stitches)]
            process = Popen(args)
            processes.append(process)
        all_advantages = []
        all_stitches = []
        for i, process in enumerate(processes):
            process.wait()
            output_file = output_path / f"{i}.npy"
            outputs = np.load(output_file)
            advantages = outputs[:, 0]
            stitches = outputs[:, 1:]
            all_advantages.append(advantages)
            all_stitches.append(stitches)
        all_advantages = np.concatenate(all_advantages, axis=0)
        all_stitches = np.concatenate(all_stitches, axis=0)
        stitches, unique_indices = np.unique(all_stitches, axis=0, return_index=True)
        advantages = all_advantages[unique_indices]
        if self.stitching_chunk_size >= len(advantages):
            indices = np.arange(len(advantages)).astype(int)
        else:
            indices = np.argpartition(advantages, self.stitching_chunk_size)[:self.stitching_chunk_size]
        stitches_to_try = stitches[indices]
        self.remove_neighbors(stitches_to_try)
        print(f'Choosing {len(indices)} edges from Boltzmann rollouts')
        return stitches_to_try

    def remove_neighbors(self, stitches_to_try):
        for stitch in stitches_to_try:
            self.stitches_tried.add(tuple(stitch[:2]))
        neighbors_tried_path = self.output_dir / 'stitches_tried.pkl'
        with open(neighbors_tried_path, 'wb') as f:
            pickle.dump(self.stitches_tried, f)
