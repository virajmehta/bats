from graph_tool import Graph, load_graph, ungroup_vector_property
import util
import time
import numpy as np
from subprocess import Popen
from tqdm import trange, tqdm
from modelling.dynamics_construction import train_ensemble
from modelling.policy_construction import train_policy, load_policy
from sklearn.neighbors import radius_neighbors_graph
from ipdb import set_trace as db  # NOQA
# torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn')


class BATSTrainer:
    def __init__(self, dataset, env, output_dir, **kwargs):
        self.dataset = dataset
        self.env = env
        self.obs_dim = self.env.observation_space.high.shape[0]
        self.action_dim = self.env.action_space.high.shape[0]
        self.output_dir = output_dir
        self.gamma = kwargs.get('gamma', 0.99)
        self.tqdm = kwargs.get('tqdm', True)
        self.n_val_iterations = kwargs.get('n_val_iterations', 20)
        all_obs = np.concatenate((dataset['observations'], dataset['next_observations']))
        self.unique_obs = np.unique(all_obs, axis=0)
        self.graph_size = self.unique_obs.shape[0]
        self.dataset_size = self.dataset['observations'].shape[0]

        # set up the parameters for the dynamics model training
        self.dynamics_ensemble = None

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
        # self.bc_params['hidden_sizes'] = kwargs.get('policy_hidden_sizes', '256, 256')

        # could do it this way or with knn, this is simpler to implement for now
        self.epsilon_neighbors = kwargs.get('epsilon_neighbors', 0.05)  # no idea what this should be
        self.possible_stitches = None
        self.possible_stitch_priorities = None
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

        # the actions are gonna be associated with each edge
        self.G.ep.action = self.G.new_edge_property("vector<float>")
        # we also associate the rewards with each edge
        self.G.ep.reward = self.G.new_edge_property("float")

        # parameters for planning
        self.epsilon_planning = kwargs.get('epsilon_planning', 0.05)  # also no idea what this should be
        self.planning_quantile = kwargs.get('planning_quantile', 0.8)
        self.num_cpus = kwargs.get('num_cpus', 1)
        self.stitching_chunk_size = kwargs.get('stitching_chunk_size', 2000000)

        # parameters for evaluation
        self.num_eval_episodes = kwargs.get("num_eval_episodes", 20)

        # parameters for interleaving
        self.num_stitching_iters = kwargs.get('num_stitching_iters', 50)

        # printing parameters
        self.neighbor_print_period = 1000

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
        if kwargs['load_neighbors'] is not None:
            stitches_path = kwargs['load_neighbors'] / 'possible_stitches.npy'
            self.possible_stitches = np.load(stitches_path)
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
        if self.policy is not None:
            self.evaluate()
            return
        # if this order is changed loading behavior might break
        self.add_dataset_edges()
        self.find_possible_stitches()
        self.train_dynamics()
        self.G.save(str(self.output_dir / 'dataset.gt'))
        processes = None
        for _ in trange(self.num_stitching_iters):
            self.value_iteration()
            self.compute_stitch_priorities()
            stitches_to_try = self.get_prioritized_stitch_chunk()
            if processes is not None:
                self.block_add_edges(processes)
            # edges_to_add should be an asynchronous result object, we'll run value iteration and
            # all other computations needed to prioritize the next round of stitches while this is running
            processes = self.test_neighbor_edges(stitches_to_try)
            self.G.save(str(self.output_dir / 'mdp.gt'))
            if stitches_to_try.shape[0] == 0:
                break
        self.block_add_edges(processes)
        self.G.save(str(self.output_dir / 'mdp.gt'))
        self.value_iteration()
        self.G.save(str(self.output_dir / 'vi.gt'))
        self.graph_stitching_done = True
        self.value_iteration_done = True
        self.train_bc()
        self.evaluate()

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
        print('testing possible stitches')
        chunksize = possible_stitches.shape[0] // self.num_cpus
        input_path = self.output_dir / 'input'
        output_path = self.output_dir / 'output'
        processes = []
        for i in range(self.num_cpus):
            cpu_chunk = possible_stitches[i * chunksize:(i + 1) * chunksize, :]
            fn = input_path / f"{i}.npy"
            np.save(fn, cpu_chunk)
            output_file = output_path / f"{i}.npy"
            process = Popen(['python', 'plan.py', str(fn), str(output_file), str(self.dynamics_ensemble_path),
                             str(self.obs_dim), str(self.action_dim), str(self.epsilon_planning),
                             str(self.planning_quantile)])
            processes.append(process)
        return processes

    def block_add_edges(self, processes):
        edges_added = 0
        output_path = self.output_dir / 'output'
        for i, process in enumerate(processes):
            process.wait()
            output_file = output_path / f"{i}.npy"
            edges_to_add = np.load(output_file)
            self.add_edges(edges_to_add)
            edges_added += edges_to_add.shape[0]
        processes = None

    def add_edges(self, edges_to_add):
        starts = edges_to_add[:, 0].astype(int)
        ends = edges_to_add[:, 1].astype(int)
        actions = edges_to_add[:, 2:self.action_dim + 2]
        rewards = edges_to_add[:, -1]
        for start, end, action, reward in zip(starts, ends, actions, rewards):
            e = self.G.add_edge(start, end)
            self.G.ep.action[e] = action
            self.G.ep.reward[e] = reward

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
            v_from = self.get_vertex(obs)
            v_to = self.get_vertex(next_obs)
            e = self.G.add_edge(v_from, v_to)
            self.G.ep.action[e] = action.tolist()  # not sure if the tolist is needed
            self.G.ep.reward[e] = reward
            last_obs = next_obs
        self.G.vp.start_node.get_array()[:] = start_nodes

    def find_possible_stitches(self):
        '''
        saves a list in the form of an ndarray with N x D
        where N is the number of possible stitches and D is 2 + (2 * obs_dim) + action_dim
        where each row is start_vertex, end_vertex, start_obs, end_obs, initial_action
        '''
        if self.possible_stitches is not None or self.graph_stitching_done:
            print(f'skipping the nearest neighbors step, loaded {self.possible_stitches.shape[0]} possible stitches')
            return
        print("finding possible neighbors")
        # this is the only step with quadratic time complexity, watch out for how long it takes
        start = time.time()
        neighbors = radius_neighbors_graph(self.unique_obs, self.epsilon_neighbors, n_jobs=-1)
        possible_neighbors = np.column_stack(neighbors.nonzero())
        print(f"Time to find possible neighbors: {time.time() - start:.2f}s")
        print(f"found {possible_neighbors.shape[0] // 2} possible neighbors")
        print(f"converting to edges")
        possible_stitches = []
        action_props = ungroup_vector_property(self.G.ep.action, range(self.action_dim))
        state_props = ungroup_vector_property(self.G.vp.obs, range(self.obs_dim))
        for row in tqdm(possible_neighbors):
            # get all the possible neighbors of the row donating from v0 to v1
            # since each pair should show up in both orders in the list, all pairs will get donations
            # both ways
            v0 = row[0]
            v1 = row[1]
            v1_obs = self.unique_obs[v1, :]
            v0_in_neighbors = self.G.get_in_neighbors(v0, vprops=state_props)
            v0_in_edges = self.G.get_in_edges(v0, eprops=action_props)
            v0_in_targets = np.ones_like(v0_in_neighbors[:, :1]) * v1
            possible_stitches.append(np.concatenate((v0_in_neighbors[:, :1],
                                                     v0_in_targets,
                                                     v0_in_neighbors[:, 1:],
                                                     np.tile(v1_obs, (v0_in_neighbors.shape[0], 1)),
                                                     v0_in_edges[:, 2:]), axis=-1))
            v0_out_neighbors = self.G.get_out_neighbors(v0, vprops=state_props)
            v0_out_edges = self.G.get_out_edges(v0, eprops=action_props)
            v0_out_start = np.ones_like(v0_out_neighbors[:, :1]) * v1
            possible_stitches.append(np.concatenate((v0_out_start,
                                                     v0_out_neighbors[:, :1],
                                                     np.tile(v1_obs, (v0_out_neighbors.shape[0], 1)),
                                                     v0_out_neighbors[:, 1:],
                                                     v0_out_edges[:, 2:]), axis=-1))

        self.possible_stitches = np.concatenate(possible_stitches, axis=0)
        print(f"found {self.possible_stitches.shape[0]} possible stitches")
        np.save(self.output_dir / 'possible_stitches.npy', self.possible_stitches)

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
        indices = np.argpartition(self.possible_stitch_priorities,
                                  self.stitching_chunk_size)[:self.stitching_chunk_size]
        stitch_chunk = self.possible_stitches[indices]
        self.possible_stitches = np.delete(self.possible_stitches, indices, axis=0)
        return stitch_chunk

    def value_iteration(self):
        '''
        This value iteration step actually computes two things:
        the value function and the occupancy distribution. Since the value function is pretty
        straightforward I'll give an explanation of the procedure for the value function in the comments.
        '''
        if self.value_iteration_done:
            print("skipping value iteration")
            return
        print("performing value iteration")
        # first we initialize the occupancies with the first nodes as 1
        self.G.vp.occupancy.get_array()[:] = self.G.vp.start_node.get_array().astype(float)

        iterator = self.get_iterator(self.n_val_iterations)
        for i in iterator:
            for v in self.G.iter_vertices():
                # should be a (num_neighbors, 2) ndarray where the first col is indices and second is values
                neighbor_values = self.G.get_out_neighbors(v, vprops=[self.G.vp.value])
                if len(neighbor_values) == 0:
                    # the state has no actions in our graph
                    self.G.vp.value[v] = 0
                    self.G.vp.best_neighbor[v] = -1
                    continue
                values = neighbor_values[:, 1]
                neighbors = neighbor_values[:, 0]
                # should be a (num_neighbors, 3) ndarray where the first col is v second col is indices, third is reward
                edges = self.G.get_out_edges(v, eprops=[self.G.ep.reward])
                rewards = edges[:, 2]
                backups = rewards + self.gamma * values
                best_arm = np.argmax(backups)
                value = backups[best_arm]
                if self.use_occupancy:
                    boltzmann_backups = np.exp(backups)
                    boltzmann_backups /= boltzmann_backups.sum()
                    occupancies = boltzmann_backups * self.G.vp.occupancy[v] * self.gamma
                    self.G.vp.occupancies.get_array()[neighbors] += occupancies
                self.G.vp.value[v] = value
                self.G.vp.best_neighbor[v] = neighbors[best_arm]

    def train_bc(self):
        print("cloning a policy")
        actions = []
        bad_indices = []
        for v in self.G.iter_vertices():
            best_neighbor = self.G.vp.best_neighbor[v]
            if best_neighbor == -1:
                # there were no outgoing edges found from this vertex
                bad_indices.append(v)
                continue
            e = self.G.edge(v, best_neighbor)
            action = np.array(self.G.ep.action[e])
            actions.append(action)
        actions = np.stack(actions)
        obs = np.delete(self.unique_obs, bad_indices, axis=0)
        dataset = {'observations': obs,
                   'actions': actions}
        self.policy = train_policy(dataset, **self.bc_params)

    def evaluate(self):
        # we have util.rollout ready for this purpose
        print("evaluating policy")
        episodes = []
        returns = []
        for i in self.get_iterator(self.num_eval_episodes):
            episode = util.rollout(self.policy, self.env)
            episodes.append(episode)
            ep_return = util.get_return(episode)
            returns.append(ep_return)
        return_mean = np.mean(returns)
        return_std = np.std(returns)
        normalized_mean = self.env.get_normalized_score(return_mean)
        tqdm.write(f"Mean Return | {return_mean:.2f} | Std Return | {return_std:.2f} | Normalized mean | {normalized_mean:.2f}")  # NOQA
