import os
from graph_tool import Graph, load_graph, ungroup_vector_property
from graph_tool.spectral import adjacency
import time
import numpy as np
from pathlib import Path
import torch
from scipy.sparse import diags
from collections import defaultdict
import pickle
from subprocess import Popen
from copy import deepcopy
from tqdm import trange
from scipy.sparse import save_npz, load_npz
from shutil import copy
from modelling.dynamics_construction import train_ensemble
from modelling.policy_construction import load_policy, behavior_clone
from modelling.dynamics_construction import load_ensemble
from modelling.bisim_construction import train_bisim, load_bisim, make_trainer, fine_tune_bisim
from modelling.utils.torch_utils import ModelUnroller
from modelling.utils.graph_util import make_boltzmann_policy_dataset
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph
from util import get_starts_from_graph


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
        self.max_val_iterations = kwargs.get('max_val_iterations', 1000)
        self.vi_tolerance = kwargs.get('vi_tolerance')
        all_obs = np.concatenate((dataset['observations'], dataset['next_observations']))
        self.unique_obs = np.unique(all_obs, axis=0)
        self.graph_size = self.unique_obs.shape[0]
        self.dataset_size = self.dataset['observations'].shape[0]
        self.verbose = kwargs['verbose']

        # set up the parameters for the dynamics model training
        self.bisim_model = None
        self.trainer = None

        self.dynamics_train_params = {}
        self.dynamics_train_params['n_members'] = kwargs.get('dynamics_n_members', 7)
        self.dynamics_train_params['n_elites'] = kwargs.get('dynamics_n_elites', 5)
        self.dynamics_train_params['save_dir'] = str(output_dir)
        self.dynamics_train_params['epochs'] = kwargs.get('dynamics_epochs', 100)
        self.dynamics_train_params['cuda_device'] = kwargs.get('cuda_device', '')

        # set up the parameters for the bisimulation metric space
        self.use_bisimulation = kwargs['use_bisimulation']
        self.penalize_stitches = kwargs['penalize_stitches']
        assert self.use_bisimulation or (not self.penalize_stitches)
        self.fine_tune_epochs = kwargs.get('fine_tune_epochs', 20)
        self.bisim_train_params = {}
        self.latent_dim = self.bisim_train_params['latent_dim'] = kwargs['bisim_latent_dim']
        self.bisim_train_params['epochs'] = kwargs.get('bisim_epochs', 250)
        self.bisim_train_params['dataset'] = self.dataset
        self.bisim_n_members = self.bisim_train_params['n_members'] = kwargs.get('bisim_n_members', 5)
        self.bisim_train_params['save_dir'] = self.output_dir

        # set up the parameters for behavior cloning
        self.policy = None
        self.bc_params = {}
        self.bc_params['save_dir'] = str(output_dir)
        self.bc_params['epochs'] = kwargs.get('bc_epochs', 50)
        self.bc_params['od_wait'] = kwargs.get('od_wait', 15)
        self.bc_params['cuda_device'] = kwargs.get('cuda_device', '')
        self.bc_params['hidden_sizes'] = kwargs.get('policy_hidden_sizes', '256,256')
        self.bc_params['batch_updates_per_epoch'] =\
            kwargs.get('batch_updates_per_epoch', 50)
        self.bc_params['add_entropy_bonus'] =\
            kwargs.get('add_entropy_bonus', True)
        self.intermediate_bc_params = deepcopy(self.bc_params)
        self.intermediate_bc_params['epochs'] = 30
        self.temperature = kwargs.get('temperature', 0.25)
        self.bolt_gather_params = {}
        self.bolt_gather_params['top_percent_starts'] =\
            kwargs.get('top_percent_starts', 0.8)
        self.bolt_gather_params['silent'] =\
            kwargs.get('silent', False)
        self.bolt_gather_params['get_unique_edges'] =\
            kwargs.get('get_unique_edges', False)
        self.bolt_gather_params['val_start_prop'] =\
            kwargs.get('val_start_prop', 0.05)
        self.bc_every_iter = kwargs['bc_every_iter']

        # could do it this way or with knn, this is simpler to implement for now
        self.epsilon_neighbors = kwargs.get('epsilon_neighbors', 0.05)  # no idea what this should be
        self.k_neighbors = kwargs['k_neighbors']
        self.neighbors = None
        self.neighbor_limit = 500000000  # 500 million

        self.max_stitch_length = kwargs.get('max_stitch_length', 1)
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
        self.G.vp.upper_value = self.G.new_vertex_property("float")
        self.G.vp.best_neighbor = self.G.new_vertex_property("int")
        self.G.vp.obs = self.G.new_vertex_property('vector<float>')
        self.G.vp.obs.set_2d_array(self.unique_obs.copy().T)
        self.G.vp.start_node = self.G.new_vertex_property('bool')
        self.G.vp.real_node = self.G.new_vertex_property('bool')
        self.G.vp.real_node.get_array()[:] = True
        self.G.vp.terminal = self.G.new_vertex_property('bool')
        self.start_states = None
        self.start_state_path = self.output_dir / 'starts.npy'
        if self.use_bisimulation:
            self.G.vp.z = self.G.new_vertex_property('vector<float>')

        # the actions are gonna be associated with each edge
        self.G.ep.action = self.G.new_edge_property("vector<float>")
        # we also associate the rewards with each edge
        self.G.ep.reward = self.G.new_edge_property("float")
        self.G.ep.upper_reward = self.G.new_edge_property("float")
        # whether an edge is real or imagined
        self.G.ep.imagined = self.G.new_edge_property('bool')
        # The model errors for stitching if this is a stitched edge.
        self.G.ep.model_errors = self.G.new_edge_property('vector<float>')
        # Iteration that the model was stitched at.
        self.G.ep.stitch_itr = self.G.new_edge_property('int')
        self.G.ep.stitch_length = self.G.new_edge_property('int')

        self.action_props = ungroup_vector_property(self.G.ep.action, range(self.action_dim))
        self.state_props = ungroup_vector_property(self.G.vp.obs, range(self.obs_dim))

        # parameters for planning
        self.epsilon_planning = kwargs['epsilon_planning']
        self.planning_quantile = kwargs.get('planning_quantile', 0.8)
        self.num_cpus = kwargs.get('num_cpus', 1)
        self.plan_cpus = 2  # self.num_cpus //  10
        self.stitching_chunk_size = kwargs['stitching_chunk_size']
        self.rollout_chunk_size = kwargs['stitching_chunk_size']
        self.max_stitches = kwargs['max_stitches']
        self.stitches_tried = set()
        self.edges_added = []
        self.penalty_coefficient = kwargs['penalty_coefficient']
        self.use_all_planning_itrs = kwargs.get('use_all_planning_itrs', False)
        # this saves an empty file so the child processes can see it
        self.remove_neighbors([])

        # parameters for evaluation
        self.num_eval_episodes = kwargs.get("num_eval_episodes", 20)

        # parameters for interleaving
        self.num_stitching_iters = kwargs.get('num_stitching_iters', 50)
        # Whether to keep making suboptimal stitches after no positive
        # advantage stitches can be made. This is good for hyper parameter opt.
        self.continue_after_no_advantage =\
                kwargs.get('continue_after_no_advantage', False)
        self.pick_positive_adv = True  # Set to False after no positive adv.

        # printing parameters
        self.neighbor_print_period = 1000

        self.normalize_obs = kwargs['normalize_obs']
        # normalizing before neighbors
        if self.use_bisimulation:
            self.mean = None
            self.std = None
            self.mean_file = None
            self.std_file = None
            self.neighbor_obs = None
        elif kwargs['normalize_obs']:
            self.mean = all_obs.mean(axis=0, keepdims=True)
            self.std = all_obs.std(axis=0, keepdims=True)
            self.neighbor_obs = (self.unique_obs - self.mean) / self.std
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

        # save graph stats stuff
        self.stats = defaultdict(list)
        self.stats_path = self.output_dir / 'graph_stats.npz'
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
        self.dynamics_ensemble_path = None
        self.dynamics_unroller = None
        if kwargs['load_model'] is not None:
            self.dynamics_ensemble_path = Path(kwargs['load_model'])
        if kwargs['load_bisim_model'] is not None:
            self.bisim_model_path = Path(kwargs['load_bisim_model'])

    def save_stats(self):
        np.savez(self.stats_path, **self.stats)

    def add_stat(self, name, value):
        self.stats[name].append(value)
        print(f"{name}: {value:.2f}")

    def get_vertex(self, obs):
        return self.G.vertex(self.vertices[obs.tobytes()])

    def get_iterator(self, n):
        if self.tqdm:
            return trange(n)
        else:
            return range(n)

    def train(self):
        # add the original dataset into the graph
        self.add_dataset_edges()
        # get a list of the start states for the graph
        self.start_states = np.argwhere(self.G.vp.start_node.get_array()).flatten()
        print(f"Found {len(self.start_states)} start nodes")
        np.save(self.start_state_path, self.start_states)
        nnz = self.train_dynamics()
        if nnz > self.neighbor_limit:
            print(f"Too many nearest neighbors ({nnz}), max is {self.neighbor_limit}")
            return None
        self.G.save(str(self.output_dir / 'dataset.gt'))
        self.G.save(str(self.output_dir / 'mdp.gt'))
        processes = None
        self.value_iteration()
        for i in self.get_iterator(self.num_stitching_iters):
            stitch_start_time = time.time()
            stitches_to_try = self.get_rollout_stitch_chunk()
            print(f"Time to find good stitches: {time.time() - stitch_start_time:.2f}s")
            # edges_to_add should be an asynchronous result object, we'll run value iteration and
            # all other computations needed to prioritize the next round of stitches while this is running
            if len(stitches_to_try) == 0:
                if self.continue_after_no_advantage:
                    self.pick_positive_adv = False
                    continue
                else:
                    break
            plan_start_time = time.time()
            processes = self.test_possible_stitches(stitches_to_try)
            self.block_add_edges(processes, i + 1)
            size = self.G.num_vertices()
            self.neighbors.resize((size, size))
            print(f"Time to test edges: {time.time() - plan_start_time:.2f}s")
            vi_start_time = time.time()
            self.value_iteration()
            print(f"Time for value iteration: {time.time() - vi_start_time:.2f}s")
            self.G.save(str(self.output_dir / 'mdp.gt'))
            if self.bc_every_iter:
                bc_start_time = time.time()
                self.train_bc(dir_name='itr_%d' % i, intermediate=True)
                print(f"Time for behavior cloning: {time.time() - bc_start_time:.2f}s")
            if self.use_bisimulation:
                nnz = self.fine_tune_dynamics()
                if nnz > self.neighbor_limit:
                    return None
            self.save_stats()
        self.G.save(str(self.output_dir / 'mdp.gt'))
        self.value_iteration()
        self.G.save(str(self.output_dir / 'vi.gt'))
        self.graph_stitching_done = True
        self.value_iteration_done = True
        self.train_bc()
        self.save_stats()

    def compute_embeddings(self):
        print("computing embeddings")
        embeddings = np.array(self.bisim_model.get_encoding(self.unique_obs))
        self.neighbor_obs = embeddings
        self.G.vp.z.set_2d_array(embeddings.copy().T)
        self.vertices = {obs.tobytes(): i for i, obs in enumerate(self.neighbor_obs)}

    def train_dynamics(self):
        if self.dynamics_ensemble_path is None and self.max_stitch_length > 1:
            print('training ensemble of dynamics models')
            train_ensemble(self.dataset, **self.dynamics_train_params)
            self.dynamics_ensemble_path = str(self.output_dir)
        else:
            print('skipping dynamics model training')

        if self.use_bisimulation:
            if self.bisim_model_path is not None:
                print('loading bisimulation model')
                self.bisim_model = load_bisim(self.bisim_model_path)
                copy(self.bisim_model_path / 'params.pkl', self.output_dir)
                self.trainer = make_trainer(self.bisim_model,
                                            self.bisim_n_members,
                                            self.output_dir)
            else:
                print('training bisimulation model')
                # viraj: it's possible I guess that the regular model training writes to the same place in the
                #        output_dir as the bisimulation training below but I'm not too worried about that rn
                self.bisim_model, self.bisim_trainer = train_bisim(**self.bisim_train_params)
                self.bisim_model_path = str(self.output_dir)
            self.compute_embeddings()
        dynamics_ensemble = load_ensemble(self.dynamics_ensemble_path, self.obs_dim, self.action_dim,
                                          cuda_device='')
        self.dynamics_unroller = ModelUnroller(self.env_name, dynamics_ensemble)
        nnz = self.find_nearest_neighbors()
        return nnz

    def fine_tune_dynamics(self):
        if not self.use_bisimulation:
            raise NotImplementedError()
        print("fine-tuning bisimulation model")
        data, _, _ = make_boltzmann_policy_dataset(
                graph=self.G,
                n_collects=self.G.num_vertices(),
                temperature=0.,
                max_ep_len=self.env._max_episode_steps,
                n_val_collects=0,
                val_start_prop=0,
                include_reward_next_obs=True,
                silent=True,
                starts=self.start_states,
                only_add_real=True)
        fine_tune_bisim(self.trainer,
                        self.fine_tune_epochs,
                        data)
        # once we've fine-tuned we need to use that model
        self.dynamics_ensemble_path = self.output_dir
        self.compute_embeddings()
        if self.penalize_stitches:
            self.recompute_edge_values()
        self.neighbors = None
        nnz = self.find_nearest_neighbors()
        return nnz

    def recompute_edge_values(self):
        if len(self.edges_added) == 0:
            return
        edges_added = np.array(self.edges_added)
        start_obs = self.neighbor_obs[edges_added[:, 0], :]
        end_obs = self.neighbor_obs[edges_added[:, 1], :]
        actions = []
        for start, end in self.edges_added:
            edge = self.G.edge(start, end)
            action = self.G.ep.action[edge]
            actions.append(action)
        actions = np.array(actions)
        conditions = np.concatenate([start_obs, actions], axis=1)
        model_outputs = self.bisim_model.get_mean_logvar(conditions)[0]
        state_outputs = model_outputs[:, :, 1:]
        reward_outputs = model_outputs[:, :, 0]
        displacements = state_outputs + start_obs - end_obs
        distances = torch.linalg.norm(displacements, dim=-1, ord=1)
        quantiles = np.quantile(distances, self.planning_quantile, axis=0)
        reward = np.quantile(reward_outputs, 0.3, axis=0)
        low_reward = reward - quantiles * self.gamma
        high_reward = reward + quantiles * self.gamma
        for i, (start, end) in enumerate(self.edges_added):
            edge = self.G.edge(start, end)
            self.G.ep.reward[edge] = low_reward[i]
            self.G.ep.upper_reward[edge] = high_reward[i]

    def test_possible_stitches(self, possible_stitches):
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
        print(f'testing {len(possible_stitches)} possible stitches')
        chunksize = len(possible_stitches) // self.plan_cpus
        input_path = self.output_dir / 'input'
        output_path = self.output_dir / 'output'
        processes = []
        for i in range(self.plan_cpus):
            cpu_chunk = possible_stitches[i * chunksize:(i + 1) * chunksize]
            fn = input_path / f"{i}.pkl"
            with fn.open('wb') as f:
                pickle.dump(cpu_chunk, f)
            output_file = output_path / f"{i}.pkl"
            model_path = str(self.bisim_model_path) if self.use_bisimulation else str(self.dynamics_ensemble_path)
            args = ['python',
                    'plan.py',
                    str(fn),
                    str(output_file),
                    model_path,
                    str(self.obs_dim),
                    str(self.action_dim),
                    str(self.latent_dim),
                    str(self.epsilon_planning),
                    str(self.planning_quantile),
                    str(self.max_stitch_length),
                    str(self.env_name)]
            if self.std_file:
                args += [self.mean_file, self.std_file]
            if self.use_bisimulation:
                args.append('-ub')
            if self.use_all_planning_itrs:
                args.append('-uapi')
            if self.verbose and i == 0:
                print(' '.join(args))
            process = Popen(args)
            processes.append(process)
        return processes

    def block_add_edges(self, processes, iteration):
        edges_added = 0
        output_path = self.output_dir / 'output'
        for i, process in enumerate(processes):
            process.wait()
            output_file = output_path / f"{i}.pkl"
            with output_file.open('rb') as f:
                edges_to_add = pickle.load(f)
            if len(edges_to_add) == 0:
                continue
            edges_added += self.add_edges(edges_to_add, iteration)
        print(f"adding {edges_added} edges")
        self.add_stat('edges_added', edges_added)

    def add_vertex(self, obs, bisim_obs=None):
        assert obs is not None
        v = self.G.add_vertex()
        self.G.vp.value[v] = 0
        self.G.vp.upper_value[v] = 0
        self.G.vp.real_node[v] = False
        self.G.vp.terminal[v] = False
        self.G.vp.obs[v] = obs
        self.vertices[obs.tobytes()] = self.unique_obs.shape[0]
        self.unique_obs = np.concatenate((self.unique_obs, obs[None, ...]))
        if self.use_bisimulation:
            self.neighbor_obs = np.concatenate((self.neighbor_obs, bisim_obs[None, ...]))
        elif self.normalize_obs:
            norm_obs = (obs - self.mean) / self.std
            self.neighbor_obs = np.concatenate((self.neighbor_obs, norm_obs[None, ...]))
        else:
            self.neighbor_obs = np.concatenate((self.neighbor_obs, obs[None, ...]))

        return self.G.vertex_index[v]

    def get_middle_obs(self, start_obs, actions):
        start_obs = torch.Tensor(start_obs)[None, ...]
        actions = torch.Tensor(actions)[None, ...]
        model_obs, model_actions, model_rewards, model_terminals = self.dynamics_unroller.model_unroll(start_obs,
                                                                                                       actions)
        if model_terminals.any():
            return None, None
        return model_obs.mean(axis=1)[0, ...], model_rewards.mean(axis=1)[0, ...]

    def add_edges(self, edges_to_add, iteration):
        added = 0
        for start, end, actions, distance, rewards, obs_history, model_errs in edges_to_add:
            start_obs = self.G.vp.obs[start]
            middle_obs, dynamics_rewards = self.get_middle_obs(start_obs, actions)
            if middle_obs is None:
                continue
            end_v = None
            for i, action in enumerate(actions):
                if i == 0:
                    start_v = start
                else:
                    start_v = end_v
                if i + 1 == len(actions):
                    end_v = end
                else:
                    end_obs = obs_history[i, :]
                    if self.use_bisimulation:
                        end_v = self.add_vertex(middle_obs[i, :], end_obs)
                        # need to get the real obs and bisim obs from somewhere and pass them
                    else:
                        end_v = self.add_vertex(end_obs)
                if self.G.vp.terminal[start] or self.G.edge(start, end) is not None:
                    break
                e = self.G.add_edge(start_v, end_v)
                self.edges_added.append((start_v, end_v))
                self.G.ep.action[e] = action
                self.G.ep.imagined[e] = True
                reward = rewards[i]
                self.G.ep.model_errors[e] = model_errs
                self.G.ep.stitch_itr[e] = iteration
                self.G.ep.stitch_length[e] = actions.shape[0]
                if self.penalize_stitches:
                    if i + 1 != len(actions):
                        self.G.ep.reward[e] = reward
                        self.G.ep.upper_reward[e] = reward
                    else:
                        self.G.ep.reward[e] = reward - distance * self.gamma * self.penalty_coefficient
                        self.G.ep.upper_reward[e] = reward + distance * self.gamma * self.penalty_coefficient
                else:
                    self.G.ep.reward[e] = reward
            added += 1
        return added

    def add_dataset_edges(self):
        if self.graph_stitching_done:
            print("skipping adding dataset edges")
            return
        print(f"Adding {self.dataset_size} initial edges to our graph")
        iterator = self.get_iterator(self.dataset_size)
        for i in iterator:
            obs = self.dataset['observations'][i, :]
            next_obs = self.dataset['next_observations'][i, :]
            v_from = self.get_vertex(obs)
            v_to = self.get_vertex(next_obs)
            if (self.G.vertex_index[v_from], self.G.vertex_index[v_to]) in self.stitches_tried:  # NOQA
                continue
            action = self.dataset['actions'][i, :]
            reward = self.dataset['rewards'][i]
            terminal = self.dataset['terminals'][i]
            v_from = self.get_vertex(obs)
            v_to = self.get_vertex(next_obs)
            self.stitches_tried.add((self.G.vertex_index[v_from], self.G.vertex_index[v_to]))
            e = self.G.add_edge(v_from, v_to)
            self.G.ep.action[e] = action.tolist()  # not sure if the tolist is needed
            self.G.ep.reward[e] = reward
            self.G.ep.upper_reward[e] = reward
            self.G.ep.imagined[e] = False
            self.G.vp.terminal[v_to] = terminal
            # This is hardcoded to assume there are 5 models.
            self.G.ep.model_errors[e] = [0 for _ in range(5)]
            self.G.ep.stitch_itr[e] = 0
        start_nodes_dense = get_starts_from_graph(self.G, self.env, self.env_name)
        self.G.vp.start_node.get_array()[start_nodes_dense] = 1

    def find_nearest_neighbors(self):
        '''
        saves a sparse matrix containing the nearest neighbors graph over neighbor_obs (which could be latent space
        or state space)
        '''
        if self.neighbors is not None or self.graph_stitching_done:
            print(f'skipping the nearest neighbors step, loaded {self.neighbors.nnz} neighbors')
            return
        print("finding possible neighbors")
        # this is the only step with quadratic time complexity, watch out for how long it takes
        start = time.time()
        p = 1 if self.use_bisimulation else 2
        size = self.G.num_vertices()
        if self.k_neighbors is None:
            self.neighbors = radius_neighbors_graph(self.neighbor_obs, self.epsilon_neighbors, p=p).astype(bool)
        else:
            self.neighbors = kneighbors_graph(self.neighbor_obs, self.k_neighbors, p=p).astype(bool)
        self.neighbors.resize((size, size))
        print(f"Time to find possible neighbors: {time.time() - start:.2f}s")
        print(f"Found {self.neighbors.nnz // 2} neighbor pairs")
        save_npz(self.output_dir / self.neighbor_name, self.neighbors)
        return self.neighbors.nnz // 2

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
        pbar = trange(self.max_val_iterations)
        reward_mat = adjacency(self.G, weight=self.G.ep.reward)
        upper_reward_mat = None
        if self.penalize_stitches:
            upper_reward_mat = adjacency(self.G, weight=self.G.ep.upper_reward)
        adjmat = adjacency(self.G)
        for i in pbar:
            # Construct sparse adjacency, reward, sparse value matrices.
            target_val = diags(
                (self.gamma * self.G.vp.value.get_array()
                            * (1 - self.G.vp.terminal.get_array())),
                format='csr',
            )
            # WARNING: graph-tool returns transpose of standard adjacency matrix
            #          hence the line target_val * adjmat (instead of reverse).
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
            old_values = self.G.vp.value.get_array()
            lower_bellman_error = bellman_error = np.max(np.square(values - old_values))
            if self.penalize_stitches:
                upper_target_val = diags(
                        (self.gamma * self.G.vp.upper_value.get_array()
                                    * (1 - self.G.vp.terminal.get_array())),
                        format='csr',
                )
                upper_target_mat = upper_target_val * adjmat
                upper_qs = upper_reward_mat + upper_target_mat
                upper_values = np.asarray(
                        upper_qs[bst_childs, np.arange(self.G.num_vertices())]).flatten()
                old_upper_values = self.G.vp.upper_value.get_array()
                upper_bellman_error = np.max(np.square(upper_values - old_upper_values))
                pbar.set_description(f"{lower_bellman_error=:.3f}, {upper_bellman_error=:.3f}")
                bellman_error = max(bellman_error, upper_bellman_error)
                upper_values[is_dead_end] = 0
                self.G.vp.upper_value.a = upper_values
            else:
                pbar.set_description(f"{bellman_error=:.3f}")
            values[is_dead_end] = 0
            bst_childs[is_dead_end] = -1
            self.G.vp.best_neighbor.a = bst_childs
            self.G.vp.value.a = values
            if bellman_error < self.vi_tolerance:
                break
        pbar.close()

        self.add_stat("Bellman Max Error", bellman_error)
        start_value = np.mean(values[self.start_states])
        self.add_stat("Mean Start Value", start_value)
        mean_value = np.mean(values)
        self.add_stat("Mean Value", mean_value)
        min_value = np.min(values)
        self.add_stat("Min Value", min_value)
        max_value = np.max(values)
        self.add_stat("Max Value", max_value)
        if len(self.stats['Mean Start Value']) > 1:
            change = start_value - self.stats['Mean Start Value'][-2]
            print(f'Change in Mean Start Value: {change:.2f}')
        if self.penalize_stitches:
            start_value = np.mean(upper_values[self.start_states])
            self.add_stat("Upper Mean Start Value", start_value)
            mean_value = np.mean(upper_values)
            self.add_stat("Upper Mean Value", mean_value)
            min_value = np.min(upper_values)
            self.add_stat("Upper Min Value", min_value)
            max_value = np.max(upper_values)
            self.add_stat("Upper Max Value", max_value)
            if len(self.stats['Upper Mean Start Value']) > 1:
                change = start_value - self.stats['Upper Mean Start Value'][-2]
                print(f'Change in Upper Mean Start Value: {change:.2f}')

    def train_bc(self, dir_name=None, intermediate=False):
        print("cloning a policy")
        data, val_data, stats = make_boltzmann_policy_dataset(
                graph=self.G,
                n_collects=self.G.num_vertices(),
                max_ep_len=self.env._max_episode_steps,
                n_val_collects=self.bolt_gather_params['val_start_prop']
                               * self.G.num_vertices(),
                starts=self.start_states,
                **self.bolt_gather_params)
        for k, v in stats.items():
            self.add_stat(k, v)
        params = deepcopy(self.intermediate_bc_params if intermediate
                          else self.bc_params)
        if dir_name is not None:
            params['save_dir'] = os.path.join(params['save_dir'], dir_name)
        self.policy, bc_trainer = behavior_clone(
                dataset=data,
                val_dataset=val_data,
                env=self.env,
                max_ep_len=self.env._max_episode_steps,
                **params
        )
        bc_stats = bc_trainer.get_stats()
        self.add_stat('avg_return', bc_stats['Returns/avg'][-1])

    def get_rollout_stitch_chunk(self):
        # need to be less than rollout_chunk_size

        chunksize = self.rollout_chunk_size // self.num_cpus
        output_path = self.output_dir / 'rollout_output'
        output_path.mkdir(exist_ok=True)
        processes = []
        print("Getting possible stitches by rolling out best Boltzmann policy")
        for i in range(self.num_cpus):
            output_file = output_path / f"{i}.pkl"
            args = ['python',
                    'rollout_stitches.py',
                    str(self.output_dir / 'mdp.gt'),
                    str(self.output_dir / self.neighbor_name),
                    str(self.output_dir / 'stitches_tried.pkl'),
                    str(self.start_state_path),
                    str(output_file),
                    str(self.action_dim),
                    str(self.obs_dim),
                    str(self.latent_dim),
                    str(chunksize),
                    str(self.temperature),
                    str(self.gamma),
                    str(self.max_stitches),
                    str(self.max_stitch_length),
                    ]
            if self.use_bisimulation:
                args.append('-ub')
            if not self.pick_positive_adv:
                args.append('-ppa')
            if self.verbose and i == 0:
                print(' '.join(args))
            process = Popen(args)
            processes.append(process)
        all_advantages = []
        all_stitches = []
        for i, process in enumerate(processes):
            process.wait()
        for i in range(len(processes)):
            output_file = output_path / f"{i}.pkl"
            with output_file.open('rb') as f:
                stitches, advantages = pickle.load(f)
            all_advantages.append(advantages)
            all_stitches += stitches
        all_advantages = np.concatenate(all_advantages, axis=0)
        stitch_array = np.array([[s[0], s[1]] for s in all_stitches])
        indices = np.unique(stitch_array, return_index=True, axis=0)[1]
        unique_advantages = all_advantages[indices]
        if len(unique_advantages) > self.rollout_chunk_size:
            start_advantages = np.argpartition(unique_advantages, len(unique_advantages) -
                                               self.rollout_chunk_size)[-self.rollout_chunk_size:]
            indices = indices[start_advantages]
            advantages = unique_advantages[start_advantages]
        else:
            advantages = unique_advantages
        unique_stitches = []
        for idx in indices:
            unique_stitches.append(all_stitches[idx])
        stitches = unique_stitches
        stitch_array = stitch_array[indices, ...]
        self.remove_neighbors(stitch_array)
        self.add_stat('Rollout Stitches', len(advantages))
        print(f'Choosing {len(indices)} edges from Boltzmann rollouts')
        return stitches

    def remove_neighbors(self, stitches_to_try):
        for stitch in stitches_to_try:
            self.stitches_tried.add(tuple(stitch[:2]))
        neighbors_tried_path = self.output_dir / 'stitches_tried.pkl'
        with open(neighbors_tried_path, 'wb') as f:
            pickle.dump(self.stitches_tried, f)
