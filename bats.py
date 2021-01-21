from graph_tool import Graph
import util
import time
import numpy as np
from tqdm import trange, tqdm
from modelling.dynamics_construction import train_ensemble
from modelling.policy_construction import train_policy
from sklearn.neighbors import radius_neighbors_graph


class BATSTrainer:
    def __init__(self, dataset, env, output_dir, **kwargs):
        self.dataset = dataset
        self.env = env
        self.output_dir = output_dir
        self.gamma = kwargs.get('gamma', 0.99)
        self.tqdm = kwargs.get('tqdm', True)
        self.n_val_iterations = kwargs.get('n_val_iterations', 100)
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
        self.epsilon_neighbors = kwargs.get('epsilon_neighors', 0.5)  # no idea what this should be
        # set up graph
        self.G = Graph()
        # add a vertex for each state
        self.G.add_vertex(self.graph_size)
        # make sure we keep the mapping from states to vertices
        self.vertices = {obs.tobytes(): i for i, obs in enumerate(self.unique_obs)}
        # the values associated with each node
        self.G.vp.value = self.G.new_vertex_property("float")
        self.G.vp.value.get_array()[:] = 0
        self.G.vp.best_neighbor = self.G.new_vertex_property("int")
        # the actions are gonna be associated with each edge
        self.G.ep.action = self.G.new_edge_property("vector<float>")
        # we also associate the rewards with each edge
        self.G.ep.reward = self.G.new_edge_property("float")

        # parameters for planning
        self.epsilon_planning = kwargs.get('epsilon_planning', 0.1)  # also no idea what this should be
        self.planning_quantile = kwargs.get('planning_quantile', 0.8)

        # parameters for evaluation
        self.num_eval_episodes = kwargs.get("num_eval_episodes", 20)

    def get_vertex(self, obs):
        return self.G.vertex(self.vertices[obs.tobytes()])

    def get_iterator(self, n):
        if self.tqdm:
            return trange(n)
        else:
            return range(n)

    def train(self):
        self.add_dataset_edges()
        start = time.time()
        possible_neighbors = self.find_possible_neighbors()
        print(f"Time: {time.time() - start}")
        self.train_dynamics()
        self.add_neighbor_edges(possible_neighbors)
        self.value_iteration()
        self.G.save(str(output_dir / 'mdp.gt'))
        self.train_bc()
        self.evaluate()

    def train_dynamics(self):
        print("training ensemble of dynamics models")
        self.dynamics_ensemble = train_ensemble(self.dataset, **self.dynamics_train_params)

    def add_neighbor_edges(self, possible_neighbors):
        '''
        '''
        print('testing possible neighbors')
        edges_to_add = []
        for row in possible_neighbors:
            adding_neighbor = row[0]
            receiving_neighbor = row[1]
            receiving_obs = self.unique_obs[receiving_neighbor, :]
            for start, end, action in self.G.iter_out_edges(adding_neighbor, eprops=[self.G.ep.action]):
                target_obs = self.unique_obs[end, :]
                # need to run CEM to go from recieving_obs to target_obs with action initialized at action
                # could also add kwargs to adjust the CEM parameters like max_iters, popsize, etc.
                best_action, predicted_reward = util.CEM(receiving_obs,
                                                         target_obs,
                                                         np.array(action),
                                                         self.dynamics_ensemble,
                                                         self.epsilon_planning,
                                                         self.planning_quantile)
                if best_action is not None:
                    edges_to_add.append((receiving_neighbor, end, best_action, predicted_reward))
        print(f"adding {len(edges_to_add)} edges to graph")
        for start, end, action, reward in edges_to_add:
            e = self.G.add_edge(start, end)
            self.G.ep.action[e] = action
            self.G.ep.reward[e] = reward

    def add_dataset_edges(self):
        print(f"Adding {self.dataset_size} initial edges to our graph")
        iterator = self.get_iterator(self.dataset_size)
        for i in iterator:
            obs = self.dataset['observations'][i, :]
            next_obs = self.dataset['next_observations'][i, :]
            action = self.dataset['actions'][i, :]
            reward = self.dataset['rewards'][i]
            v_from = self.get_vertex(obs)
            v_to = self.get_vertex(next_obs)
            e = self.G.add_edge(v_from, v_to)
            self.G.ep.action[e] = action.tolist()
            self.G.ep.reward[e] = reward

    def find_possible_neighbors(self):
        print("finding possible neighbors")
        neighbors = radius_neighbors_graph(self.unique_obs, self.epsilon_neighbors, n_jobs=-1)
        possible_neighbors = np.column_stack(neighbors.nonzero())
        print(f"found {possible_neighbors.shape[0] // 2} possible neighbors")
        return possible_neighbors

    def value_iteration(self):
        print("performing value iteration")
        iterator = self.get_iterator(self.n_val_iterations)
        for _ in iterator:
            for v in self.G.iter_vertices():
                # should be a (num_neighbors, 2) ndarray where the first col is indices and second is values
                neighbors = self.G.get_out_neighbors(v, vprops=[self.G.vp.value])
                if len(neighbors) == 0:
                    # the state has no actions in our graph
                    self.G.vp.value[v] = 0
                    self.G.vp.best_neighbor[v] = -1
                    continue
                values = neighbors[:, 1]
                # should be a (num_neighbors, 3) ndarray where the first col is v second col is indices, third is reward
                edges = self.G.get_out_edges(v, eprops=[self.G.ep.reward])
                rewards = edges[:, 2]
                backups = rewards + self.gamma * values
                best_arm = np.argmax(backups)
                value = backups[best_arm]
                self.G.vp.value[v] = value
                self.G.vp.best_neighbor[v] = neighbors[best_arm, 0]

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
        print("evaluating cloned policy")
        episodes = []
        returns = []
        for i in self.get_iterator(self.num_eval_episodes):
            episode = util.rollout(self.policy, self.env)
            episodes.append(episode)
            ep_return = util.get_return(episode)
            returns.append(ep_return)
        return_mean = np.mean(returns)
        return_std = np.std(returns)
        tqdm.write(f"Mean Return | {return_mean:.2f} | Std Return | {return_std:.2f}")
