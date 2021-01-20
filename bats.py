from graph_tool import Graph
import numpy as np
from tqdm import trange
from modelling.dynamics_construction import train_ensemble


class BATSTrainer:
    def __init__(self, dataset, env, output_dir, **kwargs):
        self.dataset = dataset
        self.env = env
        self.gamma = kwargs.get('gamma', 0.99)
        self.tqdm = kwargs.get('tqdm', True)
        self.n_val_iterations = kwargs.get('n_val_iterations', 100)
        all_obs = np.concatenate((dataset['observations'], dataset['next_observations']))
        self.unique_obs = np.unique(all_obs, axis=0)
        self.graph_size = self.unique_obs.shape[0]
        self.dataset_size = self.dataset['observations'].shape[0]

        # set up the parameters for the dynamics model training
        self.dynamics_train_params = {}
        self.dynamics_train_params['n_members'] = kwargs.get('dynamics_n_members', 7)
        self.dynamics_train_params['n_elites'] = kwargs.get('dynamics_n_elites', 5)
        self.dynamics_train_params['save_dir'] = str(output_dir)
        self.dynamics_train_params['epochs'] = kwargs.get('dynamics_epochs', 100)
        self.dynamics_train_params['cuda_device'] = kwargs.get('cuda_device', '')
        # could do it this way or with knn, this is simpler to implement for now
        self.epsilon_neighbors = kwargs.get('epsilon_neighors', 0.1)
        # set up graph
        self.G = Graph()
        # add a vertex for each state
        self.G.add_vertex(self.graph_size)
        # make sure we keep the mapping from states to vertices
        self.vertices = {obs.tobytes(): i for i, obs in enumerate(self.unique_obs)}
        # the values associated with each node
        self.G.vp.value = self.G.new_vertex_property("float")
        self.G.vp.get_array()[:] = 0
        self.G.vp.best_neighbor = self.G.new_vertex_property("int")
        # the actions are gonna be associated with each edge
        self.G.ep.action = self.G.new_edge_property("vector<float>")
        # we also associate the rewards with each edge
        self.G.ep.reward = self.G.new_edge_property("float")

    def get_vertex(self, obs):
        return self.G.vertex(self.vertices[obs.tobytes()])

    def get_iterator(self, n):
        if self.tqdm:
            return trange(n)
        else:
            return range(n)

    def train(self):
        self.add_dataset_edges()
        self.train_dynamics()
        possible_neighbors = self.find_possible_neighbors()
        self.add_neighbor_edges(possible_neighbors)
        self.value_iteration()
        self.train_bc()
        self.evaluate()

    def train_dynamics(self):
        train_ensemble(self.dataset, **self.dynamics_train_params)

    def add_neighbor_edges(self):
        raise NotImplementedError()

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
            self.e_action[e] = action.tolist()
            self.e_reward[e] = reward

    def find_possible_neighbors(self):
        pairwise_distances = np.sqrt(np.sum((self.unique_obs[None, :] - self.unique_obs[:, None]) ** 2, axis=-1))
        possible_neighbors = pairwise_distances < self.epsilon_neighbors
        possible_neighbors_list = np.argwhere(possible_neighbors)
        nonduplicates = possible_neighbors_list[:, 0] != possible_neighbors_list[:, 1]
        nonduplicate_possible_neighbors_list = np.compress(nonduplicates, possible_neighbors_list, axis=0)
        return nonduplicate_possible_neighbors_list

    def value_iteration(self):
        iterator = self.get_iterator(self.n_val_iterations)
        for _ in iterator:
            for v in self.G.iter_vertices():
                # should be a (num_neighbors, 2) ndarray where the first col is indices and second is values
                neighbors = self.G.get_out_neighbors(v, vprops=[self.G.vp.value])
                edges = self.G.get_out_edges(v, eprops=[self.G.ep.reward])
                rewards = edges[:, 1]
                values = neighbors[:, 1]
                backups = rewards + self.gamma * values
                best_arm = np.argmax(backups)
                value = backups[best_arm]
                self.G.vp.value[v] = value
                self.G.vp.best_neighbor[v] = neighbors[best_arm, 0]

    def train_bc(self):
        actions = []
        for v in self.G.iter_vertices():
            best_neighbor = self.G.best_neighbor[v]
            e = self.G.edge(v, best_neighbor)
            action = np.array(self.G.ep.action[e])
            actions.append(action)
        actions = np.stack(actions)
        # TODO: we have self.unique_observations and actions, just need to behavior-clone a policy now
        raise NotImplementedError()

    def evaluate():
        # we have util.rollout ready for this purpose
        raise NotImplementedError()
