from graph_tool import Graph
import numpy as np
from tqdm import trange


class BATSTrainer:
    def __init__(self, dataset, env, output_dir, **kwargs):
        self.dataset = dataset
        self.env = env
        self.gamma = kwargs.get('gamma', 0.99)
        self.tqdm = kwargs.get('tqdm', True)
        all_obs = np.concatenate((dataset['observations'], dataset['next_observations']))
        unique_obs = np.unique(all_obs, axis=0)
        self.graph_size = unique_obs.shape[0]
        self.dataset_size = self.dataset['observations'].shape[0]
        # set up graph
        self.G = Graph()
        # add a vertex for each state
        self.G.add_vertex(self.graph_size)
        # make sure we keep the mapping from states to vertices
        self.vertices = {obs.tobytes(): i for i, obs in enumerate(unique_obs)}
        # the values associated with each node
        self.v_value = self.G.new_vertex_property("float")
        self.v_value.get_array()[:] = 0
        # the actions are gonna be associated with each edge
        self.e_action = self.G.new_edge_property("vector<float>")
        # we also associate the rewards with each edge
        self.e_reward = self.G.new_edge_property("float")

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
        raise NotImplementedError()

    def value_iteration(self):
        raise NotImplementedError()

    def train_bc():
        raise NotImplementedError()

    def evaluate():
        raise NotImplementedError()
