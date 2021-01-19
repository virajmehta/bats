from graph_tool import Graph


class BATSTrainer:
    def __init__(self, dataset, env, **kwargs):
        self.dataset = dataset
        self.env = env
        self.gamma = kwargs.get('gamma', 0.99)
        self.dataset_size = dataset['observations'].shape[0]
        # set up graph
        self.G = Graph()
        # add a vertex for each state
        self.G.add_vertex(self.dataset_size)

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
        raise NotImplementedError()

    def find_possible_neighbors(self):
        raise NotImplementedError()

    def value_iteration(self):
        raise NotImplementedError()

    def train_bc():
        raise NotImplementedError()

    def evaluate():
        raise NotImplementedError()
