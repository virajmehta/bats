from run import main, parse_arguments
import pickle
import numpy as np
from tqdm import trange
from functools import partial
from util import make_output_dir
# from util import suppress_stdout
from copy import deepcopy

NUM_TRIALS = 8
ITERS_PER_TRIAL = 10
DUMP_FN = 'trials.pkl'


class HPConfig:
    def __init__(self, config=None, rep=None):
        self.config = config

    def __getitem__(self, key):
        return self.config['key']

    def __str__(self):
        return f"\n".join([f"{k}: {v:.3f}" for k, v in self.config.items()])

    def __setitem__(self, key, value):
        self.config[key] = value

    def items(self):
        return self.config.items()


def training_function(config, args):
    args = deepcopy(args)
    suffix_list = []
    for k, v in config.items():
        args.__setattr__(k, v)
        suffix_list.append(f"{k}:{v:.3f}")
    args.__setattr__('num_stitching_iters', ITERS_PER_TRIAL)
    args.name = args.name + "_" + ",".join(suffix_list)
    return main(args)


def sample_config(config_space):
    config = {k: np.random.uniform(*v) for k, v in config_space.items()}
    return HPConfig(config)


def hp_main(args):
    output_dir = make_output_dir(args.name, args.overwrite, args)
    output_file = output_dir / DUMP_FN
    train_fn = partial(training_function, args=args)
    # for now just doing a uniform distribution over these
    best_return = -np.inf
    best_config = None
    config_space = {
            'epsilon_planning': (0.05, 10),
            'planning_quantile': (0.4, 1.),
            'epsilon_neighbors': (0.1, 0.3),
            }
    all_results = {}
    for i in trange(NUM_TRIALS):
        config = sample_config(config_space)
        print(f'Attempting Config \n{config}')
        returns = train_fn(config)
        if returns is None:
            continue
        print(f"Returns for Config {config}:")
        print(returns)
        best_iter = np.argmax(returns)
        current_best_return = returns[best_iter]
        if current_best_return > best_return:
            best_return = current_best_return
            best_config = config
            best_config['num_stitching_iterations'] = best_iter
        print(f"Current best config: {best_config}")
        print(f"Return: {best_return}")
        all_results[str(config)] = returns
        with output_file.open('wb') as f:
            pickle.dump(all_results, f)


if __name__ == '__main__':
    args = parse_arguments()
    hp_main(args)
