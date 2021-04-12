from run import main, parse_arguments
import sherpa
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
    stats = main(args)
    bc_returns = stats.pop('Behavior Clone Return')
    print(f"Returns for Config {config}:")
    print(bc_returns)
    outputs = []
    for i, ret in enumerate(bc_returns):
        context = {k: v[i] for k, v in stats.items()}
        outputs.append((ret, context))
    return outputs


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
    parameters = [sherpa.Continuous(name='epsilon_planning', range=[0.05, 10], scale='linear'),
                  sherpa.Continuous(name='planning_quantile', range=[0.4, 1], scale='linear'),
                  sherpa.Continuous(name='epsilon_neighbors', range=[0.1, 0.3], scale='linear')]
    algorithm = sherpa.RandomSearch(max_num_trials=NUM_TRIALS)
    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=False)
    for trial in study:
        config = HPConfig(config=trial.parameters)
        print(f'Attempting Config \n{config}')
        outputs = train_fn(config)
        for ret, ctx in outputs:
            # TODO
            pass


if __name__ == '__main__':
    args = parse_arguments()
    hp_main(args)
