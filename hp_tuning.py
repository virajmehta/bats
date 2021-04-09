from run import main, parse_arguments
import numpy as np
from functools import partial
from copy import deepcopy

NUM_TRIALS = 8


def training_function(config, args):
    args = deepcopy(args)
    suffix_list = []
    for k, v in config.items():
        args.__setattr__(k, v)
        suffix_list.append(f"{k}: {v}")
    args.name = args.name + "_" + ",".join(suffix_list)
    args.__setattr__('hp_tune', True)
    return main(args)


def hp_main(args):
    train_fn = partial(training_function, args=args)
    # for now just doing a uniform distribution over these
    best_return = -np.inf
    best_config = None
    config_space = {
            'epsilon_planning': (0.05, 0.5),
            'planning_quantile': (0.4, 1.),
            'epsilon_neighbors': (1, 5),
            }
    for i in trange(NUM_TRIALS):
        config = sample_config(config_space)
        returns = train_fn(config)
        best_iter = np.argmax(returns)
        current_best_return = returns[best_iter]
        if current_best_return > best_return:
            best_return = current


if __name__ == '__main__':
    args = parse_arguments()
    hp_main(args)
