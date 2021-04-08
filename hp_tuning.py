from run import main, parse_arguments
from functools import partial
from copy import deepcopy


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
    raise NotImplementedError()


if __name__ == '__main__':
    args = parse_arguments()
    hp_main(args)
