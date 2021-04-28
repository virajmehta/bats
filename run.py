import argparse
# due to some weirdness with dynamic libraries graph_tool needs to be imported before torch
# don't change the order of the next 2 lines
from bats import BATSTrainer
import util
from pathlib import Path
from copy import deepcopy

from configs import CONFIGS


def parse_arguments():
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument('--config')
    config_arg, remaining = config_parser.parse_known_args()
    defaults = None
    if config_arg.config is not None:
        defaults = CONFIGS[config_arg.config]
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help="The name of the experiment and output directory.")
    parser.add_argument('--env', dest='env_name', default="halfcheetah-medium-v1", help="The name of the environment (will be checked at runtime for correctness).")  # NOQA
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('-ow', dest='overwrite', action='store_true')
    parser.add_argument('-notqdm', dest="tqdm", action="store_false")
    parser.add_argument('--cuda_device', default='')
    parser.add_argument('-ep', '--epsilon_planning', type=float, default=0.05, help="The threshold for the planning to add graph edges")  # NOQA
    parser.add_argument('-pq', '--planning_quantile', type=float, default=0.8, help="The quantile of the dynamics ensemble used for planning to add graph edges")  # NOQA
    parser.add_argument('-en', '--epsilon_neighbors', type=float, default=0.1, help="The threshold for two states to be considered possible neighbors in an MDP")  # NOQA
    parser.add_argument('-kn', '--k_neighbors', type=int, default=None, help="The K to use for KNN for our neighbor-finding. Overrides epsilon_neighbors")
    parser.add_argument('--dataset_fraction', type=float, default=1., help="The fraction of the offline dataset to use for the algorithm. Useful for testing on smaller data")  # NOQA
    parser.add_argument('-lm', '--load_model', type=Path, default=None, help="Load a dynamics model ensemble from this directory")
    parser.add_argument('-lbm', '--load_bisim_model', type=Path, default=None, help="Load a dynamics model ensemble from this directory")
    parser.add_argument('-lg', '--load_graph', type=Path, default=None, help="Load a graph pre-value iteration from this directory")
    parser.add_argument('-lvi', '--load_value_iteration', type=Path, default=None, help="Load a graph after value iteration from this directory.")
    parser.add_argument('-ln', '--load_neighbors', type=Path, default=None, help="Load nearest neighbors")
    parser.add_argument('-lp', '--load_policy', type=Path, default=None, help="Load a behavior cloned policy from this directory")
    parser.add_argument('-ncpus', '--num_cpus', type=int, default=1)
    parser.add_argument('-odp', '--offline_dataset_path', type=Path, default=None, help='Path for dataset, will use d4rl dataset if none is provided.')
    parser.add_argument('-scs', '--stitching_chunk_size', type=int, default=100000, help='number of stitches to attempt in an iteration of stitching')
    parser.add_argument('-ni', '--num_stitching_iters', type=int, default=50, help='number of iterations of stitching to do')
    parser.add_argument('-tvi', '--vi_tolerance', type=float, default=0.02)
    parser.add_argument('-mvi', '--max_val_iterations', type=int, default=1000, help='max number of iterations of value iterations to do during each stitching iter')
    parser.add_argument('-bei', '--bc_every_iter', action='store_true')
    parser.add_argument('-ms', '--max_stitches', type=int, default=6, help='max stitches for a single state as the boltzmann rollouts proceed')
    parser.add_argument('-norm', '--normalize_obs', action='store_true')
    parser.add_argument('--pudb', action='store_true')
    parser.add_argument('-ub', '--use_bisimulation', action='store_true')
    parser.add_argument('--bisim_latent_dim', type=int, default=6, help="How many dimensions for the latent space of the bisimulation metric")
    parser.add_argument('-p', '--penalize_stitches', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-pc', '--penalty_coefficient', type=float, default=1.0)
    parser.add_argument('-msl', '--max_stitch_length', type=int, default=1)
    if defaults is not None:
        parser.set_defaults(**defaults)
    return parser.parse_args(remaining)


def main(args):
    output_dir = util.make_output_dir(args.name, args.overwrite, deepcopy(args))
    env, dataset = util.get_offline_env(args.env_name,
                                        args.dataset_fraction,
                                        data_path=args.offline_dataset_path)
    args = vars(args)
    bats = BATSTrainer(dataset, env, output_dir, **args)
    bats.train()
    return bats.stats


if __name__ == '__main__':
    args = parse_arguments()
    if args.pudb:
        import pudb; pudb.set_trace()
    main(args)
