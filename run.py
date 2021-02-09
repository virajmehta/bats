import argparse
# due to some weirdness with dynamic libraries graph_tool needs to be imported before torch
# don't change the order of the next 2 lines
from bats import BATSTrainer
import util
from pathlib import Path
from copy import deepcopy


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help="The name of the experiment and output directory.")
    parser.add_argument('--env_name', default="halfcheetah-medium-v1", help="The name of the environment (will be checked at runtime for correctness).")  # NOQA
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('-ow', dest='overwrite', action='store_true')
    parser.add_argument('-notqdm', dest="tqdm", action="store_false")
    parser.add_argument('--cuda_device', default='')
    parser.add_argument('-ep', '--epsilon_planning', type=float, default=0.05, help="The threshold for the planning to add graph edges")  # NOQA
    parser.add_argument('-pq', '--planning_quantile', type=float, default=0.8, help="The quantile of the dynamics ensemble used for planning to add graph edges")  # NOQA
    parser.add_argument('-en', '--epsilon_neighbors', type=float, default=0.1, help="The threshold for two states to be considered possible neighbors in an MDP")  # NOQA
    parser.add_argument('--dataset_fraction', type=float, default=1., help="The fraction of the offline dataset to use for the algorithm. Useful for testing on smaller data")  # NOQA
    parser.add_argument('-lm', '--load_model', type=Path, default=None, help="Load a dynamics model ensemble from this directory")
    parser.add_argument('-lg', '--load_graph', type=Path, default=None, help="Load a graph pre-value iteration from this directory")
    parser.add_argument('-lvi', '--load_value_iteration', type=Path, default=None, help="Load a graph after value iteration from this directory.")
    parser.add_argument('-ln', '--load_neighbors', type=Path, default=None, help="Load nearest neighbors")
    parser.add_argument('-lp', '--load_policy', type=Path, default=None, help="Load a behavior cloned policy from this directory")
    parser.add_argument('-ncpus', '--num_cpus', type=int, default=1)
    parser.add_argument('-odp', '--offline_dataset_path', type=Path, default=None, help='Path for dataset, will use d4rl dataset if none is provided.')
    parser.add_argument('-scs', '--stitching_chunk_size', type=int, default=1000000, help='number of stitches to attempt in an iteration of stitching')
    parser.add_argument('-ni', '--num_stitching_iters', type=int, default=50, help='number of iterations of stitching to do')

    return parser.parse_args()


def main(args):
    output_dir = util.make_output_dir(args.name, args.overwrite, deepcopy(args))
    env, dataset = util.get_offline_env(args.env_name,
                                        args.dataset_fraction,
                                        data_path=args.offline_dataset_path)
    args = vars(args)
    bats = BATSTrainer(dataset, env, output_dir, **args)
    bats.train()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
