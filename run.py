import argparse
import util
from bats import BATSTrainer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help="The name of the experiment and output directory.")
    parser.add_argument('--env_name', default="halfcheetah-medium-v1", help="The name of the environment (will be checked at runtime for correctness).")  # NOQA
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('-ow', dest='overwrite', action='store_true')
    parser.add_argument('-notqdm', dest="tqdm", action="store_false")
    parser.add_argument('--cuda_device', default='')
    return parser.parse_args()


def main(args):
    output_dir = util.make_output_dir(args.name, args.overwrite, args)
    env, dataset = util.get_offline_env(args.env_name)
    args = vars(args)
    bats = BATSTrainer(dataset, env, output_dir, **args)
    bats.train()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
