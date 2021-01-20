import argparse
import util
from bats import BATSTrainer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help="The name of the experiment and output directory.")
    parser.add_argument('--env_name', default="HalfCheetahMedium-v0", help="The name of the environment (will be checked at runtime for correctness).")  # NOQA
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--policy_layer_sizes', default="512|512")
    parser.add_argument('--model_layer_sizes', default="512|512")
    parser.add_argument('--num_bootstrap', type=int, default=5)
    parser.add_argument('-ow', dest='overwrite', action='store_true')
    parser.add_argument('-notqdm', dest="tqdm", action="store_false")
    return parser.parse_args()


def main(args):
    output_dir = util.make_output_dir(args.name, args.overwrite, args)
    env, dataset = util.get_offline_env(args.env_name)
    flat_dataset = util.flatten_dataset(dataset, tqdm=args.tqdm)
    args = vars(args)
    bats = BATSTrainer(flat_dataset, env, output_dir, **args)
    bats.train()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
