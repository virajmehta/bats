"""
Script for doing behavior cloning on a d4rl dataset.
"""
import argparse

import d4rl
import h5py
import gym

from modelling.policy_construction import train_policy


def train(args):
    if args.pudb:
        import pudb; pudb.set_trace()
    if args.dataset_path is None:
        dataset = d4rl.qlearning_dataset(gym.make(args.env))
    else:
        dataset = {}
        with h5py.File(args.dataset_path, 'r') as hdata:
            for k, v in hdata.items():
                dataset[k] = v[()]
    train_params = vars(args)
    del train_params['env']
    del train_params['pudb']
    del train_params['dataset_path']
    train_policy(dataset, **train_params)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir')
    parser.add_argument('--env')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--od_wait', type=int, default=25)
    parser.add_argument('--val_size', type=float, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    train(parse_args())
