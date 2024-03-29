"""
Script for doing behavior cloning on a d4rl dataset.
"""
import argparse

import d4rl
import h5py
import gym
import numpy as np

from modelling.policy_construction import behavior_clone


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
    # dataset['weights'] = np.minimum(np.exp(dataset['rewards'] - 6), 20)
    train_params = vars(args)
    train_params['env'] = gym.make(args.env)
    del train_params['pudb']
    del train_params['dataset_path']
    behavior_clone(dataset, **train_params)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir')
    parser.add_argument('--env')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--od_wait', type=int, default=None)
    parser.add_argument('--val_size', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--batch_updates_per_epoch', type=int)
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    train(parse_args())
