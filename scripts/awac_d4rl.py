"""
Script for doing AWAC on a d4rl dataset.
"""
import argparse

import d4rl
import h5py
import gym
import numpy as np

from modelling.policy_construction import learn_awac_policy


def train(args):
    if args.pudb:
        import pudb; pudb.set_trace()
    env = gym.make(args.env)
    if args.dataset_path is None:
        dataset = d4rl.qlearning_dataset(gym.make(args.env))
    else:
        dataset = {}
        with h5py.File(args.dataset_path, 'r') as hdata:
            for k, v in hdata.items():
                dataset[k] = v[()]
    # dataset['weights'] = np.minimum(np.exp(dataset['rewards'] - 6), 20)
    train_params = vars(args)
    train_params['env'] = env
    del train_params['pudb']
    del train_params['dataset_path']
    learn_awac_policy(dataset, **train_params)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path')
    parser.add_argument('--env')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    train(parse_args())
