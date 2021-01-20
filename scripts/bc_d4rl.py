"""
Script for doing behavior cloning on a d4rl dataset.
"""
import argparse

import d4rl
import gym

from modelling.policy_construction import train_policy


def train(args):
    if args.pudb:
        import pudb; pudb.set_trace()
    dataset = d4rl.qlearning_dataset(gym.make(args.env))
    train_params = vars(args)
    del train_params['env']
    del train_params['pudb']
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
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    train(parse_args())
