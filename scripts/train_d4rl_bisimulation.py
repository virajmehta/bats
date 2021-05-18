"""
Script to train bisimulation on d4rl.
"""
import argparse

import d4rl
from pathlib import Path
import h5py
import gym

from modelling.bisim_construction import train_bisim


def train(args):
    if args.pudb:
        import pudb; pudb.set_trace()
    if args.data_path is not None:
        dataset = {}
        with h5py.File(args.data_path, 'r') as hdata:
            for k, v in hdata.items():
                dataset[k] = v[()]
    else:
        dataset = d4rl.qlearning_dataset(gym.make(args.env))
    args.save_dir.mkdir(exist_ok=True)
    train_params = vars(args)
    train_params['bisim_params'] = {
            'encoder_architecture': args.encoder_architecture,
    }
    del train_params['env']
    del train_params['pudb']
    del train_params['data_path']
    del train_params['encoder_architecture']
    train_bisim(dataset=dataset, **train_params)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=Path)
    parser.add_argument('--env')
    parser.add_argument('--data_path')
    parser.add_argument('--latent_dim', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=-1)
    parser.add_argument('--od_wait', type=int)
    parser.add_argument('--batch_updates_per_epoch', type=int)
    parser.add_argument('--validation_batches_per_epoch', type=int, default=50)
    parser.add_argument('--n_members', type=int, default=5)
    parser.add_argument('--encoder_architecture', default='64,64')
    parser.add_argument('--val_size', type=float, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    train(parse_args())
