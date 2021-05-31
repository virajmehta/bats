"""
Script to train a dynamics ensemble of a d4rl dataset.
"""
import argparse

import d4rl
from pathlib import Path
import h5py
import gym

from modelling.dynamics_construction import train_ensemble


def train(args):
    if args.pudb:
        import pudb; pudb.set_trace()
    args.save_dir.mkdir(exist_ok=True)
    if args.data_path:
        dataset = {}
        with h5py.File(str(args.data_path), 'r') as hdata:
            for k, v in hdata.items():
                if k == 'infos':
                    continue
                dataset[k] = v[()]
    elif args.env:
        dataset = d4rl.qlearning_dataset(gym.make(args.env))
    train_params = vars(args)
    del train_params['data_path']
    del train_params['pudb']
    del train_params['env']
    model_params = {}
    model_params['encoder_hidden'] = train_params.pop('encoder_hidden')
    model_params['latent_dim'] = train_params.pop('latent_dim')
    train_params['model_params'] = model_params
    train_ensemble(dataset, **train_params)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=Path)
    parser.add_argument('--data_path')
    parser.add_argument('--env')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--od_wait', type=int, default=25)
    parser.add_argument('--n_members', type=int, default=7)
    parser.add_argument('--n_elites', type=int, default=5)
    parser.add_argument('--val_size', type=float, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--encoder_hidden', default='200,200,200')
    parser.add_argument('--latent_dim', type=int, default=200)
    parser.add_argument('--model_type', type=str, default='PNN',
                        choices=['PNN', 'NN'])
    parser.add_argument('--dataset_path')
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    train(parse_args())
