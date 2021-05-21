"""
Script to train a dynamics ensemble of a d4rl dataset.
"""
import argparse

import d4rl
import h5py
import gym

from modelling.dynamics_construction import train_ensemble


def train(args):
    if args.pudb:
        import pudb; pudb.set_trace()
    if args.dataset_path is not None:
        dataset = {}
        with h5py.File(args.dataset_path, 'r') as hdata:
            for k, v in hdata.items():
                dataset[k] = v[()]
    else:
        dataset = d4rl.qlearning_dataset(gym.make(args.env))
    train_params = vars(args)
    enc_hidden = ','.join(args.architecture.split(',')[:-1])
    latent_dim = int(args.architecture.split(',')[-1])
    train_params['model_params'] = {
        'encoder_hidden': enc_hidden,
        'latent_dim': latent_dim,
    }
    if args.val_size > 1:
        args.val_size = int(args.val_size)
    del train_params['env']
    del train_params['pudb']
    del train_params['architecture']
    del train_params['dataset_path']
    train_ensemble(dataset, **train_params)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir')
    parser.add_argument('--env')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--od_wait', type=int, default=25)
    parser.add_argument('--n_members', type=int, default=7)
    parser.add_argument('--n_elites', type=int, default=5)
    parser.add_argument('--val_size', type=float, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--model_type', type=str, default='PNN',
                        choices=['PNN', 'NN'])
    parser.add_argument('--architecture', default='200,200,200,200')
    parser.add_argument('--dataset_path')
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    train(parse_args())
