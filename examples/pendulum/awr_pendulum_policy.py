"""
Do AWR with a learned weighting function.
"""
import argparse
import os

import h5py
from graph_tool import load_graph
from graph_tool.draw import graph_draw
import gym
import numpy as np
from tqdm import tqdm

from env_wrapper import NormalizedBoxEnv
from modelling.policy_construction import supervise_critic,\
        advantage_weighted_regression, load_critic
from modelling.utils.graph_util import make_advantage_dataset,\
        make_qlearning_dataset


def run(args):
    # Learn an advantage weighting function.
    graph = load_graph(os.path.join(args.graph_dir, 'converged_graph.gt'))
    adv_data = make_advantage_dataset(graph)
    if args.advnet_path is not None:
        advnet = load_critic(args.advnet_path, 3, 1, '32,32,32',
                             args.cuda_device)
        graph = load_graph(os.path.join(args.graph_dir, 'stitched_graph.gt'))
        data = make_qlearning_dataset(graph)
    elif args.learn_advantage:
        advnet = supervise_critic(
            adv_data,
            save_dir=os.path.join(args.save_dir, 'advnet'),
            hidden_sizes='32,32,32',
            epochs=args.adv_epochs,
            od_wait=args.adv_od_wait,
            val_size=args.adv_val_size,
            cuda_device=args.cuda_device,
        )
        graph = load_graph(os.path.join(args.graph_dir, 'stitched_graph.gt'))
        data = make_qlearning_dataset(graph)
    else:
        advnet = None
        data = make_qlearning_dataset(graph)
        data['advantages'] = adv_data['advantages']
    # Run AWR with the pre-trained qnets.
    advantage_weighted_regression(
        dataset=data,
        save_dir=os.path.join(args.save_dir, 'awr'),
        hidden_sizes='128,64',
        epochs=args.awr_epochs,
        cuda_device=args.cuda_device,
        env=NormalizedBoxEnv(gym.make('Pendulum-v0')),
        max_ep_len=1000,
        train_loops_per_epoch=25,
        advnet=advnet
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir')
    parser.add_argument('--save_dir')
    parser.add_argument('--advnet_path', type=str, default=None)
    parser.add_argument('--learn_advantage', action='store_true')
    parser.add_argument('--adv_epochs', type=int, default=2500)
    parser.add_argument('--awr_epochs', type=int, default=50)
    parser.add_argument('--adv_od_wait', type=int, default=100)
    parser.add_argument('--adv_val_size', type=float, default=0.1)
    parser.add_argument('--base_graph', action='store_true')
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    run(args)
