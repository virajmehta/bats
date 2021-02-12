"""
Behavior clone on Boltzmann policy.
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
from modelling.policy_construction import behavior_clone
from modelling.utils.graph_util import make_best_action_dataset


def run(args):
    # Learn an advantage weighting function.
    graph = load_graph(os.path.join(args.graph_dir, 'converged_graph.gt'))
    data = make_best_action_dataset(graph)
    # Run AWR with the pre-trained qnets.
    behavior_clone(
        dataset=data,
        save_dir=args.save_dir,
        hidden_sizes='128,64',
        epochs=args.epochs,
        od_wait=args.od_wait,
        val_size=args.val_size,
        cuda_device=args.cuda_device,
        env=NormalizedBoxEnv(gym.make('Pendulum-v0')),
        max_ep_len=1000,
        train_loops_per_epoch=200,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir')
    parser.add_argument('--save_dir')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--od_wait', type=int, default=50)
    parser.add_argument('--val_size', type=float, default=0)
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    run(args)
