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
from modelling.utils.graph_util import make_boltzmann_policy_dataset


def run(args):
    # Learn an advantage weighting function.
    graph = load_graph(os.path.join(args.graph_dir, 'converged_graph.gt'))
    data, val_data = make_boltzmann_policy_dataset(
            graph=graph,
            n_collects=args.n_collects,
            temperature=args.temperature,
            max_ep_len=args.max_ep_len,
            n_val_collects=args.n_val_collects,
            val_start_prop=args.val_start_prop,
            any_state_is_start=not args.use_data_starts,
    )
    # Run AWR with the pre-trained qnets.
    behavior_clone(
        dataset=data,
        save_dir=args.save_dir,
        hidden_sizes='128,64',
        epochs=args.epochs,
        od_wait=args.od_wait,
        val_dataset=val_data,
        cuda_device=args.cuda_device,
        env=NormalizedBoxEnv(gym.make('Pendulum-v0')),
        max_ep_len=1000,
        train_loops_per_epoch=1,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir')
    parser.add_argument('--save_dir')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--od_wait', type=int, default=100)
    parser.add_argument('--n_collects', type=int, default=int(1e6))
    parser.add_argument('--n_val_collects', type=int, default=0)
    parser.add_argument('--val_start_prop', type=float, default=0)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--max_ep_len', type=int, default=100)
    parser.add_argument('--use_data_starts', action='store_true')
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    run(args)
