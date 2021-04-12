"""
Collect the boltzmann dataset and save to file
"""
import argparse
import os

import d4rl
from graph_tool import load_graph
from graph_tool.draw import graph_draw
import h5py
import gym
import numpy as np
from tqdm import tqdm

from env_wrapper import NormalizedBoxEnv
from modelling.utils.graph_util import make_boltzmann_policy_dataset
from util import get_starts_from_graph


def run(args):
    # Learn an advantage weighting function.
    env = gym.make(args.env)
    graph = load_graph(os.path.join(args.graph_dir, 'vi.gt'))
    if args.n_collects is None:
        args.n_collects = graph.num_vertices()
    if 'maze' in args.env:
        starts = get_starts_from_graph(graph, env)
    else:
        starts = None
    data, _ = make_boltzmann_policy_dataset(
            graph=graph,
            n_collects=args.n_collects,
            temperature=args.temperature,
            max_ep_len=env._max_episode_steps,
            n_val_collects=args.n_val_collects,
            val_start_prop=args.val_start_prop,
            any_state_is_start=args.use_any_start,
            starts=starts,
    )
    with h5py.File(args.save_path, 'w') as wd:
        for k, v in data.items():
            wd.create_dataset(k, data=v)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env')
    parser.add_argument('--graph_dir')
    parser.add_argument('--save_path')
    # If None, then collect as many points as there are in the dataset.
    parser.add_argument('--n_collects', type=int, default=None)
    parser.add_argument('--n_val_collects', type=int, default=0)
    parser.add_argument('--val_start_prop', type=float, default=0)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--use_any_start', action='store_true')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    run(args)
