"""
Do boltzmann cloning on a converged graph.
"""
import argparse
from copy import deepcopy
from multiprocessing import Pool
import os

import numpy as np
from scripts.collect_boltzmann_dataset import run
from util import s2f, s2i


def launch_jobs(args):
    configs = create_config_list(args)
    with Pool(args.num_jobs) as p:
        p.map(run, configs)


def create_config_list(args):
    configs = []
    for graph in os.listdir(args.target_dir):
        config = deepcopy(args)
        config.save_path = os.path.join(
                args.save_location,
                graph + '.hdf5',
        )
        config.graph_dir = os.path.join(args.target_dir, graph)
        config.silent = True
        configs.append(config)
    return configs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env')
    parser.add_argument('--target_dir')
    parser.add_argument('--save_location')
    parser.add_argument('--num_jobs', type=int ,default=1)
    # If None, then collect as many points as there are in the dataset.
    parser.add_argument('--n_collects', type=int, default=None)
    parser.add_argument('--n_val_collects', type=int, default=0)
    parser.add_argument('--val_start_prop', type=float, default=0)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--real_edges_only', action='store_true')
    parser.add_argument('--unique_edges', action='store_true')
    parser.add_argument('--use_any_start', action='store_true')
    parser.add_argument('--use_graphs_starts', action='store_true')
    parser.add_argument('--all_starts_once', action='store_true')
    parser.add_argument('--max_path_length', type=int)
    parser.add_argument('--value_threshold', type=float)
    parser.add_argument('--top_percent_starts', type=float)
    parser.add_argument('--return_threshold', type=float)
    parser.add_argument('--unpenalized_rewards', action='store_true')
    parser.add_argument('--graph_name', default='vi.gt')
    parser.add_argument('--fusion', action='store_true')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    launch_jobs(args)

