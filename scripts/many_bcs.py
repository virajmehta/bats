"""
Do boltzmann cloning on a converged graph.
"""
import argparse
from copy import deepcopy
from multiprocessing import Pool
import os

from scripts.boltclone_graph import run
from util import s2f


def launch_jobs(args):
    configs = create_config_list(args)
    with Pool(args.num_jobs) as p:
        p.map(run, configs)


def create_config_list(args):
    configs = []
    for graph in os.listdir(args.target_dir):
        for t in range(1, args.trials + 1):
            config = deepcopy(args)
            config['save_dir'] = os.path.join(
                    args.save_location,
                    graph + '_trial%d' % t,
            )
            config['graph_dir'] = os.path.join(args.train_dir, graph)
            config['silent'] = True
            configs.append(config)
    return configs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env')
    parser.add_argument('--target_dir')
    parser.add_argument('--save_location')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--num_jobs', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_updates_per_epoch', type=int)
    parser.add_argument('--od_wait', type=int, default=10)
    # If None, then collect as many points as there are in the dataset.
    parser.add_argument('--n_collects', type=int, default=None)
    parser.add_argument('--n_val_collects', type=int, default=0)
    parser.add_argument('--val_start_prop', type=float, default=0)
    parser.add_argument('--val_selection_prob', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--pi_architecture', default='256,256')
    parser.add_argument('--real_edges_only', action='store_true')
    parser.add_argument('--val_real_edges_only', action='store_true')
    parser.add_argument('--unique_edges', action='store_true')
    parser.add_argument('--value_threshold', type=float)
    parser.add_argument('--top_percent_starts', type=float)
    parser.add_argument('--return_threshold', type=float)
    parser.add_argument('--use_any_start', action='store_true')
    parser.add_argument('--num_eval_eps', type=int, default=10)
    parser.add_argument('--add_entropy_bonus', action='store_true')
    parser.add_argument('--use_graphs_starts', action='store_true')
    parser.add_argument('--target_entropy', type=float, default=None)
    parser.add_argument('--all_starts_once', action='store_true')
    parser.add_argument('--stitch_itr', type=int)
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--graph_name', default='trimmed.gt')
    parser.add_argument('--fusion', action='store_true')
    parser.add_argument('--max_path_length', type=int)
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    launch_jobs(args)

