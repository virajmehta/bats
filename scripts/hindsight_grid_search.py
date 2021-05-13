"""
Grid search over possible graph configurations.
"""
import argparse
from copy import deepcopy

from scripts.boltclone_graph import run
from util import s2f


def launch_jobs(args):
    for planning_quantile in s2f(args.planning_quantiles):
        for epsilon_planning in s2f(args.epsilon_plannings):
            run_arg = deepcopy(args)
            run_arg.planning_quantile = planning_quantile
            run_arg.epsilon_planning = epsilon_planning
            run_arg.save_dir = '_'.join([
                    args.save_prefix,
                    'pq%f' % planning_quantile,
                    'ep%f' % epsilon_planning,
            ])
            run(run_arg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env')
    parser.add_argument('--graph_dir')
    parser.add_argument('--save_prefix')
    parser.add_argument('--epsilon_plannings', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_updates_per_epoch', type=int)
    parser.add_argument('--od_wait', type=int, default=10)
    # If None, then collect as many points as there are in the dataset.
    parser.add_argument('--n_collects', type=int, default=None)
    parser.add_argument('--n_val_collects', type=int, default=10000)
    parser.add_argument('--val_start_prop', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--pi_architecture', default='256,256')
    parser.add_argument('--real_edges_only', action='store_true')
    parser.add_argument('--unique_edges', action='store_true')
    parser.add_argument('--value_threshold', type=float)
    parser.add_argument('--top_percent_starts', type=float)
    parser.add_argument('--use_any_start', action='store_true')
    parser.add_argument('--num_eval_eps', type=int, default=10)
    parser.add_argument('--add_entropy_bonus', action='store_true')
    parser.add_argument('--use_graphs_starts', action='store_true')
    parser.add_argument('--target_entropy', type=float, default=None)
    parser.add_argument('--planning_quantiles', type=str, default='0.8')
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--graph_name', default='vi.gt')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    launch_jobs(args)
