"""
Add penalty to the graph and redo value iteration.
"""
import argparse
from bats import BATSTrainer
import json
from multiprocessing import Pool
from pathlib import Path
import os

from graph_tool import load_graph

from modelling.utils.graph_util import add_penalty_to_graph
from scripts.add_penalty_to_graph import run
from util import s2f, make_output_dir, get_offline_env


def launch_jobs(args):
    configs = []
    for dc in s2f(args.disagreement_coefs):
        for pc in s2f(args.planerr_coefs):
            for pq in s2f(args.planning_quantiles):
                for ep in s2f(args.epsilon_plannings):
                    configs.append({
                        'graph_dir': args.graph_dir,
                        'graph_name': args.graph_name,
                        'graph_path': os.path.join(args.graph_dir,
                                                   args.graph_name),
                        'disagreement_coef': dc,
                        'planning_coef': pc,
                        'epsilon_planning': ep,
                        'planning_quantile': pq,
                        'save_dir': os.path.join(args.parent_save_dir,
                            'dc%.3f_pc%.3f_pq%.3f_ep%.3f' % (dc, pc, pq, ep))
                        'silent': True,
                    })
    with Pool(args.num_jobs) as p:
        p.map(run, configs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True)
    parser.add_argument('--graph_dir', required=True)
    parser.add_argument('--parent_save_dir', required=True)
    parser.add_argument('--planerr_coefs', required=True)
    parser.add_argument('--epsilon_plannings', required=True)
    parser.add_argument('--num_jobs', type=int, required=True)
    parser.add_argument('--disagreement_coefs', default='0')
    parser.add_argument('--planning_qunatile', default='0.8')
    parser.add_argument('--graph_name', default='mdp.gt')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    launch_jobs(args)
