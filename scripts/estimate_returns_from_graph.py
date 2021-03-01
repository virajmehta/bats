"""
Estimate what the policy's value will be based on the graph.
"""
import argparse

import d4rl
import gym
import h5py
import numpy as np

from graph_tool import load_graph
from modelling.utils.graph_util import get_best_policy_returns
from examples.mazes.maze_util import get_starts_from_graph


def run(args):
    graph = load_graph(args.graph_path)
    if args.env is not None and 'maze' in args.env:
        env = gym.make(args.env)
        starts = get_starts_from_graph(graph, env)
    else:
        starts = None
    returns = get_best_policy_returns(graph, starts=starts,
                                      horizon=args.horizon)
    rets = np.array([r[0] for r in returns]).reshape(-1, 1)
    obs = np.vstack([r[1] for r in returns])
    ts = np.array([r[2] for r in returns]).reshape(-1, 1)
    print('Returns: %f +- %f' % (np.mean(rets), np.std(rets)))
    print('Trajectory Length: %f +- %f' % (np.mean(ts), np.std(ts)))
    if args.save_path is not None:
        with h5py.File(args.save_path, 'w') as wd:
            wd.create_dataset('observations', data=obs)
            wd.create_dataset('returns', data=rets)
            wd.create_dataset('trajectory_length', data=ts)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_path', required=True)
    parser.add_argument('--env')
    parser.add_argument('--save_path')
    parser.add_argument('--horizon', type=int, default=1000)
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    run(args)
