"""
Estimate what the policy's value will be based on the graph.
"""
import argparse
from collections import OrderedDict

import d4rl
import gym
import h5py
import numpy as np
from tqdm import tqdm

from graph_tool import load_graph
from modelling.utils.graph_util import get_best_policy_returns
from examples.mazes.maze_util import get_starts_from_graph
from util import make_mujoco_resetter


def run(args):
    graph = load_graph(args.graph_path)
    env = gym.make(args.env)
    if args.env is not None and 'maze' in args.env:
        starts = get_starts_from_graph(graph, env)
        horizon = env._max_episode_steps
        # Only need this since most mazes don't have terminal prop yet.
        ignore_terminals = True
    else:
        starts = None
        horizon = args.horizon
        ignore_terminals = False
    print('Unrolling in graph...')
    returns = get_best_policy_returns(graph, starts=starts,
                                      horizon=horizon,
                                      ignore_terminals=ignore_terminals)
    rets = np.array([r[0] for r in returns]).reshape(-1, 1)
    ts = np.array([r[1].shape[0] for r in returns])
    print('Returns: %f +- %f' % (np.mean(rets), np.std(rets)))
    print('Trajectory Length: %f +- %f' % (np.mean(ts), np.std(ts)))
    print('Replaying in environment...')
    resetter = make_mujoco_resetter(env, args.env)
    replay_returns = []
    pbar = tqdm(total=len(returns))
    for ret in returns:
        resetter(ret[1][0])
        repret = 0
        for act in ret[2]:
            _, rew, done, _ = env.step(act)
            repret += rew
            if done:
                break
        replay_returns.append(repret)
        pbar.set_postfix(OrderedDict(ReplayReturn=repret))
        pbar.update(1)
    replay_returns = np.array(replay_returns)
    print('Replay Returns: %f +- %f' % (np.mean(replay_returns),
                                        np.std(replay_returns)))
    print('Average Error: %f' % (np.mean(rets - replay_returns)))
    if args.save_path is not None:
        with h5py.File(args.save_path, 'w') as wd:
            wd.create_dataset('returns', data=rets)
            wd.create_dataset('replay_returns', data=replay_returns)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_path', required=True)
    parser.add_argument('--env', required=True)
    parser.add_argument('--save_path')
    parser.add_argument('--horizon', type=int, default=1000)
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    run(args)
