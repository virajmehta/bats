"""
Script for doing behavior cloning on a d4rl dataset.
"""
import argparse

import d4rl
import gym
import numpy as np
import torch
from tqdm import tqdm

from env_wrapper import NormalizedBoxEnv
from modelling.policy_construction import load_policy


def train(args):
    env = NormalizedBoxEnv(gym.make(args.env))
    policy = load_policy(
            args.policy_dir,
            obs_dim=env.observation_space.low.size,
            act_dim=env.action_space.low.size,
            deterministic=args.is_deterministic,
            hidden_sizes=args.hidden_sizes,
            cuda_device=args.cuda_device,
    )
    pbar = tqdm(total=args.n_episodes)
    evals = []
    for _ in range(args.n_episodes):
        evals.append(unroll(env, policy, max_ep_len=args.max_ep_len))
        pbar.set_postfix_str('Return: %f' % evals[-1])
        pbar.update(1)
    print('Returns: %f +- %f' % (np.mean(evals), np.std(evals)))


def unroll(env, policy, max_ep_len=float('inf')):
    t = 0
    done = False
    s = env.reset()
    ret = 0
    while not done and t < max_ep_len:
        with torch.no_grad():
            a = policy.get_action(torch.Tensor(s),
                                  deterministic=True).cpu().numpy()
        n, r, done, _ = env.step(a)
        tup = (s, a, r, n, done)
        ret += r
        s = n
    return ret


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_dir')
    parser.add_argument('--env')
    parser.add_argument('--n_episodes', type=int, default=50)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--hidden_sizes', type=str, default='256,256')
    parser.add_argument('--is_deterministic', action='store_true')
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    train(args)
