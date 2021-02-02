"""
Evaluate how well bc policy does.
"""
import argparse
import os
import pickle as pkl
from tqdm import tqdm

import gym
import numpy as np
import torch
from torch.nn.functional import tanh

from env_wrapper import NormalizedBoxEnv
from modelling.models import RegressionNN
from util import s2i


DEFAULT_POLICY_PARAMS = dict(
    hidden_sizes='128,64',
    output_activation=tanh,
)


def load_policy(policy_path, x_dim, y_dim):
    policy = RegressionNN(x_dim, y_dim,
            hidden_sizes=s2i(DEFAULT_POLICY_PARAMS['hidden_sizes']),
            output_activation=DEFAULT_POLICY_PARAMS['output_activation'],
    )
    policy.load_model(policy_path, map_location='cpu')
    def get_act(state):
        net_in = torch.Tensor(state).reshape(1, -1)
        with torch.no_grad():
            return policy(net_in).numpy().flatten()
    return get_act

def run(args):
    env = NormalizedBoxEnv(gym.make(args.env))
    obs_dim = env.observation_space.low.size
    act_dim = env.action_space.low.size
    if args.policy_path is None:
        policy = lambda s: env.action_space.sample()
    elif args.is_rlkit_policy:
        data = torch.load(args.policy_path)
        pnet = data['evaluation/policy']
        policy = lambda s: pnet(torch.Tensor(s).reshape(1, -1)).detach().numpy().flatten()
    else:
        policy = load_policy(args.policy_path, obs_dim, act_dim)
    evaluations = []
    pbar = tqdm(total=args.n_rollouts)
    for _ in range(args.n_rollouts):
        returns = 0
        state = env.reset()
        done = False
        t = 0
        while not done and t < args.max_path_length:
            n, r, done, _ = env.step(policy(state))
            returns += r
            state = n
        evaluations.append(returns)
        pbar.set_postfix_str('Return: %f' % returns)
        pbar.update(1)
    print('Returns: %f +- %f' % (np.mean(evaluations), np.std(evaluations)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env')
    parser.add_argument('--policy_path', default=None,
                        help='If none provided, use a random policy.')
    parser.add_argument('--n_rollouts', type=int, default=10)
    parser.add_argument('--max_path_length', type=int, default=200)
    parser.add_argument('--is_rlkit_policy', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    run(parse_args())
