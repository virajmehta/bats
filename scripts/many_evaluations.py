"""
Evaluate many policies at once.
"""
import argparse
from collections import OrderedDict
import os
import pickle as pkl

import gym
import numpy as np
from tqdm import tqdm
from scipy.stats import sem

from modelling.policy_construction import load_policy
from modelling.utils.torch_utils import unroll


def run(args):
    total = 0
    for rundir in args.target_dir.split(','):
        total += len(os.listdir(rundir))
    pbar = tqdm(total=total)
    env = None
    with open(args.save_path, 'w') as f:
        f.writeline(','.join([
            'AverageReturns' ,
            'AverageNormalizeReturns',
            'Path',
        ]))
    all_returns = []
    all_nreturns = []
    for rundir in args.target_dir.split(','):
        for trialpath in os.listdir(rundir):
            with open(os.path.join(rundir, trialpath, 'args.pkl'), 'rb') as f:
                config = pkl.load(f)
            if env is None:
                env = gym.make(config.env)
            policy = load_policy(
                os.path.join(rundir, trialpath),
                obs_dim=env.observation_space.low.size,
                act_dim=env.action_space.low.size,
                policy_file=args.filename,
                hidden_sizes=args.pi_architecture,
            )
            returns = [unroll(env, policy) for _ in range(args.episodes)]
            nreturns = [d4rl.get_normalized_score(config.env, r) for r in returns]
            pbar.set_postfix(OrderedDict(
                Return=np.mean(returns),
                NormReturn=np.mean(nreturns) * 100,
            ))
            with open(args.save_path, 'a') as f:
                f.writeline(','.join([
                    str(np.mean(returns)),
                    str(np.mean(nreturns) * 100),
                    trialpath,
                ]))
            all_returns.append(np.mean(returns))
            all_nreturns.append(np.mean(all_nreturns) * 100)
    pbar.close()
    print('================================================')
    print('Returns: %f +- %f' % (np.mean(all_returns), sem(all_returns)))
    print('Score: %.1f +- %.1f' % (np.mean(all_nreturns), sem(all_nreturns)))
    print('================================================')



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--filename', default='final.pt')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    run(args)

