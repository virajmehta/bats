"""
Behavior clone on Boltzmann policy.
"""
import argparse
import os

import h5py
from graph_tool import load_graph
from graph_tool.draw import graph_draw
import gym
import numpy as np
from tqdm import tqdm

from env_wrapper import NormalizedBoxEnv
from modelling.policy_construction import behavior_clone
from modelling.
from modelling.utils.graph_util import make_boltzmann_policy_dataset


def run(args):
    # Fetch the boltzmann data.
    graph = load_graph(os.path.join(args.graph_dir, 'converged_graph.gt'))
    data, val_data = make_boltzmann_policy_dataset(
            graph=graph,
            n_collects=args.n_collects,
            temperature=args.temperature,
            max_ep_len=args.max_ep_len,
            n_val_collects=args.n_val_collects,
            val_start_prop=args.val_start_prop,
            any_state_is_start=not args.use_data_starts,
    )
    env = NormalizedBoxEnv(gym.make('Pendulum-v0')),
    # Behavior clone on the boltzmann data.
    policy = behavior_clone(
        dataset=data,
        save_dir=os.path.join(args.save_dir, 'bc'),
        hidden_sizes='128,64',
        epochs=args.epochs,
        od_wait=args.od_wait,
        val_dataset=val_data,
        cuda_device=args.cuda_device,
        env=env,
        max_ep_len=1000,
        train_loops_per_epoch=1,
    )
    # Train a critic based on this policy.
    graph = load_graph(os.path.join(args.graph_dir, 'stitched_graph.gt'))
    all_data = make_qlearning_dataset(graph)
    qnets = sarsa_learn_critic(
        dataset=all_data,
        policy=policy,
        save_dir=os.path.join(args.save_dir, 'critic'),
        epochs=args.critic_epochs,
        hidden_sizes='32,32,32',
        cuda_device=cuda_device,
    )
    # Fine tune the policy to maximize the critic.
    train_policy_to_maximize_critic(
        dataset=data,
        env=env,
        qnets=qnets,
        save_dir=os.path.join(args.save_dir, 'policy'),
        epochs=args.policy_epochs,
        policy=policy,
        cuda_device=cuda_device,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir')
    parser.add_argument('--save_dir')
    parser.add_argument('--bc_epochs', type=int, default=10)
    parser.add_argument('--critic_epochs', type=int, default=100)
    parser.add_argument('--policy_epochs', type=int, default=25)
    parser.add_argument('--od_wait', type=int, default=100)
    parser.add_argument('--n_collects', type=int, default=int(1e5))
    parser.add_argument('--n_val_collects', type=int, default=0)
    parser.add_argument('--val_start_prop', type=float, default=0)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--max_ep_len', type=int, default=50)
    parser.add_argument('--use_data_starts', action='store_true')
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    run(args)
