"""
Do boltzmann cloning on a converged graph.
"""
import argparse
import os

import d4rl
from graph_tool import load_graph
from graph_tool.draw import graph_draw
import h5py
import gym
import numpy as np
from tqdm import tqdm
import torch

from env_wrapper import NormalizedBoxEnv
from modelling.dynamics_construction import get_pnn
from modelling.policy_construction import behavior_clone
from modelling.utils.graph_util import make_boltzmann_policy_dataset,\
        get_value_thresholded_starts
from examples.mazes.maze_util import get_starts_from_graph


def run(args):
    # Learn an advantage weighting function.
    env = gym.make(args.env)
    graph = load_graph(os.path.join(args.graph_dir, 'vi.gt'))
    if args.n_collects is None:
        args.n_collects = graph.num_vertices()
    if 'maze' in args.env:
        starts = get_starts_from_graph(graph, env)
    else:
        starts = None
    if args.value_threshold > 0:
        starts = get_value_thresholded_starts(graph, args.value_threshold,
                                              starts)
    data, val_data, _ = make_boltzmann_policy_dataset(
            graph=graph,
            n_collects=args.n_collects,
            temperature=args.temperature,
            max_ep_len=env._max_episode_steps,
            n_val_collects=args.n_val_collects,
            val_start_prop=args.val_start_prop,
            any_state_is_start=args.use_any_start,
            only_add_real=args.real_edges_only,
            get_unique_edges=True,
            starts=starts,
    )
    data = add_imaginary_points(data, args)
    behavior_clone(
        dataset=data,
        save_dir=args.save_dir,
        hidden_sizes=args.pi_architecture,
        epochs=args.epochs,
        od_wait=args.od_wait,
        val_dataset=val_data,
        cuda_device=args.cuda_device,
        env=env,
        max_ep_len=env._max_episode_steps,
        train_loops_per_epoch=1,
    )


def add_imaginary_points(data, args):
    act_dim = data['actions'].shape[1]
    pnn = get_pnn(data['observations'].shape[1], act_dim)
    pnn.load_model(args.model_path, map_location='cpu')
    pbar = tqdm(total=len(data['observations']) * args.horizon
                      * args.num_per_state)
    new_obs, new_acts = [], []
    for ob in data['observations']:
        for _ in range(args.num_per_state):
            curr = np.array(ob)
            act = np.random.uniform(size=act_dim)
            for _ in range(args.horizon):
                curr = curr \
                    + pnn.get_mean_logvar(torch.Tensor(np.append(curr, act)))[0].numpy()[1:]
                if walker_terminal(curr):
                    break
                new_obs.append(curr)
                new_acts.append(act)
            pbar.update(args.horizon)
    pbar.close()
    data['observations'] = np.vstack([data['observations'], np.array(new_obs)])
    data['actions'] = np.vstack([data['actions'], np.array(new_acts)])
    return data


def walker_terminal(states):
    """As written in MOPO code base."""
    states = _add_axis_if_needed(states)
    height = states[:, 0]
    angle = states[:, 1]
    return np.logical_or.reduce([
        height <= 0.8,
        height >= 2.0,
        angle <= -1.0,
        angle >= 1.0,
    ])


def _add_axis_if_needed(states):
    if len(states.shape) == 1:
        states = states[np.newaxis, ...]
    return states


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True)
    parser.add_argument('--graph_dir', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--num_per_state', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--od_wait', type=int, default=0)
    # If None, then collect as many points as there are in the dataset.
    parser.add_argument('--n_collects', type=int, default=None)
    parser.add_argument('--n_val_collects', type=int, default=0)
    parser.add_argument('--val_start_prop', type=float, default=0)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--pi_architecture', default='256,256')
    parser.add_argument('--real_edges_only', action='store_true')
    parser.add_argument('--value_threshold', type=float, default=0)
    parser.add_argument('--use_any_start', action='store_true')
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    run(args)
