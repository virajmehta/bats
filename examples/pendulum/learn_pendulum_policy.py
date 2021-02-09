"""
Learn pendulum policy from SARSA version of AWAC.
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
from modelling.policy_construction import supervise_critic,\
        learn_awac_policy


def run(args):
    # Assemble data.
    # Pre-train a QNet.
    if args.pretrain_qnets:
        data = assemble_dataset_from_graph(os.path.join(args.graph_dir,
                                             'converged_graph.gt'))
        qnets = [supervise_critic(
                    {k: data[k] for k in ['observations', 'actions', 'values']},
                    save_dir=os.path.join(args.save_dir, 'qnet%d' % n),
                    hidden_sizes='32,32,32',
                    epochs=args.qnet_epochs,
                    od_wait=args.qnet_od_wait,
                    val_size=args.qnet_val_size,
                    cuda_device=args.cuda_device)
                 for n in range(args.n_qnets)]
    else:
        qnets = None
    # Run AWAC with the pre-trained qnets.
    gname = 'base_graph.gt' if args.base_graph else 'stitched_graph.gt'
    data = assemble_dataset_from_graph(os.path.join(args.graph_dir, gname))
    awac_data = {k: data[k] for k in ['observations', 'actions', 'rewards',
                                      'next_observations', 'terminals']}
    if args.sarsa:
        awac_data['values'] = data['values']
    learn_awac_policy(
        dataset=awac_data,
        save_path=os.path.join(args.save_dir, 'awac'),
        policy_hidden_sizes='128,64',
        qnet_hidden_sizes='32,32,32',
        epochs=args.awac_epochs,
        cuda_device=args.cuda_device,
        n_qnets=args.n_qnets,
        qnets=qnets,
        env=NormalizedBoxEnv(gym.make('Pendulum-v0')),
        max_ep_len=1000,
        train_loops_per_epoch=50,
    )


def assemble_dataset_from_graph(graph_path, gamma=0.99):
    graph = load_graph(graph_path)
    data = {k: [] for k in ['observations', 'actions', 'rewards',
                            'next_observations', 'terminals', 'values']}
    states, actions, rews, nexts, terminals, values = [[] for _ in range(6)]
    for v in graph.iter_vertices():
        if not graph.vp.original[v]:
            continue
        for iidx, oidx, r, og in graph.get_out_edges(v,
                eprops=[graph.ep.reward, graph.ep.original]):
            if not og:
                continue
            data['observations'].append(graph.vp.obs[v])
            data['actions'].append(graph.ep.action[graph.edge(iidx, oidx)])
            data['rewards'].append(r)
            data['next_observations'].append(graph.vp.obs[oidx])
            data['terminals'].append(graph.vp.terminal[v])
            data['values'].append(r + gamma * graph.vp.value[oidx])
    data = {k: np.array(v) for k, v in data.items()}
    for k, v in data.items():
        if len(v.shape) == 1:
            data[k] = v.reshape(-1, 1)
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir')
    parser.add_argument('--save_dir')
    parser.add_argument('--n_qnets', type=int, default=2)
    parser.add_argument('--qnet_epochs', type=int, default=2500)
    parser.add_argument('--awac_epochs', type=int, default=50)
    parser.add_argument('--qnet_od_wait', type=int, default=100)
    parser.add_argument('--qnet_val_size', type=float, default=0.1)
    parser.add_argument('--sarsa', action='store_true')
    parser.add_argument('--pretrain_qnets', action='store_true')
    parser.add_argument('--base_graph', action='store_true')
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    run(args)
