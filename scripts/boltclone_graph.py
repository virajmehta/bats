"""
Do boltzmann cloning on a converged graph.
"""
import argparse
from bats import BATSTrainer
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle as pkl

import d4rl
from graph_tool import load_graph
from graph_tool.draw import graph_draw
import h5py
import gym
import numpy as np
from tqdm import tqdm

from env_wrapper import NormalizedBoxEnv
from modelling.policy_construction import behavior_clone
from modelling.utils.graph_util import make_boltzmann_policy_dataset,\
        get_value_thresholded_starts, make_graph_consistent
from util import get_starts_from_graph, get_offline_env, make_output_dir


def run(args):
    # Learn an advantage weighting function.
    env = gym.make(args.env)
    graph = load_graph(os.path.join(args.graph_dir, args.graph_name))
    if args.planning_quantile is not None and args.epsilon_planning is not None:
        print('Making graph consistent with hyperparameters...')
        graph, _ = make_graph_consistent(graph, args.planning_quantile,
                                         args.epsilon_planning,
                                         stitch_itr=args.stitch_itr)
        with open(os.path.join(args.graph_dir, 'args.json')) as f:
            train_args = json.load(f)
        train_args['save_dir'] = args.save_dir
        output_dir = make_output_dir(train_args['name'],
                                     True,
                                     deepcopy(train_args),
                                     Path(args.save_dir))
        env, dataset = get_offline_env(train_args['env_name'],
                                       train_args['dataset_fraction'],
                                       data_path=train_args['offline_dataset_path'])
        trainer = BATSTrainer(dataset, env, output_dir, **train_args)
        trainer.G = graph
        trainer.value_iteration()
        trainer.G.save(os.path.join(args.save_dir, 'trimmed.gt'))
    if args.n_collects is None:
        if args.all_starts_once:
            args.n_collects = float('inf')
        else:
            args.n_collects = graph.num_vertices()
    if args.use_graphs_starts:
        starts = None
    else:
        starts = get_starts_from_graph(graph, env, args.env)
    data, val_data, _ = make_boltzmann_policy_dataset(
            graph=graph,
            n_collects=args.n_collects,
            temperature=args.temperature,
            max_ep_len=env._max_episode_steps,
            n_val_collects=args.n_val_collects,
            val_start_prop=args.val_start_prop,
            val_selection_prob=args.val_selection_prob,
            any_state_is_start=args.use_any_start,
            only_add_real=args.real_edges_only,
            val_only_add_real=args.val_real_edges_only,
            get_unique_edges=args.unique_edges,
            starts=starts,
            threshold_start_val=args.value_threshold,
            top_percent_starts=args.top_percent_starts,
            return_threshold=args.return_threshold,
            all_starts_once=args.all_starts_once,
    )
    # Run AWR with the pre-trained qnets.
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
        num_eval_eps=args.num_eval_eps,
        add_entropy_bonus=args.add_entropy_bonus,
        batch_updates_per_epoch=args.batch_updates_per_epoch,
        target_entropy=args.target_entropy,
    )
    with open(os.path.join(args.save_dir, 'args.pkl'), 'wb') as f:
        pkl.dump(args, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env')
    parser.add_argument('--graph_dir')
    parser.add_argument('--save_dir')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_updates_per_epoch', type=int)
    parser.add_argument('--od_wait', type=int)
    # If None, then collect as many points as there are in the dataset.
    parser.add_argument('--n_collects', type=int, default=None)
    parser.add_argument('--all_starts_once', action='store_true')
    parser.add_argument('--n_val_collects', type=int, default=0)
    parser.add_argument('--val_start_prop', type=float, default=0)
    parser.add_argument('--val_selection_prob', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--pi_architecture', default='256,256')
    parser.add_argument('--real_edges_only', action='store_true')
    parser.add_argument('--val_real_edges_only', action='store_true')
    parser.add_argument('--unique_edges', action='store_true')
    parser.add_argument('--value_threshold', type=float)
    parser.add_argument('--top_percent_starts', type=float)
    parser.add_argument('--return_threshold', type=float)
    parser.add_argument('--use_any_start', action='store_true')
    parser.add_argument('--num_eval_eps', type=int, default=10)
    parser.add_argument('--add_entropy_bonus', action='store_true')
    parser.add_argument('--use_graphs_starts', action='store_true')
    parser.add_argument('--target_entropy', type=float, default=None)
    parser.add_argument('--planning_quantile', type=float)
    parser.add_argument('--epsilon_planning', type=float)
    parser.add_argument('--stitch_itr', type=int)
    parser.add_argument('--graph_name', default='vi.gt')
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    run(args)
