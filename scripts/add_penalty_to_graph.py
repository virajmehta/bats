"""
Add penalty to the graph and redo value iteration.
"""
import argparse
from bats import BATSTrainer
import json
from multiprocessing import Pool
from pathlib import Path
import os
import pickle as pkl

from graph_tool import load_graph

from modelling.utils.graph_util import add_penalty_to_graph
from util import s2f, make_output_dir, get_offline_env


def run(config):
    graph = load_graph(os.path.join(config['graph_dir'], config['graph_name']))
    graph, _ = make_graph_consistent(
            graph,
            args.planning_quantile,
            args.epsilon_planning,
            stitch_itr=args.stitch_itr,
    )
    graph, _ = add_penalty_to_graph(
            graph=graph,
            disagreement_coef=config['disagreement_coef'],
            planning_coef=config['planning_coef'],
            silent=True,
    )
    with open(os.path.join(config['graph_dir'], 'args.json')) as f:
        train_args = json.load(f)
    train_args['save_dir'] = config['save_dir']
    output_dir = make_output_dir(train_args['name'],
                                 True,
                                 deepcopy(train_args),
                                 Path(config['save_dir']))
    env, dataset = get_offline_env(train_args['env_name'],
                                   train_args['dataset_fraction'],
                                   data_path=train_args['offline_dataset_path'])
    trainer = BATSTrainer(dataset, env, output_dir, **train_args)
    trainer.G = graph
    trainer.value_iteration()
    trainer.G.save(os.path.join(config['save_dir'], 'vi.gt'))
    with open(os.path.join(config['save_dir'], 'penalty_args.pkl'), 'wb') as f:
        pkl.dump(args, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', requ)
    parser.add_argument('--graph_dir', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--disagreement_coef', type=float, required=True)
    parser.add_argument('--planerr_coef', type=float, required=True)
    parser.add_argument('--planning_quantile', type=float, required=True)
    parser.add_argument('--epsilon_planning', type=float, required=True)
    parser.add_argument('--stitch_itr', type=int)
    parser.add_argument('--graph_name', default='mdp.gt')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    run(vars(args))
