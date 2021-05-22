"""
Add penalty to the graph and redo value iteration.
"""
import argparse
from bats import BATSTrainer
import json
from multiprocessing import Pool
from pathlib import Path
import os

from graph_tool import load_graph

from modelling.utils.graph_util import add_penalty_to_graph
from util import s2f, make_output_dir, get_offline_env


def launch_jobs(args):
    configs = []
    for dc in s2f(args.disagreement_coefs):
        for pc in s2f(args.planerr_coefs):
            configs.append({
                'graph_dir': args.graph_dir,
                'graph_name': args.graph_name,
                'graph_path': os.path.join(args.graph_dir, args.graph_name),
                'disagreement_coef': dc,
                'planning_coef': pc,
                'save_dir': os.path.join(args.parent_save_dir,
                    'dc%.3f_pc%.3f' % (dc, pc))
            })
    with Pool(args.num_jobs) as p:
        p.map(penalize_graph, configs)


def penalize_graph(config):
    graph = load_graph(os.path.join(config['graph_dir'], config['graph_name']))
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', requ)
    parser.add_argument('--graph_dir', required=True)
    parser.add_argument('--parent_save_dir', required=True)
    parser.add_argument('--disagreement_coefs', required=True)
    parser.add_argument('--planerr_coefs', required=True)
    parser.add_argument('--num_jobs', type=int, required=True)
    parser.add_argument('--graph_name', default='mdp.gt')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    run(args)
