"""
Do TD value learning on a graph.
"""
import argparse
import os

from graph_tool import load_graph

from modelling.critic_construction import tdlambda_critic
from modelling.utils.graph_util import get_nstep_learning_set


def run(args):
    # Learn an advantage weighting function.
    graph = load_graph(os.path.join(args.graph_dir, 'vi.gt'))
    data = get_nstep_learning_set(graph, args.nsteps, max_data=args.max_data)
    tdlambda_critic(
        dataset=data,
        save_dir=args.save_dir,
        epochs=args.epochs,
        hidden_sizes=args.vf_architecture,
        batch_updates_per_epoch=args.batch_updates_per_epoch,
        cuda_device=args.cuda_device,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--nsteps', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_updates_per_epoch', type=int, default=50)
    parser.add_argument('--max_data', type=int, default=1000)
    parser.add_argument('--vf_architecture', default='256,256')
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    run(args)
