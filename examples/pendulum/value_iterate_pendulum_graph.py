"""
Do value iteration on pendulum graph.
"""
import argparse
import os

from graph_tool import load_graph
from graph_tool.draw import graph_draw
import numpy as np
from tqdm import tqdm


def run(args):
    # Load in the graph.
    graph = load_graph(os.path.join(args.graph_dir, 'stitched_graph.gt'))
    # Trim any loose threads.
    trim_loose_threads(graph)
    # Do value iteration.
    pbar = tqdm(total=args.num_iterations)
    changes = []
    for _ in range(args.num_iterations):
        changes.append(value_iteration(graph, gamma=args.gamma))
        pbar.set_postfix_str('Average Change: %f' % changes[-1])
        pbar.update(1)
    graph_draw(graph, vertex_fill_color=graph.vp.value,
               output=os.path.join(args.graph_dir, 'converged_graph.png'))
    graph.save(os.path.join(args.graph_dir, 'converged_graph.gt'))
    np.save(os.path.join(args.graph_dir, 'value_itr_stats.npy'),
            np.array(changes))


def trim_loose_threads(graph):
    trimmed = False
    num_trimmed = 0
    while not trimmed:
        trimmed = True
        verts = graph.get_vertices()
        out_degrees = graph.get_out_degrees(verts)
        # Docs say that vertex indices should be sorted in decreasing order
        trim_idxs = np.argwhere(out_degrees == 0)[::-1]
        if len(trim_idxs) > 0:
            trimmed = False
            graph.remove_vertex(verts[trim_idxs])
            num_trimmed += len(trim_idxs)
    print('%d nodes trimmed.' % num_trimmed)


def value_iteration(graph, gamma=0.99):
    total_change = 0
    for v in graph.iter_vertices():
        neighbors = graph.get_out_neighbors(v, vprops=[graph.vp.value])
        vidxs = neighbors[:, 0]
        values = neighbors[:, 1]
        edges = graph.get_out_edges(v, eprops=[graph.ep.reward])
        rewards = edges[:, 2]
        backups = rewards + gamma * values
        best_arm = np.argmax(backups)
        updated_value = backups[best_arm]
        total_change += np.abs(graph.vp.value[v] - updated_value)
        graph.vp.value[v] = updated_value
        graph.vp.best_action_idx[v] = best_arm
    return total_change / graph.num_vertices()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir')
    parser.add_argument('--num_iterations', type=int)
    parser.add_argument('--saved_graph', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    run(args)
