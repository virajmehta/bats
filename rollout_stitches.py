import argparse
from graph_tool import load_graph, ungroup_vector_property
from scipy.sparse import load_npz
from pathlib import Path
import numpy as np
import pickle


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_path', type=str)
    parser.add_argument('neighbors_path', type=Path)
    parser.add_argument('stitches_tried_path', type=Path)
    parser.add_argument('start_states_path', type=Path)
    parser.add_argument('output_path', type=Path)
    parser.add_argument('action_dim', type=int)
    parser.add_argument('obs_dim', type=int)
    parser.add_argument('rollout_chunk_size', type=int)
    parser.add_argument('temperature', type=float)
    parser.add_argument('gamma', type=float)
    return parser.parse_args()


def get_possible_stitches(
        G,
        all_neighbors,
        stitches_tried,
        state_props,
        action_props,
        currv,
        children,
        child_actions,
        Q,
        max_stitches):
    neighbors = np.nonzero(all_neighbors[currv, :])[1]
    possible_stitches = []
    advantages = []
    # take all nearest neighbors to the vertex in question and add their outgoing edges to the set to stitch
    n_stitches = 0
    for neigh in neighbors:
        out_neighbors = G.get_out_neighbors(neigh, vprops=[*state_props, G.vp.value])
        out_edges = G.get_out_edges(neigh, eprops=action_props)
        deletes = []
        for i, out_neighbor in enumerate(out_neighbors[:, 0]):
            if (currv, out_neighbor) in stitches_tried:
                deletes.append(i)
        out_neighbors = np.delete(out_neighbors, deletes, axis=0)
        out_edges = np.delete(out_edges, deletes, axis=0)
        out_start = np.ones_like(out_neighbors[:, :1]) * currv
        possible_stitches.append(np.concatenate((out_start,
                                                 out_neighbors[:, :1],
                                                 np.tile(G.vp.obs[currv], (out_neighbors.shape[0], 1)),
                                                 out_neighbors[:, 1:-1],
                                                 out_edges[:, 2:]), axis=-1))
        advantages.append(out_neighbors[:, -1] - Q)
        n_stitches += out_start.shape[0]
        if n_stitches >= max_stitches:
            return possible_stitches, advantages, n_stitches
    # take all nearest neighbors to each child vertex and add them as edges to plan toward
    for i, child in enumerate(children):
        child_neighbors = np.nonzero(all_neighbors[child, :])[1]
        deletes = []
        for j, child in enumerate(child_neighbors):
            if (currv, child) in stitches_tried:
                deletes.append(j)
        child_neighbors = np.delete(child_neighbors, deletes, axis=0)
        child_neighbor_obs = G.get_vertices(state_props)[child_neighbors, 1:]
        values = G.vp.value.get_array()[child_neighbors]
        action = child_actions[i, :]
        actions = np.tile(action, (len(child_neighbors), 1))
        out_start = np.ones((len(child_neighbors), 1)) * currv
        possible_stitches.append(np.concatenate((out_start,
                                                 child_neighbors[:, np.newaxis],
                                                 np.tile(G.vp.obs[currv], (len(child_neighbors), 1)),
                                                 child_neighbor_obs,
                                                 actions), axis=-1))
        advantages.append(values - Q)
        n_stitches += values.shape[0]
        if n_stitches >= max_stitches:
            return possible_stitches, advantages, n_stitches
    return possible_stitches, advantages, n_stitches


def main(args):
    G = load_graph(args.graph_path)
    neighbors = load_npz(args.neighbors_path)
    with args.stitches_tried_path.open('rb') as f:
        stitches_tried = pickle.load(f)
    start_states = np.load(args.start_states_path)
    stitches = []
    advantages = []
    action_props = ungroup_vector_property(G.ep.action, range(args.action_dim))
    state_props = ungroup_vector_property(G.vp.obs, range(args.obs_dim))
    # max_ep_len = 1000
    max_ep_len = 1000
    max_eps = 2 * args.rollout_chunk_size / max_ep_len
    # max_eps = 1000
    neps = 0
    total_stitches = 0
    while total_stitches < args.rollout_chunk_size and neps < max_eps:
        t = 0
        currv = np.random.choice(start_states)
        while t < max_ep_len:
            # do a Boltzmann rollout
            if args.temperature > 0:
                if G.vp.terminal[currv]:
                    break
                childs = G.get_out_neighbors(currv, vprops=[G.vp.value])
                edges = G.get_out_edges(currv, eprops=[G.ep.reward, *action_props])
                if len(childs) == 0:
                    new_stitches, new_advantages, n_stitches = get_possible_stitches(G,
                                                                                     neighbors,
                                                                                     stitches_tried,
                                                                                     state_props,
                                                                                     action_props,
                                                                                     currv,
                                                                                     childs[:, 0].astype(int),
                                                                                     edges[:, -args.action_dim:],
                                                                                     0,
                                                                                     args.rollout_chunk_size - total_stitches)
                    stitches += new_stitches
                    advantages += new_advantages
                    total_stitches += n_stitches
                    break
                qs = edges[:, -1] + args.gamma * childs[:, 1]
                minq, maxq = np.min(qs), np.max(qs)
                if minq == maxq:
                    norm_qs = np.ones(qs.shape)
                else:
                    norm_qs = (qs - minq) / (maxq - minq)
                probs = np.exp(norm_qs / args.temperature)
                probs /= np.sum(probs)
                arm = np.random.choice(childs.shape[0], p=probs)
                nxtv = childs[arm, 0]
                Q = qs[arm]
            else:
                raise NotImplementedError()
                nxtv = G.vp.best_neighbor[currv]
                edge = G.edge(currv, nxtv)
                reward = G.ep.reward[edge]
                value = G.vp.value[nxtv]
                Q = reward + args.gamma * value
            new_stitches, new_advantages, n_stitches = get_possible_stitches(G,
                                                                             neighbors,
                                                                             stitches_tried,
                                                                             state_props,
                                                                             action_props,
                                                                             currv,
                                                                             childs[:, 0].astype(int),
                                                                             edges[:, -args.action_dim:],
                                                                             Q,
                                                                             args.rollout_chunk_size - total_stitches)  # NOQA
            stitches += new_stitches
            advantages += new_advantages
            total_stitches += n_stitches
            if total_stitches >= args.rollout_chunk_size:
                break
            t += 1
            currv = nxtv
        neps += 1
    stitches = np.concatenate(stitches, axis=0)
    advantages = np.concatenate(advantages, axis=0)[:, np.newaxis]
    data = np.concatenate([advantages, stitches], axis=1)
    if advantages.shape[0] > args.rollout_chunk_size:
        indices = np.argpartition(advantages[:, 0], args.rollout_chunk_size)[:args.rollout_chunk_size]
        data = data[indices, :]
    np.save(args.output_path, data)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
