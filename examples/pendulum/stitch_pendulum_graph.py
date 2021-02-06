"""
Create a stitched together pendulum graph.
"""
import argparse
import os

from graph_tool import Graph
from graph_tool.draw import graph_draw
from gym.envs.classic_control.pendulum import angle_normalize
import h5py
import numpy as np
from tqdm import tqdm


def run(args):
    # Set up save directory.
    os.makedirs(args.save_dir, exist_ok=args.save_dir == 'test')
    # Load in data.
    dataset = load_data(args)
    # Create the base graph and save off.
    graph, start_idxs = make_base_graph(dataset)
    graph.save(os.path.join(args.save_dir, 'base_graph.gt'))
    graph_draw(graph, output=os.path.join(args.save_dir, 'base_graph.png'),
               fmt='png')
    np.save(os.path.join(args.save_dir, 'start_idxs.npy'), start_idxs)
    # Create the stitched graph and save off.
    stitch_graph(dataset, graph)
    graph.save(os.path.join(args.save_dir, 'stitched_graph.gt'))
    graph_draw(graph, output=os.path.join(args.save_dir, 'sitched_graph.png'),
               fmt='png')
    print('Done.')


def load_data(args):
    dataset = {}
    with h5py.File(args.data_path, 'r') as hdata:
        for k, v in hdata.items():
            dataset[k] = v[()]
    return dataset


def make_base_graph(dataset):
    # Set up basic graph properties.
    graph = Graph()
    graph.vp.vertex_idx = graph.new_vertex_property('int')
    graph.vp.obs = graph.new_vertex_property('vector<float>')
    graph.vp.value = graph.new_vertex_property('float')
    graph.vp.best_action_idx = graph.new_vertex_property('int')
    graph.vp.start = graph.new_vertex_property('bool')
    graph.vp.terminal = graph.new_vertex_property('bool')
    graph.vp.traj = graph.new_vertex_property('int')
    graph.vp.original = graph.new_vertex_property('bool')
    graph.ep.action = graph.new_edge_property('float')
    graph.ep.reward = graph.new_edge_property('float')
    graph.ep.original = graph.new_edge_property('bool')
    # Add a vertex and edges for datasets.
    node_idx = 0
    traj_idx = 0
    num_data_pts = len(dataset['observations'])
    to_node, from_node = None, None
    start_idxs = []
    for idx in range(num_data_pts):
        # If this is the start add the from node.
        if from_node is None:
            traj_idx += 1
            from_node = add_blank_node(
                    dataset['observations'][idx],
                    node_idx,
                    graph,
                    start=True,
                    traj=traj_idx,
            )
            start_idxs.append(node_idx)
            node_idx += 1
        # Add the to node.
        to_node = add_blank_node(
            dataset['next_observations'][idx],
            node_idx,
            graph,
            start=False,
            terminal=dataset['ends'][idx],
            traj=traj_idx,
        )
        node_idx += 1
        # Add the edge between the to and from nodes.
        edge = graph.add_edge(from_node, to_node)
        graph.ep.action[edge] = float(dataset['actions'][idx])
        graph.ep.reward[edge] = float(dataset['rewards'][idx])
        graph.ep.original[edge] = True
        # Load for next iteration.
        if dataset['ends'][idx]:
            from_node = None
        else:
            from_node = to_node
        to_node = None
    return graph, np.array(start_idxs)


def stitch_graph(dataset, graph):
    all_obs, nxt_obs = [], []
    for idx, ob in enumerate(dataset['observations']):
        all_obs.append(ob)
        nxt = dataset['next_observations'][idx]
        nxt_obs.append(nxt)
        if dataset['ends'][idx]:
            all_obs.append(nxt)
            nxt_obs.append(np.array([0, 0, -100]))
    all_obs = np.vstack(all_obs)
    nxt_obs = np.vstack(nxt_obs)
    all_states = obs_to_state(all_obs)
    nxt_states = obs_to_state(nxt_obs)
    num_og_states = len(all_states)
    # Loop through
    new_states = []
    new_idx = num_og_states
    pbar = tqdm(total=num_og_states ** 2)
    for fidx, st in enumerate(all_states):
        for tidx, parent in enumerate(all_states):
            child = nxt_states[tidx]
            if (tidx + 1 < num_og_states
                    and np.all(child == all_states[tidx + 1])
                    and abs(fidx - tidx) > 2):
                add1, add2 = try_to_stitch(st, parent, child)
                if add1 is not None:
                    new_node = add_blank_node(add1[1], new_idx, graph,
                                              original=False)
                    new_states.append(add1[1])
                    edge = graph.add_edge(graph.vertex(fidx), new_node)
                    graph.ep.action[edge] = float(add1[2])
                    graph.ep.reward[edge] = float(add1[3])
                    graph.ep.original[edge] = False
                    edge = graph.add_edge(new_node, graph.vertex(tidx + 1))
                    graph.ep.action[edge] = float(add2[2])
                    graph.ep.reward[edge] = float(add2[3])
                    graph.ep.original[edge] = False
        pbar.set_postfix_str('Num Added: %d' % len(new_states))
        pbar.update(num_og_states)
    return new_states


def try_to_stitch(node, parent, child):
    # Constants.
    g = 10.
    m = 1.
    l = 1.
    dt = 0.05
    max_torque = 2.
    # Input parameters.
    ntheta, ndot = node
    ptheta, pdot = parent
    ctheta, cdot = child
    # Compute actions.
    act1 = (m * l ** 2 / 3) * ((ptheta - ntheta) / dt ** 2 - ndot / dt
                               + 3 * g / (2 * l) * np.sin(ntheta + np.pi))
    if np.abs(act1) > max_torque:
        return None, None
    nxttheta = ptheta
    nxtdot = ndot + (-3 * g / (2 * l) * np.sin(ntheta + np.pi)
                     + 3 / (m * l ** 2) * act1) * dt
    act2 = (m * l ** 2 / 3) * ((cdot - nxtdot) / dt
                               + 3 * g / (2 * l) * np.sin(nxttheta + np.pi))
    if np.abs(act2) > max_torque:
        return None, None
    cost1 = (angle_normalize(ntheta) ** 2 + 0.1 * ndot ** 2
             + 0.001 * (act1 ** 2))
    cost2 = (angle_normalize(nxttheta) ** 2 + 0.1 * nxtdot ** 2
             + 0.001 * (act2 ** 2))
    nxt = np.array([nxttheta, nxtdot])
    return ((state_to_obs(node).flatten(), state_to_obs(nxt).flatten(),
                act1 / 2, -cost1),
            (state_to_obs(nxt).flatten(), state_to_obs(child).flatten(),
                act2 / 2, -cost2))


def obs_to_state(obs):
    if len(obs.shape) == 1:
        obs = obs[np.newaxis]
    thetas = np.arccos(obs[:, 0]) * np.sign(np.arcsin(obs[:, 1]))
    return np.hstack([
        thetas.reshape(-1, 1),
        obs[:, -1].reshape(-1, 1),
    ])


def state_to_obs(state):
    if len(state.shape) == 1:
        state = state[np.newaxis]
    return np.hstack([
        np.cos(state[:, 0]).reshape(-1, 1),
        np.sin(state[:, 0]).reshape(-1, 1),
        state[:, -1].reshape(-1, 1),
    ])


def add_blank_node(obs, node_idx, graph,
                   start=False, terminal=False, original=True, traj=-1):
    new_node = graph.add_vertex()
    graph.vp.vertex_idx[new_node] = node_idx
    graph.vp.obs[new_node] = obs
    graph.vp.value[new_node] = 0
    graph.vp.best_action_idx[new_node] = -1
    graph.vp.start[new_node] = start
    graph.vp.terminal[new_node] = terminal
    graph.vp.traj[new_node] = traj
    graph.vp.original[new_node] = original
    return new_node


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir')
    parser.add_argument('--data_path')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    run(args)
