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
    parser.add_argument('latent_dim', type=int)
    parser.add_argument('rollout_chunk_size', type=int)
    parser.add_argument('temperature', type=float)
    parser.add_argument('gamma', type=float)
    parser.add_argument('max_stitches', type=int)
    parser.add_argument('max_stitch_length', type=int)
    parser.add_argument('-ub', '--use_bisimulation', action='store_true')
    parser.add_argument('-ppa', '--pick_positive_adv',
                        action='store_true')
    parser.add_argument('--temp', type=float)
    return parser.parse_args()


def sample_truncated_geometric(p, max_idx):
    val = np.inf
    while val > max_idx:
        val = np.random.geometric(p)
    return val


def clip_possible_stitches(max_node_stitches, *args, **kwargs):
    possible_stitches, advantages = get_neighbor_stitches(*args)
    stitch_array = np.array([[s[0], s[1]] for s in possible_stitches])
    indices = np.unique(stitch_array, return_index=True, axis=0)[1]
    unique_stitches = []
    unique_advantages = advantages[indices]
    for idx in indices:
        unique_stitches.append(possible_stitches[idx])
    if len(unique_advantages) < max_node_stitches:
        return unique_stitches, unique_advantages
    indices = np.argpartition(unique_advantages, len(unique_advantages) - max_node_stitches)[-max_node_stitches:]
    best_stitches = []
    for idx in indices:
        best_stitches.append(unique_stitches[idx])
    best_advantages = unique_advantages[indices]
    return best_stitches, best_advantages


def get_future_stitches(G,
                        gamma,
                        all_neighbors,
                        stitches_tried,
                        state_props,
                        action_props,
                        actions_this_far,
                        startv,
                        start_obs,
                        currv,
                        Q,
                        max_stitches,
                        depth_remaining,
                        pick_positive_adv=True,
                        ):
    if depth_remaining == 0 or max_stitches <= 0:
        return [], np.array([])
    possible_stitches = []
    advantages = []
    # gives a numpy array num_neighbors * (2 + state_dim)
    neighbors = G.get_out_neighbors(currv, vprops=[*state_props, G.vp.value])
    # gives a numpy array num_neighbors * (3 + action_dim)
    edges = G.get_out_edges(currv, eprops=[*action_props, G.ep.reward])
    # Compute all advantages ahead of time and prioritize search by adding
    # them to list by advantage. Only matters if max_stitches is restrictive
    all_advantages = []
    for neigh, edge in zip(neighbors, edges):
        nidx = int(neigh[0])
        n_obs = neigh[1:-1]
        action = edge[2:-1]
        actions = actions_this_far + [action]
        reward = edge[-1]
        nv = neigh[-1]
        updated_Q = (Q - reward) / gamma
        advantage = nv - updated_Q
        all_advantages.append(advantage)
    ranking = np.argsort(all_advantages)[::-1]
    all_advantages = np.array(all_advantages)[ranking]
    neighbors = neighbors[ranking]
    edges = edges[ranking]
    for neigh, edge, advantage in zip(neighbors, edges, all_advantages):
        # how to calculate advantage / propagate rewards?
        adv_crit_met = advantage > 0 or not pick_positive_adv
        if adv_crit_met and (startv, nidx) not in stitches_tried and G.vp.real_node[startv] and G.vp.real_node[nidx] and (G.vp.traj_num[startv] != G.vp.traj_num[nidx] or G.vp.traj_num[startv] == -1):
            possible_stitches += [(startv, nidx, start_obs, n_obs, actions)]
            advantages.append(advantage)
            max_stitches -= 1
        output = get_future_stitches(G,
                                     gamma,
                                     all_neighbors,
                                     stitches_tried,
                                     state_props,
                                     action_props,
                                     actions,
                                     startv,
                                     start_obs,
                                     nidx,
                                     updated_Q,
                                     max_stitches,
                                     depth_remaining - 1,
                                     pick_positive_adv=pick_positive_adv)
        possible_stitches += output[0]
        advantages += list(output[1])
        max_stitches -= output[1].shape[0]
        if max_stitches <= 0:
            break
    return possible_stitches, np.array(advantages) * gamma


def get_neighbor_stitches(G,
                          gamma,
                          all_neighbors,
                          stitches_tried,
                          state_props,
                          action_props,
                          actions_this_far,
                          startv,
                          start_obs,
                          currv,
                          Q,
                          max_stitches,
                          depth_remaining,
                          pick_positive_adv=True):
    if depth_remaining == 0 or max_stitches <= 0:
        return [], np.array([])
    possible_stitches = []
    advantages = []
    nearest_neighbors = np.nonzero(all_neighbors[currv, :])[1]
    nn_values = G.vp.value.a[nearest_neighbors]
    # hack
    nn_indices = np.argsort(nn_values)[:10]
    nearest_neighbors = nearest_neighbors[nn_indices]
    for neighbor in nearest_neighbors:
        output = get_future_stitches(G,
                                     gamma,
                                     all_neighbors,
                                     stitches_tried,
                                     state_props,
                                     action_props,
                                     actions_this_far,
                                     currv,
                                     start_obs,
                                     neighbor,
                                     Q,
                                     max_stitches,
                                     depth_remaining,
                                     pick_positive_adv=pick_positive_adv)
        possible_stitches += output[0]
        advantages += list(output[1])
        max_stitches -= len(output[1])
        if max_stitches <= 0:
            return possible_stitches, np.array(advantages) * gamma
    # gives a numpy array num_neighbors * 2
    neighbors = G.get_out_neighbors(currv)
    # gives a numpy array num_neighbors * (3 + action_dim)
    edges = G.get_out_edges(currv, eprops=[*action_props, G.ep.reward])
    for child, edge in zip(neighbors, edges):
        action = edge[2:-1]
        actions = actions_this_far + [action]
        reward = edge[-1]
        updated_Q = (Q - reward) / gamma
        cneighbors = np.nonzero(all_neighbors[child, :])[1]
        cnn_values = G.vp.value.a[cneighbors]
        # hack
        cnn_indices = np.argsort(cnn_values)[:10]
        cneighbors = cneighbors[cnn_indices]
        for cn in cneighbors:
            value = G.vp.value[cn]
            cnobs = G.get_vertices(vprops=state_props)[cn, 1:]
            advantage = value - updated_Q
            adv_crit_met = advantage > 0 or not pick_positive_adv
            if adv_crit_met and (startv, cn) not in stitches_tried and G.vp.real_node[startv] and G.vp.real_node[cn]:
                possible_stitches += [(startv, cn, start_obs, cnobs, actions)]
                advantages += [advantage]
        output = get_neighbor_stitches(G,
                                       gamma,
                                       all_neighbors,
                                       stitches_tried,
                                       state_props,
                                       action_props,
                                       actions,
                                       startv,
                                       start_obs,
                                       currv,
                                       updated_Q,
                                       max_stitches,
                                       depth_remaining - 1,
                                       pick_positive_adv=pick_positive_adv)
        max_stitches -= len(output[1])
        if max_stitches <= 0:
            return possible_stitches, np.array(advantages) * gamma
        possible_stitches += output[0]
        advantages += list(output[1])
    return possible_stitches, np.array(advantages) * gamma


def main(args):
    G = load_graph(args.graph_path)
    neighbors = load_npz(args.neighbors_path)
    with args.stitches_tried_path.open('rb') as f:
        stitches_tried = pickle.load(f)
    start_states = np.load(args.start_states_path)
    stitches = []
    advantages = []
    action_props = ungroup_vector_property(G.ep.action, range(args.action_dim))
    if args.use_bisimulation:
        state_props = ungroup_vector_property(G.vp.z, range(args.latent_dim))
    else:
        state_props = ungroup_vector_property(G.vp.obs, range(args.obs_dim))
    # max_ep_len = 1000
    max_ep_len = 3000
    max_eps = args.rollout_chunk_size / 5
    # max_eps = 1000
    neps = 0
    total_stitches = 0
    if args.temp is not None:
        vals = G.vp.value[start_states]
        vals /= args.temp
        probs = np.exp(vals) / np.sum(np.exp(vals))
    else:
        probs = np.ones_like(start_states) / len(start_states)
    currv_obs_Q = []
    while total_stitches < args.rollout_chunk_size and neps < max_eps:
        t = 0
        currv = int(np.random.choice(start_states, p=probs))
        while t < max_ep_len:
            # do a Boltzmann rollout
            assert args.temperature > 0
            if G.vp.terminal[currv]:
                break
            childs = G.get_out_neighbors(currv, vprops=[G.vp.value])
            edges = G.get_out_edges(currv, eprops=[G.ep.reward, *action_props])
            Q = G.vp.value[currv]
            if len(childs) > 0:
                qs = edges[:, 2] + args.gamma * childs[:, 1]
                norm_qs = qs - np.max(qs)
                probs = np.exp(norm_qs / args.temperature)
                probs /= np.sum(probs)
                arm = np.random.choice(childs.shape[0], p=probs)
                nxtv = int(childs[arm, 0])
                Q = qs[arm]
            curr_obs = G.get_vertices(vprops=state_props)[currv, 1:]
            currv_obs_Q.append((currv, curr_obs, Q))
            if np.random.random() > args.gamma:
                new_stitches, new_advantages = clip_possible_stitches(
                        args.max_stitches,
                        G,
                        args.gamma,
                        neighbors,
                        stitches_tried,
                        state_props,
                        action_props,
                        [],
                        int(currv),
                        curr_obs,
                        int(currv),
                        Q,
                        args.rollout_chunk_size -
                        total_stitches,
                        args.max_stitch_length,
                        pick_positive_adv=args.pick_positive_adv)
                stitches += new_stitches
                advantages += list(new_advantages)
                total_stitches += len(new_advantages)
                currv_obs_Q = []
                break
            if len(childs) == 0:
                break
            t += 1
            currv = nxtv
        if len(currv_obs_Q) != 0:
            # need to sample and add stitches
            idx = sample_truncated_geometric(1 - args.gamma, len(currv_obs_Q)) - 1
            currv, curr_obs, Q = currv_obs_Q[idx]
            new_stitches, new_advantages = clip_possible_stitches(
                    args.max_stitches,
                    G,
                    args.gamma,
                    neighbors,
                    stitches_tried,
                    state_props,
                    action_props,
                    [],
                    int(currv),
                    curr_obs,
                    int(currv),
                    Q,
                    args.rollout_chunk_size -
                    total_stitches,
                    args.max_stitch_length,
                    pick_positive_adv=args.pick_positive_adv)
            stitches += new_stitches
            advantages += list(new_advantages)
            total_stitches += len(new_advantages)
            currv_obs_Q = []

        neps += 1
    advantages = np.array(advantages)
    '''
    I don't think this will be an issue, really:

    if advantages.shape[0] > args.rollout_chunk_size:
        indices = np.argpartition(advantages[:, 0], args.rollout_chunk_size)[:args.rollout_chunk_size]
        data = data[indices, :]
    '''
    with args.output_path.open('wb') as f:
        pickle.dump((stitches, advantages), f)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
