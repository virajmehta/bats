"""
Util for graph interaction with models.
"""
import numpy as np
from tqdm import tqdm


def make_qlearning_dataset(graph):
    data = {k: [] for k in ['observations', 'actions', 'rewards',
                            'next_observations', 'terminals']}
    for v in graph.iter_vertices():
        for iidx, oidx, r in graph.get_out_edges(v, eprops=[graph.ep.reward]):
            data['observations'].append(graph.vp.obs[v])
            data['actions'].append(graph.ep.action[graph.edge(iidx, oidx)])
            data['rewards'].append(r)
            data['next_observations'].append(graph.vp.obs[oidx])
            data['terminals'].append(graph.vp.terminal[v])
    data = {k: np.array(v) for k, v in data.items()}
    for k, v in data.items():
        if len(v.shape) == 1:
            data[k] = v.reshape(-1, 1)
    return data


def make_best_action_dataset(graph):
    data = {k: [] for k in ['observations', 'actions']}
    for v in graph.iter_vertices():
        data['observations'].append(graph.vp.obs[v])
        data['actions'].append(graph.ep.action[
            # graph.edge(v, graph.vp.best_child[v])])
            graph.edge(v, graph.vp.best_neighbor[v])])
    data = {k: np.array(v) for k, v in data.items()}
    for k, v in data.items():
        if len(v.shape) == 1:
            data[k] = v.reshape(-1, 1)
    return data


def make_boltzmann_policy_dataset(graph, n_collects,
                                  temperature=0.001,
                                  max_ep_len=1000,
                                  gamma=0.99,
                                  normalize_qs=True,
                                  n_val_collects=0,
                                  val_start_prop=0,
                                  any_state_is_start=False,
                                  starts=None,
                                  silent=False):
    """Collect a Q learning dataset by running boltzmann policy in MDP.
    Args:
        graph: The graph object.
        n_collects: The number of data point to collect for the training set.
        temperature: Boltzmann policy picks action with probability prop to
            exp(Q(s, a) / temp) at each state s.
        gamma: Discount factor.
        normalize_qs: Whether to normalize Q values for each state s. This
            is often necessary to avoid numerical error.
        n_val_collects: Number of data points to collect for a validation set.
        val_start_prop: [0, 1) percent of of starts to use for the validation
            set.
        starts: User provided start states. Otherwise will look for vertex
            property "start".
        silent: Whether to be silent.
    """
    data = {k: [] for k in ['observations', 'actions']}
    # Get the start states.
    if starts is None:
        if any_state_is_start:
            starts = np.arange(graph.num_vertices())
        else:
            starts = np.argwhere(graph.get_vertices(
                vprops=[graph.vp.start])[:, 1]).flatten()
    # Get separate into train and validation set starts.
    np.random.shuffle(starts)
    if len(starts) > 1 and val_start_prop > 0 and n_val_collects > 0:
        val_size = max(int(len(starts) * val_start_prop), 1)
        val_data = make_boltzmann_policy_dataset(
              graph=graph,
              n_collects=n_val_collects,
              temperature=temperature,
              max_ep_len=max_ep_len,
              gamma=gamma,
              normalize_qs=normalize_qs,
              n_val_collects=0,
              val_start_prop=0,
              starts=starts[:val_size],
              silent=False,
        )[0]
        starts = starts[val_size:]
    else:
        val_data = None
    n_added = 0
    if not silent:
        pbar = tqdm(total=n_collects)
    # Do Boltzmann rollouts.
    while n_added < n_collects:
        done = False
        t = 0
        currv = np.random.choice(starts)
        while not done and t < max_ep_len:
            if temperature > 0:
                childs = graph.get_out_neighbors(currv,
                        vprops=[graph.vp.value])
                        # vprops=[graph.vp.value, graph.vp.terminal])
                if len(childs) == 0:
                    break
                edges = graph.get_out_edges(currv, eprops=[graph.ep.reward])
                qs = edges[:, -1] + gamma * childs[:, 1]
                # qs = edges[:, -1] + gamma * childs[:, 1] * (1 - childs[:, 2])
                if normalize_qs:
                    minq, maxq = np.min(qs), np.max(qs)
                    if minq ==  maxq:
                        qs = np.ones(qs.shape)
                    else:
                        qs = (qs - minq) / (maxq - minq)
                probs = np.exp(qs / temperature)
                probs /= np.sum(probs)
                nxtv = np.random.choice(childs[:, 0], p=probs)
            else:
                nxtv = graph.vp.best_neighbor[currv]
                # nxtv = graph.vp.best_child[currv]
            data['observations'].append(np.array(graph.vp.obs[currv]))
            data['actions'].append(
                    np.array(graph.ep.action[graph.edge(currv, nxtv)]))
            # done = graph.vp.terminal[nxtv]
            done = False
            currv = nxtv
            n_added += 1
            t += 1
            if n_added >= n_collects:
                break
        if not silent:
            pbar.update(t)
    if not silent:
        pbar.close()
        print('Done collecting.')
    data = {k: np.vstack(v) for k, v in data.items()}
    for k, v in data.items():
        if len(v.shape) == 1:
            data[k] = v.reshape(-1, 1)
    return data, val_data


def get_best_policy_returns(
        graph,
        starts=None,
        gamma=1,
        horizon=1000,
        silent=False,
):
    """For each start, trace through the graph MDP to estimate value.
    Args:
        graph: The MDP as a graph.
        starts: User provided start states. Otherwise will look for vertex
            property "start".
        gamma: Discount factor.
        horizon: Time to run in the MDP.
    Returns: The list of tuples (return, start_obs).
    """
    returns = []
    # Get the start states.
    if starts is None:
        starts = np.argwhere(graph.get_vertices(
            vprops=[graph.vp.start])[:, 1]).flatten()
    # Do rollouts at start states.
    itr = starts if silent else tqdm(starts)
    for sidx in itr:
        start_ob = graph.vp.observation[sidx]
        currv = sidx
        ret = 0
        for t in range(horizon):
            nxtv = graph.vp.best_neighbor[currv]
            ret += gamma ** t * graph.ep.reward[graph.edge(currv, nxtv)]
            if graph.vp.terminal[nxtv]:
                break
            currv = nxtv
        retturns.append((ret, start_ob))
    return returns


def make_advantage_dataset(graph, gamma=0.99, suboptimal_only=False):
    data = {k: [] for k in ['observations', 'actions', 'advantages']}
    for v in graph.iter_vertices():
        for iidx, oidx, r in graph.get_out_edges(v, eprops=[graph.ep.reward]):
            advantage = r + gamma * graph.vp.value[oidx] - graph.vp.value[iidx]
            if suboptimal_only and np.abs(advantage) < 1e-6:
                continue
            data['observations'].append(graph.vp.obs[v])
            data['actions'].append(graph.ep.action[graph.edge(iidx, oidx)])
            data['advantages'].append(advantage)
    data = {k: np.array(v) for k, v in data.items()}
    for k, v in data.items():
        if len(v.shape) == 1:
            data[k] = v.reshape(-1, 1)
    return data
