"""
Util for graph interaction with models.
"""
from collections import OrderedDict
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
                                  temperature=0.1,
                                  max_ep_len=1000,
                                  gamma=0.99,
                                  normalize_qs=True,
                                  n_val_collects=0,
                                  val_start_prop=0,
                                  any_state_is_start=False,
                                  only_add_real=False,
                                  get_unique_edges=True,
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
        only_add_real: Whether to only add real edges.
        get_unique_edges: Whether to only return unique edges to train on.
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
                vprops=[graph.vp.start_node])[:, 1]).flatten()
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
    n_imagined = 0
    n_edges = 0
    returns = []
    if not silent:
        pbar = tqdm(total=n_collects)
    # Do Boltzmann rollouts.
    edges = set()
    while n_edges < n_collects:
        done = False
        t = 0
        ret = 0
        currv = np.random.choice(starts)
        while not done and t < max_ep_len:
            # bstv = graph.vp.best_child[currv]
            bstv = graph.vp.best_neighbor[currv]
            if temperature > 0:
                childs = graph.get_out_neighbors(currv,
                        vprops=[graph.vp.value, graph.vp.terminal])
                if len(childs) == 0:
                    break
                edges = graph.get_out_edges(currv, eprops=[graph.ep.reward])
                qs = edges[:, -1] + gamma * childs[:, 1] * (1 - childs[:, 2])
                qs -= np.max(qs)
                probs = np.exp(qs / temperature)
                probs /= np.sum(probs)
                nxtv = np.random.choice(childs[:, 0], p=probs)
            else:
                nxtv = graph.vp.best_neighbor[currv]
            if nxtv < 1:
                break
            edge = graph.edge(currv, nxtv)
            is_imagined = graph.ep.imagined[edge]
            n_imagined += is_imagined
            n_edges += 1
            if not only_add_real or not graph.ep.imagined[edge]:
                if not get_unique_edges or (currv, nxtv) not in edges:
                    data['observations'].append(np.array(graph.vp.obs[currv]))
                    data['actions'].append(np.array(graph.ep.action[edge]))
                edges.add((currv, nxtv))
            done = graph.vp.terminal[nxtv]
            ret += graph.ep.reward[edge]
            currv = nxtv
            t += 1
            if n_edges >= n_collects:
                break
        if not silent:
            pbar.set_postfix(OrderedDict(
                Edges=n_edges,
                Imaginary=(n_imagined/ n_edges),
                Return=ret,
            ))
            pbar.update(t)
        returns.append(ret)
    if not silent:
        pbar.close()
        print('Done collecting.')
        print('Proportion imagined edges taken: %f' % (n_imagined / n_edges))
        print('Unique Edges: %d' % len(edges))
        print('Returns: %f +- %f' % (np.mean(returns), np.std(returns)))
    stats = OrderedDict(
        ImaginaryProp=n_imagined/n_edges,
        ReturnsAvg=np.mean(returns),
        ReturnsStd=np.std(returns),
        UniqueEdges=len(edges),
    )
    data = {k: np.vstack(v) for k, v in data.items()}
    for k, v in data.items():
        if len(v.shape) == 1:
            data[k] = v.reshape(-1, 1)
    return data, val_data, stats


def get_best_policy_returns(
        graph,
        starts=None,
        gamma=1,
        horizon=1000,
        ignore_terminals=False,
        silent=False,
):
    """For each start, trace through the graph MDP to estimate value.
    Args:
        graph: The MDP as a graph.
        starts: User provided start states. Otherwise will look for vertex
            property "start".
        gamma: Discount factor.
        horizon: Time to run in the MDP.
    Returns: List of tuples of the form (return, observations, actions)
    """
    to_return = []
    # Get the start states.
    if starts is None:
        starts = np.argwhere(graph.get_vertices(
            vprops=[graph.vp.start_node])[:, 1]).flatten()
    # Do rollouts at start states.
    if not silent:
        pbar = tqdm(total=len(starts))
    for sidx in starts:
        observations = []
        actions = []
        currv = sidx
        ret = 0
        t = 0
        while t < horizon:
            nxtv = graph.vp.best_neighbor[currv]
            if nxtv == -1:
                break
            observations.append(graph.vp.obs[currv])
            actions.append(graph.ep.action[graph.edge(currv, nxtv)])
            ret += gamma ** t * graph.ep.reward[graph.edge(currv, nxtv)]
            t += 1
            if not ignore_terminals:
                if graph.vp.terminal[nxtv]:
                    break
            currv = nxtv
        to_return.append((ret, np.vstack(observations), np.vstack(actions)))
        if not silent:
            pbar.set_postfix(OrderedDict(Return=ret))
            pbar.update(1)
    if not silent:
        pbar.close()
    return to_return


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


def get_value_thresholded_starts(
    graph,
    threshold,
    starts=None,
):
    if starts is None:
        starts = np.argwhere(graph.get_vertices(
            vprops=[graph.vp.start_node])[:, 1]).flatten()
    values = graph.vp.value.get_array()[starts].flatten()
    acceptable = np.argwhere(values > threshold).flatten()
    return starts[acceptable]
