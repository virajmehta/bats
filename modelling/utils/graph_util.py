"""
Util for graph interaction with models.
"""
from collections import OrderedDict, defaultdict
import numpy as np
from tqdm import tqdm


ACTION_MAX = 2500000

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
                                  temperature=0.0,
                                  max_ep_len=1000,
                                  gamma=0.99,
                                  normalize_qs=True,
                                  n_val_collects=0,
                                  val_start_prop=0,
                                  val_selection_prob=0,
                                  any_state_is_start=False,
                                  only_add_real=False,
                                  val_only_add_real=False,
                                  get_unique_edges=True,
                                  include_reward_next_obs=False,
                                  starts=None,
                                  threshold_start_val=None,
                                  top_percent_starts=None,
                                  return_threshold=None,
                                  all_starts_once=False,
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
        val_selection_prob: Probability that an observed edge will be put
            into validation dataset.
        only_add_real: Whether to only add real edges.
        val_only_add_real: Whether to only add real edges to validation
            set.
        get_unique_edges: Whether to only return unique edges to train on.
        starts: User provided start states. Otherwise will look for vertex
            property "start".
        threshold_start_val: Value to threshold start states to pick.
        top_percent_starts: Percent of start states to take.
        return_threshold: The threshold on return values to place on starts.
        all_starts_once: Whether to just collect trajectories from all start
            states once instead of an amount of samples.
        silent: Whether to be silent.
    """
    data = defaultdict(list)
    # Get the start states.
    if starts is None:
        if any_state_is_start:
            starts = np.arange(graph.num_vertices())
        else:
            starts = np.argwhere(graph.get_vertices(
                vprops=[graph.vp.start_node])[:, 1]).flatten()
    if top_percent_starts is not None:
        starts = get_top_performing_starts(graph, top_percent_starts,
                                           starts=starts)
    if threshold_start_val is not None:
        starts = get_value_thresholded_starts(graph, threshold_start_val,
                                              starts=starts)
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
        val_data = defaultdict(list)
    n_imagined = 0
    n_edges = 0
    returns = []
    upper_returns = []
    if not silent:
        if all_starts_once:
            amt_on_bar = min(n_collects, len(starts) * max_ep_len)
        else:
            amt_on_bar = n_collects
        pbar = tqdm(total=amt_on_bar)
    # Do Boltzmann rollouts.
    edge_set = set()
    val_edge_set = set()
    running = True
    nxt_start_idx = -1
    while running:
        currdata = defaultdict(list)
        currval = defaultdict(list)
        curredgeset = set()
        currvalset = set()
        done = False
        t = 0
        ret = 0
        upper_ret = 0
        if all_starts_once:
            nxt_start_idx += 1
        else:
            nxt_start_idx = np.random.randint(len(starts))
        currv = starts[nxt_start_idx]
        lastact = None
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
                edgehash = (currv, nxtv)
                if edgehash in edge_set or edgehash in curredgeset:
                    toadd = currdata
                elif edgehash in val_edge_set or edgehash in currvalset:
                    toadd = currval
                else:
                    if ((not val_only_add_real or not graph.ep.imagined[edge])
                            and np.random.uniform() < val_selection_prob):
                        toadd = currval
                        currvalset.add(edgehash)
                    else:
                        toadd = currdata
                        curredgeset.add(edgehash)
                toadd['observations'].append(np.array(graph.vp.obs[currv]))
                toadd['actions'].append(np.array(graph.ep.action[edge]))
                if include_reward_next_obs:
                    toadd['next_observations'].append(np.array(graph.vp.obs[nxtv]))
                    toadd['rewards'].append(graph.ep.reward[edge])
                    toadd['terminals'].append(graph.vp.terminal[nxtv])
                    toadd['infos'].append(dict(
                        stitch_itr = graph.ep.stitch_itr[edge],
                        upper_reward = graph.ep.upper_reward[edge],
                        t = t,
                    ))
            done = graph.vp.terminal[nxtv]
            ret += graph.ep.reward[edge]
            upper_ret += graph.ep.upper_reward[edge]
            currv = nxtv
            t += 1
            if n_edges >= n_collects:
                break
        if not silent:
            pbar.set_postfix(OrderedDict(
                TrainEdges=len(edge_set),
                ValEdges=len(val_edge_set),
                Imaginary=(n_imagined / n_edges),
                Return=ret,
                UpperReturn=upper_ret,
            ))
            pbar.update(t)
        if return_threshold is None or ret > return_threshold:
            for full, curr in [(data, currdata), (val_data, currval)]:
                for k, v in curr.items():
                    full[k] += v
            edge_set = edge_set.union(curredgeset)
            val_edge_set = val_edge_set.union(currvalset)
            returns.append(ret)
            upper_returns.append(upper_ret)
        running = n_edges < n_collects
        if all_starts_once:
            running = running and nxt_start_idx < len(starts) - 1
    if not silent:
        pbar.close()
        print('Done collecting.')
        print('Proportion imagined edges taken: %f' % (n_imagined / n_edges))
        print('Unique Train Edges: %d' % len(edge_set))
        print('Unique Validation Edges: %d' % len(val_edge_set))
        print('Returns: %f +- %f' % (np.mean(returns), np.std(returns)))
        print('Upper Returns: %f +- %f' %
                (np.mean(upper_returns), np.std(upper_returns)))
    stats = OrderedDict(
        ImaginaryProp=n_imagined/n_edges,
        ReturnsAvg=np.mean(returns),
        ReturnsStd=np.std(returns),
        UpperReturnsAvg=np.mean(upper_returns),
        UpperReturnsStd=np.std(upper_returns),
        UniqueEdges=len(edge_set),
    )
    for ds in [data, val_data]:
        for k, v in ds.items():
            if k == 'infos':
                continue
            elif isinstance(v, list):
                v = np.vstack(v)
                if len(v.shape) == 1:
                    v = v.reshape(-1, 1)
                ds[k] = v
    return data, val_data, stats


def make_graph_consistent(
    graph,
    planning_quantile,
    epsilon_planning,
    stitch_itr=None,
    silent=False,
):
    """Go through the graph and remove any edges that are not consistent with
    the given hyperparameters.
    """
    stats = OrderedDict(EdgesRemoved=0, EdgesKept=0)
    # Loop through all of the edges in the graph.
    if not silent:
        pbar = tqdm(total=graph.num_edges())
    for inv, outv, im in graph.get_edges([graph.ep.imagined]):
        if im:
            edge = graph.edge(inv, outv)
            errs = graph.ep.model_errors[edge]
            should_remove =\
                    np.quantile(errs, planning_quantile) > epsilon_planning
            if stitch_itr is not None:
                should_remove = (should_remove
                        or graph.ep.stitch_itr[edge] <= stitch_itr)
            if should_remove:
                stats['EdgesRemoved'] += 1
                graph.remove_edge(graph.edge(inv, outv))
            else:
                stats['EdgesKept'] += 1
        if not silent:
            pbar.update(1)
            pbar.set_postfix(stats)
    if not silent:
        pbar.close()
    return graph, stats


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

def get_top_performing_starts(
    graph,
    top_percent,
    starts=None,
):
    if starts is None:
        starts = np.argwhere(graph.get_vertices(
            vprops=[graph.vp.start_node])[:, 1]).flatten()
    values = graph.vp.value.get_array()[starts].flatten()
    srtidxs = np.argsort(values)
    acceptable = srtidxs[-int(len(srtidxs) * top_percent):]
    return starts[acceptable]

def get_return_thresholded_starts(
    graph,
    threshold,
    horizon,
    starts=None,
):
    print('Filtering out bad states...')
    if starts is None:
        starts = np.argwhere(graph.get_vertices(
            vprops=[graph.vp.start_node])[:, 1]).flatten()
    best_pol = get_best_policy_returns(graph, starts, horizon=horizon)
    returns = np.array([bp[0] for bp in best_pol])
    valididxs = np.argwhere(returns >= threshold).flatten()
    if len(valididxs) == 0:
        print('No starts meet threshold requirement! Returning all starts!')
        return starts
    print('Keeping %d/%d states.' % (len(valididxs), len(starts)))
    return starts[valididxs].flatten()
