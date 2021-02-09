"""
Util for graph interaction with models.
"""
import numpy as np


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
