"""
Utility for maze data.
"""
import numpy as np


def get_starts_from_graph(graph, env):
    # When env is made it is wrapped in TimeLimiter, hence the .env
    env = env.env
    obs = graph.vp.obs.get_2d_array(np.arange(env.observation_space.low.size))
    obs = obs.T
    diffs = np.array([obs - np.array([st[0], st[1], 0, 0])
                      for st in env.empty_and_goal_locations])
    is_starts = np.any(np.all(np.abs(diffs) < 0.1, axis=-1), 0)
    return np.argwhere(is_starts).flatten()
