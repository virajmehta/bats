"""
Functions for labelling terminal states.

Inputs to the function are the states: ndarray of shape (num_states, state_dim)
"""
import numpy as np

def get_terminal_function(env):
    if 'hopper' in env.lower():
        return hopper_terminal
    elif 'walker' in env.lower():
        return walker_terminal
    if 'mountaincar' in env.lower():
        return mc_terminal
    if 'antmaze' in env.lower():
        return antmaze_terminal()
    return no_terminal

def no_terminal(states):
    states = _add_axis_if_needed(states)
    return np.full(states.shape[0], False)

def mc_terminal(states):
    goal_position = 0.45
    goal_velocity = 0.
    states = _add_axis_if_needed(states)
    positions = states[:, 0]
    velocities = states[:, 1]
    done = np.logical_and.reduce(positions >= goal_position, velocities >= goal_velocity)
    return done

def antmaze_terminal(states):
    states = _add_axis_if_needed(states)
    pos = states[:, :2]
    target = np.array([[0.35, 8.35]])
    displacements = pos - target
    norms = np.linalg.norm(displacements, axis=1)
    return norms <= 0.5


def hopper_terminal(states):
    """As written in MOPO code base."""
    states = _add_axis_if_needed(states)
    height = states[:, 0]
    angle = states[:, 1]
    return np.logical_or.reduce([
        ~np.isfinite(states).all(axis=-1),
        np.abs(states[:, 1:] >= 100).all(axis=-1),
        height <= 0.7,
        np.abs(angle) >= 0.2,
    ])

def walker_terminal(states):
    """As written in MOPO code base."""
    states = _add_axis_if_needed(states)
    height = states[:, 0]
    angle = states[:, 1]
    return np.logical_or.reduce([
        height <= 0.8,
        height >= 2.0,
        angle <= -1.0,
        angle >= 1.0,
    ])

def _add_axis_if_needed(states):
    if len(states.shape) == 1:
        states = states[np.newaxis, ...]
    return states
