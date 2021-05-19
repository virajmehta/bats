import json
from pathlib import Path, PosixPath
from shutil import rmtree
from tqdm import trange
import numpy as np
# from scipy import stats
import d4rl
import gym
import h5py
import torch
from ipdb import set_trace as db

DATA_DIR = 'experiments'
ENSEMBLE = None


def get_output_dir(name):
    return Path(DATA_DIR) / name


def make_output_dir(name, overwrite, args, dir_path=None):
    if dir_path is None:
        dir_path = get_output_dir(name)
    if dir_path.exists():
        if overwrite:
            rmtree(dir_path)
        else:
            raise ValueError(f"{dir_path} already exists! Use a different name")
    dir_path.mkdir()
    input_path = dir_path / 'input'
    input_path.mkdir()
    output_path = dir_path / 'output'
    output_path.mkdir()
    args_path = dir_path / 'args.json'
    if not isinstance(args, dict):
        args = vars(args)
    print(args)
    for k, v in args.items():
        if type(v) is PosixPath:
            args[k] = str(v)
    with args_path.open('w') as f:
        json.dump(args, f)
    return dir_path


def get_offline_env(name, dataset_fraction, data_path=None):
    if name is None:
        env = None
    else:
        env = gym.make(name)
    if data_path is None:
        dataset = d4rl.qlearning_dataset(env)
    else:
        dataset = {}
        with h5py.File(str(data_path), 'r') as hdata:
            for k, v in hdata.items():
                dataset[k] = v[()]
    for name in dataset:
        item = dataset[name]
        size = item.shape[0]
        keep = int(size * dataset_fraction)
        dataset[name] = item[:keep, ...]
    return env, dataset


def flatten_dataset(dataset, tqdm=True):
    '''
    if N is the dataset size
    expects dataset to be a dict with contents:
        'observations': N * obsdim ndarray
        'actions': N * action dim ndarray
        'rewards': N ndarray
        'terminals': N ndarray
        'timeouts': N ndarray

    returns a dataset that's a dict with contents
    (will use M in place of N since we'll lose the last timestep of every episode flattening)
        'observations': M * obsdim ndarray
        'actions': M * action dim ndarray
        'next_observations': M * obsdim ndarray
        'rewards': M ndarray
        'terminals': M ndarray --  it's false unless there's a real terminal
    '''
    print("Flattening dataset")
    ndata = dataset['rewards'].shape[0]
    iterator = trange(ndata) if tqdm else range(ndata)
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    terminals = dataset['terminals']
    timeouts = dataset['timeouts']
    new_observations = []
    new_actions = []
    new_next_observations = []
    new_rewards = []
    new_terminals = []
    for i in iterator:
        if terminals[i] or timeouts[i] or i + 1 == ndata:
            continue
        new_observations.append(observations[i, :])
        new_actions.append(actions[i, :])
        new_next_observations.append(observations[i + 1, :])
        new_rewards.append(rewards[i])
        new_terminals.append(terminals[i])

    new_observations = np.stack(new_observations)
    new_actions = np.stack(new_actions)
    new_next_observations = np.stack(new_next_observations)
    new_rewards = np.array(new_rewards)
    new_terminals = np.array(new_terminals)
    new_dataset = {
            'observations': new_observations,
            'actions': new_actions,
            'next_observations': new_next_observations,
            'rewards': new_rewards,
            'terminals': new_terminals,
            }

    return new_dataset


def get_return(episode):
    return sum(episode['rewards'])


def rollout(policy, environment):
    observations = []
    actions = []
    rewards = []
    infos = []
    done = False
    obs = environment.reset()
    observations.append(obs)
    while not done:
        obs = torch.Tensor(obs)
        with torch.no_grad():
            action = policy(obs)
            action = action.cpu().numpy()
        actions.append(action)
        obs, reward, done, info = environment.step(action)
        observations.append(obs)
        rewards.append(reward)
        infos.append(info)
    episode = dict(observations=observations, actions=actions, rewards=rewards, infos=infos)
    return episode


def dict_append(d, k, v):
    """Append item to list if already exists, otherwise make new list."""
    if k in d:
        d[k].append(v)
    else:
        d[k] = [v]


def s2i(string):
    """Make a comma separated string of ints into a list of ints."""
    if ',' not in string:
        if len(string) > 0:
            return [int(string)]
        else:
            return []
    return [int(s) for s in string.split(',')]


def s2f(string):
    """Make a comma separated string of ints into a list of ints."""
    if ',' not in string:
        if len(string) > 0:
            return [float(string)]
        else:
            return []
    return [float(s) for s in string.split(',')]


def prepare_model_inputs(obs, actions):
    return torch.Tensor(np.hstack([obs, actions]))


def ceildiv(a, b):
    return -(-a // b)


def make_mujoco_resetter(env, task):
    append_zero = True
    if 'maze' in task.lower():
        midpt = 2
        append_zero = False
    elif 'hopper' in task.lower():
        midpt = 6
    elif 'halfcheetah' in task.lower() or 'walker' in task.lower():
        midpt = 9
    else:
        NotImplementedError('No resetter implemented for %s.' % task)
    def resetter(obs):
        env.reset()
        if append_zero:
            obs = np.append(0, obs)
        env.env.set_state(obs[:midpt], obs[midpt:])
    return resetter


def get_starts_from_graph(graph, env, env_name):
    # When env is made it is wrapped in TimeLimiter, hence the .env
    env = env.env
    if env_name.startswith('maze'):
        obs = graph.vp.obs.get_2d_array(np.arange(env.observation_space.low.size))
        obs = obs.T
        diffs = np.array([obs - np.array([st[0], st[1], 0, 0])
                        for st in env.empty_and_goal_locations])
        start_conditions = np.all(np.abs(diffs) < 0.1, axis=-1)
        no_start_rows = np.argwhere(np.sum(start_conditions, axis=1) == 0)
        loose_conditions = np.all(np.abs(diffs[..., :2]) < 0.1, axis=-1)
        start_conditions[no_start_rows] = loose_conditions[no_start_rows]
        is_starts = np.any(start_conditions, 0)
        return np.argwhere(is_starts).flatten()
    elif env_name.startswith('halfcheetah') or env_name.startswith('walker') or env_name.startswith('hopper'):
        dataset = env.get_dataset()
        ends = dataset['timeouts'].astype(bool) | dataset['terminals'].astype(bool)
        ends_dense = np.nonzero(ends)[0]
        start_states = np.concatenate([[0], ends_dense + 1])
        if start_states[-1] >= graph.get_vertices().shape[0]:
            start_states = start_states[:-1]
        return start_states
    else:
        raise NotImplementedError('env {env_name} not supported for start state detection')

class BlankEnv(object):

    def __init__(self, obs_dim=20):
        self.obs_dim = obs_dim
        self.observation_space = gym.spaces.Box(
            low=-1 * np.ones(obs_dim),
            high=np.ones(obs_dim),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32,
        )

    def step(self, act):
        return np.ones(self.obs_dim), 0, False, {}

    def reset(self):
        return np.ones(self.obs_dim)
