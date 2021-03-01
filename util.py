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


def make_output_dir(name, overwrite, args):
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
    args = vars(args)
    for k, v in args.items():
        if type(v) is PosixPath:
            args[k] = str(v)
    with args_path.open('w') as f:
        json.dump(args, f)
    return dir_path


def get_offline_env(name, dataset_fraction, data_path=None):
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
        return []
    return [int(s) for s in string.split(',')]


def prepare_model_inputs(obs, actions):
    return torch.Tensor(np.hstack([obs, actions]))


def ceildiv(a, b):
    return -(-a // b)


def make_mujoco_resetter(env, task):
    if 'maze' in task:
        midpt = 2
    elif 'hopper' in task:
        midpt = 6
    elif 'halfcheetah' in task or 'walker' in task:
        midpt = 9
    else:
        NotImplementedError('No resetter implemented for %s.' % task)
    def resetter(obs):
        env.reset()
        env.env.set_state([obs[:midpt], obs[midpt:]])
    return resetter
