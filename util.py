import json
from pathlib import Path
from shutil import rmtree
from tqdm import trange
import numpy as np
import d4rl  # NOQA
import gym
import torch

DATA_DIR = 'experiments'


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
    args_path = dir_path / 'args.json'
    args = vars(args)
    with args_path.open('w') as f:
        json.dump(args, f)
    return dir_path


def get_offline_env(name):
    env = gym.make(name)
    dataset = env.get_dataset()
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


def rollout(policy, environment):
    observations = []
    actions = []
    rewards = []
    infos = []
    done = False
    obs = environment.reset()
    observations.append(obs)
    while not done:
        action = policy(obs)
        actions.append(action)
        obs, reward, done, info = environment.step(action)
        observations.append(obs)
        rewards.append(reward)
        infos.append(info)
    episode = dict(observations=observations, actions=actions, rewards=rewards, infos=infos)
    return episode


def train_mlp(X, Y, val_X=None, val_Y=None, **kwargs):
    '''
    X: numpy array (examples, feature_dim)
    Y: numpy array (examples, target_dim)
    val_X: same
    val_Y: same
    '''
    X = torch.Tensor(X)
    Y = torch.Tensor(Y)
    train_dataset = torch.utils.data.TensorDataset(X, Y)
    train_loader = torch.utils.data.DataLoader(train_dataset)
    if val_X and val_Y:
        val_X = torch.Tensor(val_X)
        val_Y = torch.Tensor(val_Y)
        val_dataset = torch.utils.data.TensorDataset(val_X, val_Y)
        val_loader = torch.utils.data.DataLoader(val_dataset)
    else:
        val_loader = None


