import json
from pathlib import Path, PosixPath
from shutil import rmtree
from tqdm import trange
import numpy as np
from scipy import stats
import d4rl
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
    for k, v in args.items():
        if type(v) is PosixPath:
            args[k] = str(v)
    with args_path.open('w') as f:
        json.dump(args, f)
    return dir_path


def get_offline_env(name, dataset_fraction):
    env = gym.make(name)
    dataset = d4rl.qlearning_dataset(env)
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


def CEM(start_state, end_state, init_action, ensemble, epsilon, quantile, **kwargs):
    '''
    attempts CEM optimization to plan a single step from the start state to the end state.
    initializes the mean with the init action.

    if successful, returns the action and predicted reward of the transition.
    if unsuccessful, returns None, None
    for now we assume that the actions are bounded in [-1, 1]
    '''
    action_upper_bound = kwargs.get('action_upper_bound', 1.)
    action_lower_bound = kwargs.get('action_lower_bound', -1.)
    initial_variance_divisor = kwargs.get('initial_variance_divisor', 4)
    max_iters = kwargs.get('max_iters', 5)
    popsize = kwargs.get('popsize', 256)
    num_elites = kwargs.get('num_elites', 64)
    alpha = kwargs.get('alpha', 0.25)
    action_dim = init_action.shape[0]
    mean = init_action
    var = np.ones_like(mean) * ((action_upper_bound - action_lower_bound) / initial_variance_divisor) ** 2
    X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))
    for i in range(max_iters):
        lb_dist, ub_dist = mean - action_lower_bound, action_upper_bound - mean
        constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

        samples = X.rvs(size=[popsize, action_dim]) * np.sqrt(constrained_var) + mean
        start_states = np.tile(start_state, (popsize, 1))
        input_data = prepare_model_inputs(start_states, samples)
        model_outputs = np.stack([member.get_mean_logvar(input_data)[0].cpu() for member in ensemble])
        # this is because the dynamics models are written to predict the reward in the first component of the output
        # and the next state in all the following components
        state_outputs = model_outputs[:, :, 1:]
        reward_outputs = model_outputs[:, :, 0]
        displacements = state_outputs - end_state
        distances = np.linalg.norm(displacements, axis=-1)
        quantiles = np.quantile(distances, quantile, axis=0)
        if quantiles.min() < epsilon:
            # success!
            min_index = quantiles.argmin()
            return samples[min_index, :], reward_outputs[:, min_index].mean()
        elites = samples[np.argsort(quantiles)[:num_elites], ...]
        new_mean = np.mean(elites, axis=0)
        new_var = np.var(elites, axis=0)
        mean = alpha * mean + (1 - alpha) * new_mean
        var = alpha * var + (1 - alpha) * new_var
    return None, None
