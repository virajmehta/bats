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
                dataset[k] =  v[()]
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


def prepare_model_inputs_torch(obs, actions):
    return torch.cat([obs, actions], dim=-1)


def CEM_wrapper(row, ensemble_fn, load_ensemble_fun, obs_dim, action_dim, epsilon, quantile, **kwargs):
    global ENSEMBLE
    if ENSEMBLE is None:
        ENSEMBLE = load_ensemble_fun(ensemble_fn)
    return CEM(row, obs_dim, action_dim, ENSEMBLE, epsilon, quantile, **kwargs)


def CEM(obs_dim,
        action_dim,
        ensemble,
        epsilon,
        quantile,
        row,
        action_upper_bound=1,
        action_lower_bound=-1,
        initial_variance_divisor=4,
        max_iters=3,
        popsize=512,
        num_elites=128,
        alpha=0.25):
    '''
    attempts CEM optimization to plan a single step from the start state to the end state.
    initializes the mean with the init action.

    row: a numpy array of size D = 2 + 2 * obs_dim + action_dim, assumed to be of the form:
    start state (int); end state (int); start_obs (obs_dim floats); end_obs (obs_dim floats);
        init_action (action_dim floats);

    if successful, returns the action and predicted reward of the transition.
    if unsuccessful, returns None, None
    for now we assume that the actions are bounded in [-1, 1]
    '''
    row = torch.Tensor(row)
    start_state = row[2:2 + obs_dim]
    end_state = row[2 + obs_dim:2 + 2 * obs_dim]
    init_action = row[-action_dim:]
    action_dim = init_action.shape[0]
    mean = init_action
    var = torch.ones_like(mean) * ((action_upper_bound - action_lower_bound) / initial_variance_divisor) ** 2
    # X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))
    for i in range(max_iters):
        # lb_dist, ub_dist = mean - action_lower_bound, action_upper_bound - mean
        # constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

        # samples = X.rvs(size=[popsize, action_dim]) * np.sqrt(var) + mean
        samples = torch.fmod(torch.randn(size=(popsize, action_dim)), 2) * torch.sqrt(var) + mean
        samples = torch.clip(samples, action_lower_bound, action_upper_bound)
        start_states = start_state.repeat(popsize, 1)
        input_data = prepare_model_inputs_torch(start_states, samples)
        model_outputs = torch.stack([member.get_mean_logvar(input_data)[0] for member in ensemble])
        # this is because the dynamics models are written to predict the reward in the first component of the output
        # and the next state in all the following components
        state_outputs = model_outputs[:, :, 1:]
        reward_outputs = model_outputs[:, :, 0]
        displacements = state_outputs + start_states - end_state
        distances = torch.linalg.norm(displacements, dim=-1)
        quantiles = torch.quantile(distances, quantile, dim=0)
        if quantiles.min() < epsilon:
            # success!
            min_index = quantiles.argmin()
            return row[0], row[1], samples[min_index, :], reward_outputs[:, min_index].mean()
        elites = samples[torch.argsort(quantiles)[:num_elites], ...]
        new_mean = torch.mean(elites, axis=0)
        new_var = torch.var(elites, axis=0)
        mean = alpha * mean + (1 - alpha) * new_mean
        var = alpha * var + (1 - alpha) * new_var
    return None, None, None, None


def ceildiv(a, b):
    return -(-a // b)
