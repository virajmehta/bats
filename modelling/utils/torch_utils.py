"""
Utils functions for torch.

Author: Ian Char
Date: 8/26/2020
"""
from typing import Any, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from modelling.utils.terminal_functions import get_terminal_function,\
        no_terminal


torch_device = 'cpu'


def set_cuda_device(device: str) -> None:
    """Set the cuda device to use. If blank string, use CPU."""
    global torch_device
    if device == '':
        torch_device = 'cpu'
    else:
        torch_device = 'cuda:' + device


def torch_to(torch_obj: Any) -> Any:
    """Put the torch object onto a device."""
    return torch_obj.float().to(torch_device)


def torchify_to(obj: Any) -> torch.Tensor:
    if type(obj) is not torch.Tensor:
        obj = torch.Tensor(obj)
    return torch_to(obj)


def create_simple_net(
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int],
        activation,
):
    """Return a simple sequential neural network."""
    layer_sizes = [in_dim] + hidden + [out_dim]
    layers = [nn.Linear(layer_sizes[0], layer_sizes[1])]
    for lidx in range(1, len(layer_sizes) - 1):
        layers.append(activation())
        layers.append(nn.Linear(layer_sizes[lidx], layer_sizes[lidx + 1]))
    return nn.Sequential(*layers)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Do the reparameterization trick."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return torch_to(eps * std + mu)


def unroll(env, policy, max_ep_len=float('inf'), replay_buffer=None):
    t = 0
    done = False
    s = env.reset()
    ret = 0
    while not done and t < max_ep_len:
        with torch.no_grad():
            a = policy.get_action(torch.Tensor(s),
                                  deterministic=True).cpu().numpy()
        n, r, done, _ = env.step(a)
        if replay_buffer is not None:
            replay_buffer['observations'].append(s)
            replay_buffer['actions'].append(a)
            replay_buffer['rewards'].append(r)
            replay_buffer['next_observations'].append(n)
            replay_buffer['terminals'].append(done)
        ret += r
        s = n
    return ret


def swish(x):
    return x * torch.sigmoid(x)


def arctanh(x):
    return 0.5 * (torch.log(1 + x) - torch.log(1 - x))


class IteratedDataLoader(object):

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iterator = iter(dataloader)

    def next(self):
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._dataloader)
            batch = next(self._iterator)
        return batch


class ModelUnroller(object):

    def __init__(self, env_name, model, mean_transitions=True):
        """
        Args:
            env_name: Name of environment.
            model: Either list of PNN or bisim model.
            mean_transitions: Whether to sample or take mean estimate for
                transition.
        """
        self.is_bisim = not isinstance(model, list)
        if self.is_bisim:
            self.terminal_function = no_terminal
            self.n_ensemble = model.num_dyn_nets
        else:
            self.terminal_function = get_terminal_function(env_name)
            self.n_ensemble = len(model)
        self.model = model
        self.mean_transitions = mean_transitions

    def model_unroll(self, start_states, actions):
        """Unroll for multiple trajectories at once.
        Args:
            start_states: The start states to unroll at as ndarray
                w shape (num_starts, obs_dim).
            actions: The actions as (num_starts, horizon, action_dim)
        """
        horizon = actions.shape[1]
        act_dim = actions.shape[-1]
        obs_dim = start_states.shape[-1]

        obs = np.zeros((start_states.shape[0], self.n_ensemble, horizon + 1,
                        start_states.shape[1]))
        obs[:, :, 0, :] = torch.unsqueeze(start_states, 1).repeat(1, self.n_ensemble, 1)
        rewards = np.zeros((start_states.shape[0], self.n_ensemble, horizon))
        terminals = np.full((start_states.shape[0], self.n_ensemble, horizon), True)
        is_running = np.full((start_states.shape[0], self.n_ensemble), True)
        for hidx in range(horizon):
            acts = torch.unsqueeze(actions[:, hidx], 1).repeat(1, self.n_ensemble, 1)
            flat_acts = acts.reshape(-1, act_dim)
            flat_obs = obs[:, :, hidx, :].reshape(-1, obs_dim)
            # Roll all states forward.
            nxt_info = self.get_next_transition(flat_obs, flat_acts)
            # need to get the model predictions from the same model to the same model
            deltas = nxt_info['deltas'].reshape(-1, self.n_ensemble, self.n_ensemble, obs_dim)[:,
                                                    range(self.n_ensemble), range(self.n_ensemble), :]
            obs[:, :, hidx+1, :] = obs[:, :, hidx, :] + deltas
            new_rewards = nxt_info['rewards'].T.reshape(-1, self.n_ensemble, self.n_ensemble)[:,
                                                    range(self.n_ensemble), range(self.n_ensemble)]
            rewards[..., hidx] = new_rewards
            terminal_inputs = obs[:, :, hidx + 1, :].reshape(-1, obs_dim)
            terminals[..., hidx] = self.terminal_function(terminal_inputs).reshape(-1, self.n_ensemble)
            is_running = np.logical_and(is_running, ~terminals[..., hidx])
            if np.sum(is_running) == 0:
                break
        return obs, actions, rewards, np.any(terminals, axis=-1)

    def get_next_transition(self, obs, acts):
        net_ins = torch.cat([
            torch.Tensor(obs),
            torch.Tensor(acts),
        ], dim=1)
        if self.is_bisim:
            with torch.no_grad():
                mean, logvar = self.model.get_mean_logvar(net_ins)
            means = mean.numpy()
            stds = (0.5 * logvar).exp().numpy()
        else:
            means, stds = [], []
            with torch.no_grad():
                for ens in self.model:
                    ens_mean, ens_logvar = ens.get_mean_logvar(net_ins)
                    means.append(ens_mean.cpu().numpy())
                    stds.append(np.exp(ens_logvar.cpu().numpy() / 2))
            means, stds = np.asarray(means), np.asarray(stds)
        '''
        Viraj: I want all the model outputs
        # Randomly select one of the models to get the next obs from.
        if self.mean_transitions:
            samples = means[0]
        else:
            samples = np.random.normal(means[0], stds[0])
        '''
        if self.mean_transitions:
            samples = means
        else:
            samples = np.random.normal(means, stds)
        # Get penalty term.
        rewards, deltas = samples[..., 0], samples[..., 1:]
        return {'deltas': deltas, 'rewards': rewards}


class Standardizer(nn.Module):

    def __init__(
            self,
            standardizers: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ):
        """Constructor."""
        super(Standardizer, self).__init__()
        for pairnum, pair in enumerate(standardizers):
            self.register_buffer('mean_%d' % pairnum, pair[0])
            self.register_buffer('std_%d' % pairnum, pair[1])

    def get_stats(self, data_loc: int):
        mu = getattr(self, 'mean_%d' % data_loc)
        sigma = getattr(self, 'std_%d' % data_loc)
        return mu, sigma

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor, data_loc: int):
        """Set the mean and std for a data."""
        setattr(self, 'mean_%d' % data_loc, mean)
        setattr(self, 'std_%d' % data_loc, std)

    def standardize(
            self,
            data: torch.Tensor,
            data_loc: int,
    ):
        mu, sigma = self.get_stats(data_loc)
        sigma[sigma < 1e-6] = 1
        return (data - mu) / sigma

    def unstandardize(
            self,
            data: torch.Tensor,
            data_loc: int,
    ):
        mu, sigma = self.get_stats(data_loc)
        return data * sigma + mu
