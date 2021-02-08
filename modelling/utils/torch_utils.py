"""
Utils functions for torch.

Author: Ian Char
Date: 8/26/2020
"""
from typing import Any, Sequence, Tuple

import torch
from torch import nn


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
    return eps * std + mu


def unroll(env, policy, max_ep_len=float('inf')):
    t = 0
    done = False
    s = env.reset()
    ret = 0
    while not done and t < max_ep_len:
        with torch.no_grad():
            a = policy.get_action(torch.Tensor(s),
                                  deterministic=True).cpu().numpy()
        n, r, done, _ = env.step(a)
        tup = (s, a, r, n, done)
        ret += r
        s = n
    return ret


def swish(x):
    return x * torch.sigmoid(x)


def arctanh(x):
    return 0.5 * (torch.log(1 + x) - torch.log(1 - x))


class Standardizer(nn.Module):

    def __init__(
            self,
            standardizers:Sequence[Tuple[torch.Tensor, torch.Tensor]],
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
        return (data - mu) / sigma

    def unstandardize(
            self,
            data: torch.Tensor,
            data_loc: int,
    ):
        mu, sigma = self.get_stats(data_loc)
        return data * sigma + mu
