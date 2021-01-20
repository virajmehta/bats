"""
Standard neural network architectures.

Author: Ian Char
Date: 11/13/2020
"""
import abc
from typing import Optional, Sequence, Tuple

import torch


class MLP(torch.nn.Module):
    """MLP Network."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        output_dim: int,
        hidden_activation=torch.nn.functional.relu,
        output_activation=None,
        linear_wrapper=None,
    ):
        """Constructor."""
        super(MLP, self).__init__()
        self._linear_wrapper = linear_wrapper
        if len(hidden_sizes) == 0:
            self._add_linear_layer(input_dim, output_dim, 0)
            self.n_layers = 1
        else:
            self._add_linear_layer(input_dim, hidden_sizes[0], 0)
            for hidx in range(len(hidden_sizes) - 1):
                self._add_linear_layer(hidden_sizes[hidx],
                                       hidden_sizes[hidx+1], hidx + 1)
            self._add_linear_layer(hidden_sizes[-1], output_dim,
                                   len(hidden_sizes))
            self.n_layers = len(hidden_sizes) + 1
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def forward(
            self,
            net_in: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through network."""
        curr = net_in
        for layer_num in range(self.n_layers - 1):
            curr = getattr(self, 'linear_%d' % layer_num)(curr)
            curr = self.hidden_activation(curr)
        curr = getattr(self, 'linear_%d' % (self.n_layers - 1))(curr)
        if self.output_activation is not None:
            return self.output_activation(curr)
        return curr

    def _add_linear_layer(self, lin_in, lin_out, layer_num):
        layer = torch.nn.Linear(lin_in, lin_out)
        if self._linear_wrapper is not None:
            layer = self._linear_wrapper(layer)
        self.add_module('linear_%d' % layer_num, layer)


class GaussianNet(torch.nn.Module):
    """Network that returns mean and logvar of a Gaussian."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        encoder_hidden_sizes: Sequence[int],
        mean_hidden_sizes: Sequence[int],
        logvar_hidden_sizes: Sequence[int],
        latent_dim: int,
        hidden_activation=torch.nn.functional.relu,
        linear_wrappers = [None, None, None],
    ):
        """Constructor."""
        super(GaussianNet, self).__init__()
        self.encode_net = MLP(
            input_dim=input_dim,
            hidden_sizes=encoder_hidden_sizes,
            output_dim=latent_dim,
            hidden_activation=hidden_activation,
            output_activation=hidden_activation,
            linear_wrapper=linear_wrappers[0],
        )
        self.mean_net = MLP(
            input_dim=latent_dim,
            hidden_sizes=mean_hidden_sizes,
            output_dim=output_dim,
            hidden_activation=hidden_activation,
            linear_wrapper=linear_wrappers[1],
        )
        self.logvar_net = MLP(
            input_dim=latent_dim,
            hidden_sizes=logvar_hidden_sizes,
            output_dim=output_dim,
            hidden_activation=hidden_activation,
            linear_wrapper=linear_wrappers[2],
        )

    def forward(
            self,
            net_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through network. Returns (mean, logvar)"""
        latent = self.encode_net.forward(net_in)
        return self.mean_net.forward(latent), self.logvar_net.forward(latent)
