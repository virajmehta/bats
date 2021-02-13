"""
Neural Network for regression.

Author: Ian Char
Date: 11/20/2020
"""
import abc
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from modelling.models.base_model import BaseModel
from modelling.models.networks import MLP, GaussianNet
from modelling.utils.torch_utils import torch_to, reparameterize


class Policy(BaseModel, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_action(
            self,
            states: torch.Tensor,
            deterministic: bool = False,
    ):
        """Get action from policy."""


class DeterministicPolicy(Policy):
    """Neural Network for regression."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Sequence[int],
        hidden_activation=torch.nn.functional.relu,
        output_activation=torch.tanh,
        standardize_targets: bool = False,
    ):
        """Constructor.
        Args:
            input_dim: Dimension of input data.
            output_dim: Dimensions of the target data.
            hidden_sizes: Hidden size of the network.
            hidden_activation: Activation of the networks hidden layers.
            output_activation: Activation to put on the output layer.
            standardize_targets: Whether to standardize the targets to predict.
            linear_wrapper: Wrapper for linear layers such as regularizer.
        """
        super(DeterministicPolicy, self).__init__([input_dim, output_dim])
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._net = MLP(
            input_dim=input_dim,
            hidden_sizes=hidden_sizes,
            output_dim=output_dim,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
        )
        self._criterion = torch.nn.MSELoss()
        self._standardize_targets = standardize_targets

    def __call__(self, net_in, unstandardize_targets=True):
        """Get predictions from the network."""
        net_in = torch_to(net_in)
        net_in = self.standardize_batch([net_in])[0]
        net_out = self._net.forward(net_in)
        if self._standardize_targets and unstandardize_targets:
            net_out = self.unstandardize_batch([net_in, net_out])[1]
        return net_out

    def get_action(
            self,
            states: torch.Tensor,
            deterministic: bool = False,
    ):
        return self.__call__(net_in)

    def model_forward(self, batch: Sequence[torch.Tensor]) -> Dict[str, Any]:
        """Forward pass data through the model."""
        xi, yi = batch
        preds = self._net.forward(xi)
        if self._standardize_targets:
            targets = yi
        else:
            targets = self.unstandardize_batch(batch)[1]
        return OrderedDict(
            preds=preds,
            labels=targets,
        )

    def loss(
            self,
            forward_out: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute loss from the output of the network and targets.
        Returns the loss and additional stats.
        """
        preds = forward_out['preds']
        labels = forward_out['labels']
        diffs = (preds - labels) ** 2
        if 'weighting' in forward_out:
            diffs *= forward_out['weighting']
        loss = diffs.mean()
        stats = OrderedDict(
            ModelLoss=loss.item(),
        )
        stats['Loss'] = loss.item()
        return {'Model': loss}, stats


class StochasticPolicy(Policy):
    """Probabilistic Neural Network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        encoder_hidden_sizes: Sequence[int],
        latent_dim: int,
        mean_hidden_sizes: Sequence[int],
        logvar_hidden_sizes: Sequence[int],
        tanh_transform: bool = True,
        add_entropy_bonus: bool = False,
        train_alpha_entropy: bool = True,
        log_alpha_entropy: float = 0,
        target_entropy: Optional[float] = -3,
        hidden_activation=torch.nn.functional.relu,
        standardize_targets: bool = False,
    ):
        """Constructor.
        Args:
            input_dim: Dimension of input data.
            output_dim: Dimensions of the target data.
            encoder_hidden_sizes: Hidden size of the encoder network.
            latent_dim: The output of the encoder net and input of the
                mean and logvar networks.
            mean_hidden_sizes: Hidden size of the mean networks.
            logvar_hidden_sizes: Hidden size of the logvar networks.
            tanh_transform: Whether to do tanh transform on output actions.
            train_alpha_entropy: Whether to train alpha entropy.
            log_alpha_entropy: The coefficient on entropy term for loss.
            target_entropy: Target entropy used for tuning alpha. If none,
                then the target is -action_dim.
            hidden_activation: Activation of the networks hidden layers.
            standardize_targets: Whether to standardize the targets to predict.
        """
        super(StochasticPolicy, self).__init__([input_dim, output_dim])
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._gaussian_net = GaussianNet(
            input_dim=input_dim,
            output_dim=output_dim,
            encoder_hidden_sizes=encoder_hidden_sizes,
            mean_hidden_sizes=mean_hidden_sizes,
            logvar_hidden_sizes=logvar_hidden_sizes,
            latent_dim=latent_dim,
            hidden_activation=hidden_activation,
            tanh_transform=tanh_transform,
        )
        self._tanh_transform = tanh_transform
        self._standardize_targets = standardize_targets
        self._add_entropy_bonus = add_entropy_bonus
        self._train_alpha_entropy = train_alpha_entropy
        self.log_alpha = torch.nn.Parameter(
                torch.Tensor([log_alpha_entropy]),
                requires_grad=train_alpha_entropy,
        )
        if target_entropy is None:
            target_entropy = -1* output_dim
        self._target_entropy = target_entropy

    def __call__(self, net_in, unstandardize_targets=True):
        """Get the mean and logvar at specified condition points."""
        net_in = torch_to(net_in)
        net_in = self.standardize_batch([net_in])[0]
        mean, logvar = self._gaussian_net(net_in)
        if unstandardize_targets and self._standardize_targets:
            mu, sigma = self.standardizer.get_stats(1)
            mean = mean * sigma + mu
            logvar += 2 * torch.log(sigma)
        return mean, logvar

    def get_action(
            self,
            states: torch.Tensor,
            deterministic: bool = False,
    ):
        """Get action from policy."""
        mean, logvar = self.__call__(states)
        if deterministic:
            action = mean
        else:
            action = reparameterize(mean, logvar)
        if self._tanh_transform:
            return torch.tanh(action)
        return action

    def get_log_prob(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self._gaussian_net.get_log_prob(mean, logvar, actions)

    def sample_actions_and_logprobs(
            self,
            states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self.__call__(states)
        actions = self._gaussian_net.sample_from_mean_logvar(mean, logvar)
        logprobs = self._gaussian_net.get_log_prob(mean, logvar, actions)
        return actions, logprobs

    def model_forward(self, batch: Sequence[torch.Tensor]) -> Dict[str, Any]:
        """Forward pass data through the model."""
        xi, yi = batch
        mean, logvar = self._gaussian_net(xi)
        log_pi = self._gaussian_net.sample_logpis_from_mean_logvar(
                mean, logvar)
        if self._standardize_targets:
            targets = yi
        else:
            targets = self.standardizer.unstandardize(yi, 1)
        return OrderedDict(
            mean=mean,
            logvar=logvar,
            labels=targets,
            log_pi=log_pi,
        )

    def loss(
            self,
            forward_out: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute loss from the output of the network and targets.
        Returns the loss and additional stats.
        """
        mean = forward_out['mean']
        logvar = forward_out['logvar']
        labels = forward_out['labels']
        log_pi = forward_out['log_pi']
        # Get the policy loss.
        lls = self._gaussian_net.get_log_prob(mean, logvar, labels)
        if 'weighting' in forward_out:
             lls *= forward_out['weighting'].flatten()
        loss = -1 * lls
        if self._add_entropy_bonus:
            loss += self.log_alpha.exp().detach() * log_pi
        loss = loss.mean()
        stats = OrderedDict(
            Loss=loss.item(),
            Alpha=self.log_alpha.exp().detach().cpu().item(),
            PIWidths=(0.5 * logvar.detach().cpu()).exp().mean().item(),
            LogPis=log_pi.detach().cpu().mean().item(),
        )
        loss_dict = {'Model': loss}
        # Get alpha loss if training alpha.
        if self._train_alpha_entropy:
            alpha_loss = -(self.log_alpha.exp()
                           * (log_pi + self._target_entropy).detach()).mean()
            stats['AlphaLoss'] = alpha_loss.item()
            loss_dict['Alpha'] = alpha_loss
        # Compute the MSE for statistics.
        mean_est = mean.detach()
        if self._tanh_transform:
            mean_est = torch.tanh(mean_est)
        stats['MSE'] = ((mean_est - labels) ** 2).mean().item()
        if 'weighting' in forward_out:
            stats['Weighting'] = forward_out['weighting'].cpu().mean().item()
        return loss_dict, stats

    def get_parameter_sets(self) -> Dict[str, torch.Tensor]:
        """Get mapping from string description of parameters to parameters."""
        return_dict = {'Model': list(self._gaussian_net.parameters())}
        if self._train_alpha_entropy:
            return_dict['Alpha'] = [self.log_alpha]
        return return_dict
