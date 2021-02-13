"""
Probabilistic Neural Network.

Author: Ian Char
Date: 11/13/2020
"""
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from modelling.models import BaseModel
from modelling.models.networks import GaussianNet
from modelling.utils.torch_utils import torch_to, reparameterize


class PNN(BaseModel):
    """Probabilistic Neural Network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        encoder_hidden_sizes: Sequence[int],
        latent_dim: int,
        mean_hidden_sizes: Sequence[int],
        logvar_hidden_sizes: Sequence[int],
        hidden_activation=torch.nn.functional.relu,
        logvar_bounds: Optional[Tuple[float, float]] = None,
        bound_loss_coef: float = 1e-3,
        standardize_targets: bool = True,
        linear_wrappers = [None, None, None],
        bound_loss_function = None,
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
            hidden_activation: Activation of the networks hidden layers.
            logvar_bounds: Bounds on the logvariance.
            bound_loss_coef: Coefficient on bound in the loss.
            standardize_targets: Whether to standardize the targets to predict.
            linear_wrappers: Wrapper for linear layers such as regularizers,
                order is [encoder, mean, logvar].
            bound_loss_function: Loss function for the logvar bounds.
        """
        super(PNN, self).__init__([input_dim, output_dim])
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
            linear_wrappers=linear_wrappers,
        )
        self._bound_loss_coef = bound_loss_coef
        self._standardize_targets = standardize_targets
        if bound_loss_function is None:
            self._bound_loss_function = torch.nn.L1Loss(reduction='sum')
        else:
            self._bound_loss_function = bound_loss_function
        if logvar_bounds is not None:
            self._var_pinning = True
            self._min_logvar = torch.nn.Parameter(torch_to(logvar_bounds[0]
                    * torch.ones(1, output_dim, dtype=torch.float32,
                                 requires_grad=True)))
            self._max_logvar = torch.nn.Parameter(torch_to(logvar_bounds[1]
                    * torch.ones(1, output_dim, dtype=torch.float32,
                                 requires_grad=True)))
        else:
            self._var_pinning = False
            self._min_logvar = None
            self._max_logvar = None

    def model_forward(self, batch: Sequence[torch.Tensor]) -> Dict[str, Any]:
        """Forward pass data through the model."""
        xi, yi = batch
        mean, logvar = self._apply_gnet(xi)
        if self._standardize_targets:
            targets = yi
        else:
            targets = self.standardizer.unstandardize(yi, 1)
        return OrderedDict(
            mean=mean,
            logvar=logvar,
            labels=targets,
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
        sq_diffs = (mean - labels) ** 2
        mse = torch.mean(sq_diffs)
        loss = torch.mean(torch.exp(-logvar) * sq_diffs + logvar)
        stats = OrderedDict(
            ModelLoss=loss.item(),
            MSE=mse.item(),
        )
        if self._var_pinning:
            bound_loss = self._bound_loss_coef * self._bound_loss_function(
                    self._max_logvar, self._min_logvar)
            stats['BoundLoss'] = bound_loss.item()
            loss += bound_loss
        stats['Loss'] = loss.item()
        return {'Model': loss}, stats

    def get_mean_logvar(
            self,
            conditions: torch.Tensor,
            unstandardize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the mean and logvar at specified condition points."""
        conditions = torch_to(conditions)
        conditions = self.standardize_batch([conditions])[0]
        with torch.no_grad():
            mean, logvar = self._apply_gnet(conditions)
        if unstandardize and self._standardize_targets:
            mu, sigma = self.standardizer.get_stats(1)
            mean = mean * sigma + mu
            logvar += 2 * torch.log(sigma)
        return mean, logvar

    def sample(
            self,
            conditions: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from the model."""
        mean, logvar = self.get_mean_logvar(conditions)
        return reparameterize(mean, logvar)

    def _apply_gnet(
            self,
            conditions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the Gaussian net and possible bound the logvar."""
        mean, logvar = self._gaussian_net.forward(conditions)
        if self._var_pinning:
            logvar = self._max_logvar - torch.nn.functional.softplus(
                    self._max_logvar - logvar)
            logvar = self._min_logvar + torch.nn.functional.softplus(
                    logvar - self._min_logvar)
        return mean, logvar
