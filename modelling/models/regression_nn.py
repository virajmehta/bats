"""
Neural Network for regression.

Author: Ian Char
Date: 11/20/2020
"""
from collections import OrderedDict
from typing import Any, Dict, Sequence, Tuple

import torch

from modelling.models.base_model import BaseModel
from modelling.models.networks import MLP
from modelling.utils.torch_utils import torch_to


class RegressionNN(BaseModel):
    """Neural Network for regression."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Sequence[int],
        hidden_activation=torch.nn.functional.relu,
        output_activation=None,
        standardize_targets: bool = True,
        linear_wrapper = None,
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
        super(RegressionNN, self).__init__([input_dim, output_dim])
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
        loss = self._criterion(preds, labels)
        stats = OrderedDict(
            ModelLoss=loss.item(),
        )
        stats['Loss'] = loss.item()
        return loss, stats
