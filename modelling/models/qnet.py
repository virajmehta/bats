"""
QNetwork model.
"""
from collections import OrderedDict
from typing import Any, Dict, Sequence, Tuple

import torch

from modelling.models.base_model import BaseModel
from modelling.models.networks import MLP
from modelling.utils.torch_utils import torch_to


class QNet(BaseModel):
    """Neural Network for regression."""

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        hidden_activation=torch.nn.functional.relu,
        output_activation=None,
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
        super(QNet, self).__init__([state_dim, act_dim, 1])
        self.input_dim = state_dim + act_dim
        self.output_dim = 1
        self._net = MLP(
            input_dim=state_dim + act_dim,
            hidden_sizes=hidden_sizes,
            output_dim=1,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
        )
        self._criterion = torch.nn.MSELoss()
        self._standardize_targets = standardize_targets

    def __call__(self, state, action):
        """Get predictions from the network."""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        if len(action.shape) == 1:
            action = action.reshape(1, -1)
        return self._net.forward(torch_to(torch.cat([state, action], dim=1)))

    def model_forward(self, batch: Sequence[torch.Tensor]) -> Dict[str, Any]:
        """Forward pass data through the model."""
        st, at, targets = batch
        preds = self._net.forward(torch.cat([st, at], dim=1))
        od = OrderedDict(
            preds=preds,
            targets=targets,
        )
        return od

    def loss(
            self,
            forward_out: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute loss from the output of the network and targets.
        Returns the loss and additional stats.
        """
        preds = forward_out['preds']
        targets = forward_out['targets']
        error = (preds - targets) ** 2
        if 'weighting' in forward_out:
            error *= forward_out['weighting']
        loss = error.mean()
        stats = OrderedDict(
            Loss=loss.item(),
            TargetValue=targets.detach().cpu().mean().item(),
        )
        return {'Model': loss}, stats

    def soft_update(
            self,
            other_net,
            soft_tau: float = 5e-3,
    ) -> None:
        net_params = other_net._net.named_parameters()
        target_params = self._net.named_parameters()
        target_dict_params = dict(target_params)
        for name1, param1 in net_params:
            if name1 in target_dict_params:
                target_dict_params[name1].data.copy_(soft_tau * param1.data
                        + (1 - soft_tau) * target_dict_params[name1].data)
        self._net.load_state_dict(target_dict_params)
