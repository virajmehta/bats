"""
Model for a value function.
"""
from collections import OrderedDict
from typing import Any, Dict, Sequence, Tuple

import torch

from modelling.models.base_model import BaseModel
from modelling.models.networks import MLP
from modelling.utils.torch_utils import torch_to, torch_zeros, torch_ones


class ValueFunction(BaseModel):
    """Neural Network for regression."""

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Sequence[int],
        hidden_activation=torch.nn.functional.relu,
        output_activation=None,
        standardize_targets: bool = False,
        discount: float = 0.99,
        lmbda: float = 0.95,
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
        super(ValueFunction, self).__init__([state_dim, state_dim, 1, 1, 1])
        self.input_dim = state_dim
        self.output_dim = 1
        self.discount = discount
        self.lmbda = lmbda
        self._net = MLP(
            input_dim=state_dim,
            hidden_sizes=hidden_sizes,
            output_dim=1,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
        )
        self._criterion = torch.nn.MSELoss()
        self._standardize_targets = standardize_targets
        self._supervision_train_mode = False

    def __call__(self, state):
        """Get predictions from the network."""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        return self._net.forward(torch_to(state))

    def model_forward(self, batch: Sequence[torch.Tensor]) -> Dict[str, Any]:
        """Forward pass data through the model."""
        obs, nxts, rewards, values, nsteps, terminals = batch
        batch_size = len(obs)
        max_steps = nxts.shape[1]
        preds = self._net.forward(obs)
        if self._supervision_train_mode:
            targets = values
        else:
            # Form TD-Lambda targets.
            with torch.no_grad():
                nxt_vals = self._net.forward(nxts.reshape(-1, self.input_dim))
            nxt_vals = nxt_vals.reshape(-1, max_steps)
            # Put into matrices with 0 entries past information.
            target_vals = torch_zeros((batch_size, max_steps))
            rewmat = torch_zeros((batch_size, max_steps))
            coefs = torch_zeros((batch_size, max_steps))
            for validx, valrow in enumerate(nxt_vals):
                end = int(nsteps[validx].item())
                target_vals[validx, :end] = valrow[:end]
                rewmat[validx, :end] = rewards[validx, :end]
                if terminals[validx].item():
                    target_vals[validx, end - 1] = 0
                if end > 1:
                    coefs[validx, :end] = torch.pow(
                            torch_ones(end) * self.lmbda,
                            torch_to(torch.arange(end)),
                    )
                    coefs[validx, :end - 1] *= (1 - self.lmbda)
                else:
                    coefs[validx, 0] = 1
            targets = torch_zeros(batch_size)
            gammas = torch.pow(torch_ones(max_steps) * self.discount,
                               torch_to(torch.arange(max_steps)))
            for st in range(1, max_steps):
                targets += (coefs[:, st - 1]
                            * (torch.sum(rewmat[:, :st] * gammas[:st], dim=1)
                               + (gammas[st] * target_vals[:, st-1])))
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
            Value=targets.detach().cpu().mean().item(),
        )
        return {'Model': loss}, stats

    def set_supervision_train_mode(self, mode):
        self._supervision_train_mode = mode
