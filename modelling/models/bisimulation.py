"""
Model for bisimulation.
"""
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from modelling.models import BaseModel, PNN
from modelling.models.networks import MLP
from modelling.utils.torch_utils import torch_to, reparameterize


class BisimulationModel(BaseModel):
    """Bisimulation model based on: https://arxiv.org/abs/2006.10742"""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        latent_dim: int,
        num_dyn_nets: int,
        encoder_architecture: Sequence[int],
        dyn_architecture: Sequence[int],
        dyn_latent_dim: int,
        mean_head_architecture: Sequence[int],
        logvar_head_architecture: Sequence[int],
        logvar_bounds: Optional[Tuple[float, float]] = None,
        bound_loss_coef: float = 1e-3,
        bound_loss_function = None,
        discount: float = 0.99,
        encoder_hidden_activation=torch.nn.functional.relu,
        pnn_hidden_activation=torch.nn.functional.relu,
    ):
        """Constructor.
        Args:
            obs_dim: The observation dimension.
            act_dim: The action dimension.
            latent_dim: The dimension of the latent space to learn.
            encoder_architecture: Architecture of encoder net as MLP.
            dyn_architecture: Architecture of the base dynamics model.
            dyn_latent_dim: The output dimension of the dynamics base net.
            mean_head_architecture: The architecture of the dynamics mean
                head network.
            logvar_head_architecture: The dyn_architecture of the logvar
                head network.
            hidden_activation: Activation of the networks hidden layers.
            logvar_bounds: Bounds on the logvariance.
            bound_loss_coef: Coefficient on bound in the loss.
            bound_loss_function: Loss function for the logvar bounds.
        """
        # Input to forward should be observation, actions, rewards, next obs.
        super(BisimulationModel, self).__init__([obs_dim, act_dim, 1, obs_dim])
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.num_dyn_nets = num_dyn_nets
        self.discount = discount
        self._retain_graph = True
        self.encoder = MLP(
            input_dim=obs_dim,
            hidden_sizes=encoder_architecture,
            output_dim=latent_dim,
            hidden_activation=encoder_hidden_activation,
        )
        for i in range(num_dyn_nets):
            setattr(self, 'pnn_%d' % i, PNN(
                input_dim=latent_dim + act_dim,
                output_dim=latent_dim + 1,
                encoder_hidden_sizes=dyn_architecture,
                mean_hidden_sizes=mean_head_architecture,
                logvar_hidden_sizes=logvar_head_architecture,
                latent_dim=dyn_latent_dim,
                hidden_activation=pnn_hidden_activation,
                logvar_bounds=logvar_bounds,
                bound_loss_coef=bound_loss_coef,
                standardize_targets=False,
                bound_loss_function=bound_loss_function,
            ))

    def model_forward(self, batch: Sequence[torch.Tensor]) -> Dict[str, Any]:
        """Forward pass data through the model."""
        oi, ai, ri, ni = batch
        # Encode the states.
        zi = self.encoder(oi)
        with torch.no_grad():
            nzi = self.encoder(ni)
        # Get next latent transitions.
        nxt_mean, nxt_logvar = self._apply_dynamics(
                torch.cat([zi.detach(), ai], dim=1))
        return OrderedDict(
            oi=oi,
            ai=ai,
            ri=ri,
            ni=ni,
            zi=zi,
            nzi=nzi,
            nxt_mean=nxt_mean,
            nxt_logvar=nxt_logvar,
        )

    def loss(
            self,
            forward_out: Dict[str, Any],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Compute loss from the output of the network and targets.
        Returns the loss and additional stats.
        """
        losses, stats = [OrderedDict() for _ in range(2)]
        for i in range(self.num_dyn_nets):
            pnn = getattr(self, 'pnn_%d' % i)
            to_pnn = OrderedDict(
                mean=forward_out['nxt_mean'][i],
                logvar=forward_out['nxt_logvar'][i],
                labels=torch.cat([forward_out['ri'], forward_out['nzi']],
                                 dim=1)
            )
            dynloss, dynstats = pnn.loss(to_pnn)
            for v in dynloss.values():
                losses['PNN%d' % i] = v
            for k, v in dynstats.items():
                stats['PNN%d_%s' % (i, k)] = v
        encoder_loss, encoder_stats = self.get_encoder_loss(forward_out)
        losses.update(encoder_loss)
        stats.update(encoder_stats)
        return losses, stats


    def get_encoder_loss(
            self,
            forward_out: Dict[str, Any],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        stats = OrderedDict()
        stacked_means = torch.stack(
                [nm.detach() for nm in forward_out['nxt_mean']])
        stacked_logvars = torch.stack(
                [nv.detach() for nv in forward_out['nxt_logvar']])
        # Get a permuation.
        permutation = torch.randperm(len(forward_out['oi']))
        # Reward difference.
        rewards = torch.mean(stacked_means[:, :, 0].detach(), dim=0)
        rew_diff = torch.abs(rewards - rewards[permutation])
        stats['RewardDifference'] = torch.mean(rew_diff).item()
        # W2 distance.
        means = torch.mean(stacked_means[:, :, 1:].detach(), dim=0)
        stds = torch.mean(
                torch.exp(0.5 * stacked_logvars[:, :, 1:].detach()),
                dim=0,
        )
        mean_diffs = torch.sum((means - means[permutation]) ** 2, dim=1)
        std_diffs = torch.sum((stds - stds[permutation]) ** 2, dim=1)
        w2 = mean_diffs + std_diffs
        stats['Wasserstein'] = torch.mean(w2).item()
        # Get latent distance loss.
        zi = forward_out['zi']
        latent_dist = torch.norm(zi - zi[permutation], p=1)
        target_dist = rew_diff + self.discount * w2
        loss = torch.mean((latent_dist - target_dist) ** 2)
        stats['EncoderLoss'] = loss.item()
        return {'Encoder': loss}, stats


    def get_encoding(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        observations = torch_to(observations)
        with torch.no_grad():
            return self.encoder(observations)

    def get_mean_logvar(
            self,
            conditions: torch.Tensor, # [latentobs, actions]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the mean and logvar at specified condition points."""
        conditions = torch_to(conditions)
        with torch.no_grad():
            mean, logvar = self._apply_dynamics(conditions)
        return torch.stack(mean), torch.stack(logvar)

    def sample(
            self,
            conditions: torch.Tensor, # [latentobs, actions]
    ) -> torch.Tensor:
        """Sample from the model."""
        means, logvars = self.get_mean_logvar(conditions)
        return [reparameterize(mean, logvar)
                for mean, logvar in zip(means, logvars)]

    def _apply_dynamics(
            self,
            conditions: torch.Tensor,
    ) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
        """Apply the Gaussian net and possible bound the logvar."""
        mean_outs, logvar_outs = [], []
        for i in range(self.num_dyn_nets):
            pnn = getattr(self, 'pnn_%d' % i)
            pnn_mean, pnn_logvar = pnn._apply_gnet(conditions)
            mean_outs.append(pnn_mean)
            logvar_outs.append(pnn_logvar)
        return mean_outs, logvar_outs

    def get_parameter_sets(self) -> Dict[str, torch.Tensor]:
        """Get mapping from string description of parameters to parameters."""
        psets = OrderedDict(Encoder=list(self.encoder.parameters()))
        for i in range(self.num_dyn_nets):
            psets['PNN%d' % i] = list(getattr(self, 'pnn_%d').parameters())
        return psets
