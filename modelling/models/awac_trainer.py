"""
AWAC Trainer for policy.
"""
from collections import OrderedDict
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from modelling.utils.torch_utils import set_cuda_device, torch_to, unroll
from util import dict_append


class AWACTrainer(object):

    def __init__(
            self,
            policy,
            qnets,
            qtargets,
            gamma: float = 0.99,
            learning_rate: float = 1e-3,
            weight_decay: float = 0.0,
            soft_tau: float = 5e-3, # Weight for target net soft updates.
            cuda_device: str = '',
            save_path: Optional[str] = None,
            save_freq: int = -1,
            silent: bool = False,
            save_path_exists_ok: bool = True,
            track_stats: Sequence[str] = None,
            adv_weighting: float = 1, # How much to weight advantage by.
            weight_max: float = 20, # Maximum weighting a datapoint can have.
            env = None, # Gym environment for doing evaluations.
            max_ep_len: int = 1000,
            num_eval_eps: int = 10,
    ):
        # Put networks on correct device.
        set_cuda_device(cuda_device)
        self.policy = torch_to(policy)
        self.qnets = [torch_to(qnet) for qnet in qnets]
        self.qtargets = [torch_to(target) for target in qtargets]
        self.policy.train(False)
        for qnet in self.qnets:
            qnet.train(False)
        for qtarg in self.qtargets:
            qtarg.train(False)
        # Set up optimizer for policy.
        to_opt = {'policy': self.policy}
        to_opt.update({'qnet%d' % idx: qnet
                       for idx, qnet in enumerate(self.qnets)})
        self.optimizers = {}
        for netname, net in to_opt.items():
            psets = net.get_parameter_sets()
            if type(learning_rate) is not dict:
                ldict = {'%s_%s' % (netname, k): learning_rate
                                 for k in psets.keys()}
            else:
                ldict = learning_rate
            if type(weight_decay) is not dict:
                wdict = {'%s_%s' % (netname, k): weight_decay
                                 for k in psets.keys()}
            else:
                wdict = weight_decay
            self.optimizers.update({'%s_%s' % (netname, k): torch.optim.Adam(
                v,
                lr=ldict['%s_%s' % (netname, k)],
                weight_decay=wdict['%s_%s' % (netname, k)],
            ) for k, v in psets.items()})
        # Set up save directory and logging.
        self._save_path = save_path
        if save_path is not None:
            os.makedirs(save_path, exist_ok=save_path_exists_ok)
            os.makedirs(os.path.join(save_path, 'policy'),
                        exist_ok=save_path_exists_ok)
            for idx in range(len(qnets)):
                os.makedirs(os.path.join(save_path, 'qnet%d' % idx),
                            exist_ok=save_path_exists_ok)
            self._writer = SummaryWriter(
                log_dir=os.path.join(save_path, 'tensorboard'))
        else:
            self._writer = None
        self._save_freq = save_freq
        self._silent = silent
        self._pbar = None
        self._soft_tau = soft_tau
        self._gamma = gamma
        self._adv_weighting = adv_weighting
        self._weight_max = weight_max
        self._total_epochs = 0
        self._stats = OrderedDict()
        self._tr_stats = OrderedDict()
        if track_stats is None:
            track_stats = ['Returns', 'policy/Loss'] + ['qnet%d/Loss' % i
                                             for i in range(len(qnets))]
        self._track_stats = track_stats
        self._env = env
        self._max_ep_len = max_ep_len
        self._num_eval_eps = num_eval_eps

    def train(
            self,
            dataset: DataLoader,
            epochs: int,
            reset_models: bool = False,
    ) -> None:
        """Fit the model on a dataset. Returns train and validation losses.
        Args:
            epochs: Number of epochs to train for.
            dataset: Dataset to train on.
            reset_models: Whether to reset the models.
        """
        if reset_models:
            self.policy.reset()
            for qnet in self.qnets:
                qnet.reset()
            for qtarg in self.qtargets:
                qtarg.reset()
        if not self._silent:
            self._pbar = tqdm(total=epochs)
        for _ in range(epochs):
            self._set_train(True)
            for batch in dataset:
                self.batch_train(batch)
            self._set_train(False)
            self._evaluate_policy()
            self.end_epoch()
        # Save the networks off.
        spath = os.path.join(self._save_path, 'policy/model.pt')
        self.policy.save_model(spath)
        for idx, qnet in enumerate(self.qnets):
            spath = os.path.join(self._save_path, 'qnet%d/model.pt' % idx)
            qnet.save_model(spath)
        if not self._silent:
            self._pbar.close()
        self._pbar = None

    def batch_train(
            self,
            batch: Sequence[torch.Tensor],
    ) -> float:
        """Do a train step on batch. If validation batch, just log stats."""
        st, at = batch[0], batch[1]
        if len(batch) > 5:
            vt = batch[-1]
        else:
            vt = None
        losses, bstats = OrderedDict(), OrderedDict()
        # Compute the weightings to use on the batch.
        weighting = self.get_advantage_weighting(st, at, values=vt)
        # Get the policy loss.
        policy_out = self.policy.forward(batch[:2])
        policy_out['weighting'] = weighting
        policy_losses, policy_stats = self.policy.loss(policy_out)
        losses.update({'policy_%s' % k: v for k, v in policy_losses.items()})
        bstats.update({'policy/%s' % k: v for k, v in policy_stats.items()})
        # Get the Qnet losses.
        for qidx, qnet in enumerate(self.qnets):
            qtarg = self.qtargets[qidx]
            if vt is not None:
                targets = vt
                qnet_out['targets'] = vt
            else:
                with torch.no_grad():
                    qtvals = qtarg(batch[3], self.policy.get_action(batch[3]))
                targets = batch[2] + self._gamma * batch[4] * qtvals
            qnet_out = qnet.forward([batch[0], batch[1], targets])
            qnet_losses, qnet_stats = qnet.loss(qnet_out)
            losses.update({'qnet%d_%s' % (qidx, k): v
                           for k, v in qnet_losses.items()})
            bstats.update({'qnet%d/%s' % (qidx, k): v
                           for k, v in qnet_stats.items()})
        # Take gradient steps for the networks.
        for lossname, loss in losses.items():
            self.optimizers[lossname].zero_grad()
            loss.backward()
            self.optimizers[lossname].step()
        # Update the Qtargets.
        for qnet, qtarg in zip(self.qnets, self.qtargets):
            qtarg.soft_update(qnet, self._soft_tau)
        # Update statistics.
        for k, v in bstats.items():
            dict_append(self._tr_stats, k, v)
        return {k: loss.item() for k, loss in losses.items()}

    def get_advantage_weighting(self, st, at, values=None):
        # Compute Q values.
        with torch.no_grad():
            qvals = torch.min(torch.stack(
                [qnet(st, at) for qnet in self.qnets]), dim=0)[0]
        # Compute policy values.
        if values is None:
            pi_actions = self.policy.get_action(st)
            with torch.no_grad():
                values = torch.min(torch.stack(
                    [qnet(st, pi_actions) for qnet in self.qnets]), dim=0)[0]
        # Compute the advantage weighting.
        advantage = qvals - values
        weighting = torch.clamp((advantage / self._adv_weighting).exp(),
                                max=self._weight_max)
        return weighting

    def get_stats(self) -> Dict[str, Any]:
        """Get the statistics collected."""
        return self._stats

    def end_epoch(self) -> None:
        """End the epoch by logging information, saving if need be."""
        self._update_stats()
        self._total_epochs += 1
        # Write statistics to file.
        if self._save_path is not None:
            stat_path = os.path.join(self._save_path, 'stats.txt')
            if self._total_epochs == 1:
                with open(stat_path, 'w') as f:
                    f.write(','.join(['Epoch'] + list(self._stats.keys()))
                            + '\n')
            with open(stat_path, 'a') as f:
                f.write(','.join(
                    ['%d' % self._total_epochs]
                    + ['%f' % s[-1] for s in self._stats.values()]) + '\n')
            for k, v in self._stats.items():
                self._writer.add_scalar(k, v[-1], self._total_epochs)
        # Print statistics.
        if self._pbar is not None:
            print_stats = OrderedDict()
            for ts in self._track_stats:
                key = '%s/avg' % ts
                if key in self._stats:
                    print_stats[ts] = self._stats[key][-1]
            self._pbar.set_postfix(ordered_dict=print_stats)
            self._pbar.update(1)
        # Save model.
        if (self._save_path is not None
                and self._save_freq > 0
                and self._total_epochs % self._save_freq == 0):
            itr_save = os.path.join(self._save_path,
                                    'policy/itr_%d.pt' % self._total_epochs)
            self.policy.save_model(itr_save)
            for idx, qnet in enumerate(self.qnets):
                itr_save = os.path.join(self._save_path,
                                'qnet%d/itr_%d.pt' % (idx, self._total_epochs))
                qnet.save_model(itr_save)

    def _evaluate_policy(self):
        if self._env is not None:
            dict_append(self._tr_stats, 'Returns',
                        [unroll(self._env, self.policy, self._max_ep_len)
                         for _ in range(self._num_eval_eps)])

    def _update_stats(self) -> None:
        """Update the statistics for the epoch."""
        for k, v in self._tr_stats.items():
            dict_append(self._stats, '/'.join([k, 'avg']), np.mean(v))
            dict_append(self._stats, '/'.join([k, 'std']), np.std(v))
        self._tr_stats = OrderedDict()
        self._val_stats = OrderedDict()

    def _set_train(self, train):
        self.policy.train(train)
        for qnet in self.qnets:
            qnet.train(train)
