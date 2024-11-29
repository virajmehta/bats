"""
Trainer that trains a value function.
"""
from collections import OrderedDict
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from modelling.utils.torch_utils import set_cuda_device, torch_to, unroll,\
        IteratedDataLoader
from util import dict_append


class ValueFunctionTrainer(object):

    def __init__(
        self,
        vf,
        vf_target,
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
    ):
        # Put networks on correct device.
        set_cuda_device(cuda_device)
        self.vf = torch_to(vf)
        self.vf_target = torch_to(vf_target)
        self.vf.train(False)
        self.vf_target.train(False)
        to_opt = {'vf': self.vf}
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
        self.save_path_exists_ok = save_path_exists_ok
        self.set_save_path(save_path)
        self._save_freq = save_freq
        self._silent = silent
        self._pbar = None
        self._soft_tau = soft_tau
        self._gamma = gamma
        self._total_epochs = 0
        self._stats = OrderedDict()
        self._tr_stats = OrderedDict()
        if track_stats is None:
            track_stats = ['vf/Value', 'vf/Loss']
        self._track_stats = track_stats

    def train(
            self,
            dataset: DataLoader,
            epochs: int,
            pi_data: DataLoader = None,
            batch_updates_per_epoch: int = 1000,
    ) -> None:
        """Fit the model on a dataset. Returns train and validation losses.
        Args:
            epochs: Number of epochs to train for.
            dataset: Dataset to train on.
            reset_models: Whether to reset the models.
        """
        replay = IteratedDataLoader(dataset)
        if not self._silent:
            self._pbar = tqdm(total=epochs)
        for _ in range(epochs):
            self._set_train(True)
            for _ in range(batch_updates_per_epoch):
                self.crtic_batch_train(replay.next())
            self._set_train(False)
            self.end_epoch()
        # Save the networks off.
        self.vf.save_model(os.path.join(self._save_path, 'model.pt'))
        if not self._silent:
            self._pbar.close()
        self._pbar = None


    def crtic_batch_train(
            self,
            batch: Sequence[torch.Tensor],
    ) -> dict:
        """Do a train step on batch. If validation batch, just log stats."""
        st, at, rews, nxts, terms = [torch_to(b) for b in batch[:5]]
        losses, bstats = OrderedDict(), OrderedDict()
        # Form targets if no value is provided.
        with torch.no_grad():
            nxt_vals= self.vf_target(nxts)
        targets = rews + self._gamma * (1. - terms) * nxt_vals
        # Get the Qnet losses.
        vf_out = self.vf.forward([st, targets])
        vf_losses, vf_stats = self.vf.loss(vf_out)
        losses.update({'vf_%s' % k: v for k, v in vf_losses.items()})
        bstats.update({'vf/%s' % k: v for k, v in vf_stats.items()})
        # Take gradient steps for the networks.
        for lossname, loss in losses.items():
            self.optimizers[lossname].zero_grad()
            loss.backward()
            self.optimizers[lossname].step()
        # Update the value target function.
        self.vf_target.soft_update(self.vf, self._soft_tau)
        # Update statistics.
        for k, v in bstats.items():
            dict_append(self._tr_stats, k, v)
        return {k: loss.item() for k, loss in losses.items()}

    def set_save_path(self, save_path):
        self._save_path = save_path
        if save_path is not None:
            os.makedirs(save_path, exist_ok=self.save_path_exists_ok)
            self._writer = SummaryWriter(
                log_dir=os.path.join(save_path, 'tensorboard'))
        else:
            self._writer = None

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
            self.vf.save_model(os.path.join(self._save_path,
                'itr_%d.pt' % self._total_epochs))

    def _update_stats(self) -> None:
        """Update the statistics for the epoch."""
        for k, v in self._tr_stats.items():
            dict_append(self._stats, '/'.join([k, 'avg']), np.mean(v))
            dict_append(self._stats, '/'.join([k, 'std']), np.std(v))
        self._tr_stats = OrderedDict()
        self._val_stats = OrderedDict()

    def _set_train(self, train):
        self.vf.train(train)
