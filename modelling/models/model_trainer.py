"""
Trainer for density estimation model.

Author: Ian Char
Date: 8/26/2020
"""
from collections import OrderedDict
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from modelling.models import BaseModel
from modelling.models.policies import Policy
from modelling.utils.torch_utils import set_cuda_device, torch_to, unroll
from util import dict_append


class ModelTrainer(object):
    """Trainer for deep density models."""

    def __init__(
            self,
            model: BaseModel,
            learning_rate: float = 1e-4,
            weight_decay: float = 0.0,
            cuda_device: str = '',
            save_path: Optional[str] = None,
            save_freq: int = -1,
            silent: bool = False,
            save_path_exists_ok: bool = True,
            save_best_model: bool = True,
            validation_tune_metric: str = 'Loss',
            optimizers=None,
            track_stats: Sequence[str] = [],
            env = None, # Gym environment for doing evaluations.
            max_ep_len: int = 1000,
            num_eval_eps: int = 10,
            train_loops_per_epoch=1, # Number of times to loop through dataset.
    ) -> None:
        """Constructor.
        Args:
            model: Density estimation model.
            learning_rate: Learning rate for the optimization.
            weight_decay: Amount of weight decayy to have in optimizer.
            cuda_device: The GPU device to use for the model. If not provided
                defaults to using CPU.
            save_path: Path to save information at if provided.
            save_freq: How often to save off the model. -1 does not save until
                end.
            silent: Whether to not print info to the terminal.
            save_path_exists_ok: Whether it is ok to overwrite on top of a
                pre-existing save path.
            save_best_model: If True, save the best model according to
                validation loss under model.pt
            validation_tune_metric: The metric to print and do overfitting
                detection on.
            optimizer: Sequence of custom optimizer, use Adam if None.
            track_stats: List of stats to track on progress bar.
        """
        set_cuda_device(cuda_device)
        self.model = torch_to(model)
        self.model.train(False)
        if optimizers is None:
            psets = self.model.get_parameter_sets()
            if type(learning_rate) is not dict:
                learning_rate = {k: learning_rate for k in psets.keys()}
            if type(weight_decay) is not dict:
                weight_decay = {k: weight_decay for k in psets.keys()}
            self.optimizers = {k: torch.optim.Adam(
                    v,
                    lr=learning_rate[k],
                    weight_decay=weight_decay[k],
            ) for k, v in psets.items()}
        else:
            self.optimizers = optimizers
        self._save_path = save_path
        if save_path is not None:
            os.makedirs(save_path, exist_ok=save_path_exists_ok)
            self._writer = SummaryWriter(
                log_dir=os.path.join(save_path, 'tensorboard'))
        else:
            self._writer = None
        self._save_freq = save_freq
        self._train_loops_per_epoch = train_loops_per_epoch
        self._silent = silent
        self._pbar = None
        self._save_best_model = save_best_model
        self._total_epochs = 0
        self._stats = OrderedDict()
        self._tr_stats = OrderedDict()
        self._val_stats = OrderedDict()
        self._best_tr_loss = float('inf')
        self._best_val_loss = float('inf')
        self._best_val_loss_epoch = 0
        self._validation_tune_metric = validation_tune_metric
        self._track_stats = track_stats
        self._env = env
        self._max_ep_len = max_ep_len
        self._num_eval_eps = num_eval_eps

    def fit(
            self,
            dataset: DataLoader,
            epochs: int,
            validation: Optional[DataLoader] = None,
            early_stop_wait_time: Optional[int] = None,
            dont_reset_model: bool = False,
            last_column_is_weights: bool = False,
    ) -> None:
        """Fit the model on a dataset. Returns train and validation losses.
        Args:
            dataset: Dataset to train on.
            epochs: Number of epochs to train for.
            validation: Validation dataset if we want validation scores.
            early_stop_wait_time: Providing this will enable overfitting
                detection if a validation set was provided. This is the
                number of epochs to wait since seeing best validation score
                before stopping early.
            dont_reset_model: Whether to reset the model or not.
        """
        if not dont_reset_model:
            self.model.reset()
        if not self._silent:
            self._pbar = tqdm(total=epochs)
        for _ in range(epochs):
            self.model.train(True)
            for _ in range(self._train_loops_per_epoch):
                for batch in dataset:
                    self.batch_train(
                            batch,
                            last_column_is_weights=last_column_is_weights,
                    )
            self.model.train(False)
            if validation is not None:
                for batch in validation:
                    self.batch_train(batch,
                                 validation=True,
                                 last_column_is_weights=last_column_is_weights)
            self._evaluate_policy()
            self.end_epoch()
            if early_stop_wait_time is not None and validation is not None:
                time_gap = self._total_epochs - self._best_val_loss_epoch
                if time_gap > early_stop_wait_time:
                    break
        if not self._silent:
            self._pbar.close()
        self._pbar = None
        if validation is None:
            self.model.save_model(os.path.join(self._save_path, 'model.pt'))

    def batch_train(
            self,
            batch: Sequence[torch.Tensor],
            validation: bool = False,
            last_column_is_weights: bool = False,
    ) -> float:
        """Do a train step on batch. If validation batch, just log stats."""
        forward_in = batch[:-1] if last_column_is_weights else batch
        if validation:
            with torch.no_grad():
                model_out = self.model.forward(forward_in)
        else:
            model_out = self.model.forward(forward_in)
        if last_column_is_weights:
            model_out['weights'] = batch[-1]
        losses, bstats = self.model.loss(model_out)
        stat_dict = self._val_stats if validation else self._tr_stats
        for k, v in bstats.items():
            dict_append(stat_dict, k, v)
        if not validation:
            for loss_name, loss in losses.items():
                self.optimizers[loss_name].zero_grad()
                loss.backward()
                self.optimizers[loss_name].step()
        return {k: loss.item() for k, loss in losses.items()}

    def get_stats(self) -> Dict[str, Any]:
        """Get the statistics collected."""
        return self._stats

    def end_epoch(self) -> None:
        """End the epoch by logging information, saving if need be."""
        self._update_stats()
        self._total_epochs += 1
        val_header = '%s/avg/val' % self._validation_tune_metric
        # Update the best losses seen.
        if 'Loss/avg/train' in self._stats:
            self._best_tr_loss = min(self._best_tr_loss,
                                     self._stats['Loss/avg/train'][-1])
        if val_header in self._stats:
            val_loss = self._stats[val_header][-1]
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._best_val_loss_epoch = self._total_epochs
                if self._save_best_model and self._save_path is not None:
                    self.model.save_model(os.path.join(self._save_path,
                                                       'model.pt'))
        # Write statistics to file.
        if self._save_path is not None:
            stat_path = os.path.join(self._save_path, 'stats.txt')
            if self._total_epochs == 1:
                with open(stat_path, 'a') as f:
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
            if 'Loss/avg/train' in self._stats:
                print_stats['TrainLoss'] = self._stats['Loss/avg/train'][-1]
                print_stats['BestTrainLoss'] = self._best_tr_loss
            if val_header in self._stats:
                print_stats['ValLoss'] = self._stats[val_header][-1]
                print_stats['BestValLoss'] = self._best_val_loss
            if 'Returns/avg' in self._stats:
                print_stats['ReturnsAvg'] = self._stats['Returns/avg'][-1]
                print_stats['ReturnsStd'] = self._stats['Returns/std'][-1]
            for ts in self._track_stats:
                key = '%s/avg/train' % ts
                if key in self._stats:
                    print_stats[ts] = self._stats[key][-1]
            self._pbar.set_postfix(ordered_dict=print_stats)
            self._pbar.update(1)
        # Save model.
        if (self._save_path is not None
                and self._save_freq > 0
                and self._total_epochs % self._save_freq == 0):
            itr_save = os.path.join(self._save_path,
                                    'itr_%d.pt' % self._total_epochs)
            self.model.save_model(itr_save)

    def _evaluate_policy(self) -> None:
        if issubclass(type(self.model), Policy) and self._env is not None:
            unrolls = [unroll(self._env, self.model, self._max_ep_len)
                       for _ in range(self._num_eval_eps)]
            dict_append(self._stats, 'Returns/avg', np.mean(unrolls))
            dict_append(self._stats, 'Returns/std', np.std(unrolls))

    def _update_stats(self) -> None:
        """Update the statistics for the epoch."""
        for k, v in self._tr_stats.items():
            dict_append(self._stats, '/'.join([k, 'avg', 'train']), np.mean(v))
            dict_append(self._stats, '/'.join([k, 'std', 'train']), np.std(v))
        for k, v in self._val_stats.items():
            dict_append(self._stats, '/'.join([k, 'avg', 'val']), np.mean(v))
            dict_append(self._stats, '/'.join([k, 'std', 'val']), np.std(v))
        self._tr_stats = OrderedDict()
        self._val_stats = OrderedDict()
