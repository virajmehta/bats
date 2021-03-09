"""
Trainer that does standard actor critic methods.
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


class ActorCriticTrainer(object):

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
            track_stats = ['Returns', 'policy/Loss'] + ['qnet%d/Value' % i
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
            pi_data: DataLoader = None,
            batch_updates_per_epoch: int = 1000,
            train_critic: bool = True,
            train_policy: bool = True,
    ) -> None:
        """Fit the model on a dataset. Returns train and validation losses.
        Args:
            epochs: Number of epochs to train for.
            dataset: Dataset to train on.
            reset_models: Whether to reset the models.
        """
        critic_rb = IteratedDataLoader(dataset)
        pi_rb = (IteratedDataLoader(dataset) if pi_data is None
                 else IteratedDataLoader(pi_data))
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
            for _ in range(batch_updates_per_epoch):
                if train_critic:
                    self.crtic_batch_train(critic_rb.next())
                if train_policy:
                    self.policy_batch_train(pi_rb.next())
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


    def crtic_batch_train(
            self,
            batch: Sequence[torch.Tensor],
    ) -> dict:
        """Do a train step on batch. If validation batch, just log stats."""
        st, at, rews, nxts, terms = [torch_to(b) for b in batch[:5]]
        if len(batch) > 5:
            vt = torch_to(batch[-1])
        else:
            vt = None
        losses, bstats = OrderedDict(), OrderedDict()
        # Form targets if no value is provided.
        if vt is None:
            pi_acts = self.policy.get_action(nxts)
            with torch.no_grad():
                qts = torch.min(torch.stack(
                    [qt(nxts, pi_acts) for qt in self.qtargets]), dim=0)[0]
            targets = rews + self._gamma * (1. - terms) * qts
        else:
            targets = vt
        # Get the Qnet losses.
        for qidx, qnet in enumerate(self.qnets):
            qnet_out = qnet.forward([st, at, targets])
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

    def policy_batch_train(
            self,
            batch: Sequence[torch.Tensor],
    ) -> dict:
        # Estimate policy value on batch.
        st = torch_to(batch[0])
        actions, log_pis = self.policy.sample_actions_and_logprobs(st)
        values = torch.min(torch.stack(
            [qn(st, actions) for qn in self.qnets]), dim=0)[0]
        # Policy model loss.
        model_loss = (-1 * values).mean()
        self.optimizers['policy_Model'].zero_grad()
        model_loss.backward()
        self.optimizers['policy_Model'].step()
        ret_dict = {'policy_Model': model_loss.item()}
        # Take alpha update steps.
        if self.policy._train_alpha_entropy:
            alpha_loss = self.policy.get_alpha_loss(log_pis)
            self.optimizers['policy_Alpha'].zero_grad()
            alpha_loss.backward()
            self.optimizers['policy_Alpha'].step()
            dict_append(self._tr_stats, 'policy/AlphaLoss', alpha_loss.item())
            ret_dict['policy_Alpha'] = alpha_loss.item()
        # Update statistics.
        dict_append(self._tr_stats, 'policy/Loss', model_loss.item())
        dict_append(self._tr_stats, 'policy/Alpha',
                self.policy.log_alpha.exp().detach().cpu().item())
        dict_append(self._tr_stats, 'policy/LogPis',
                log_pis.detach().cpu().mean().item())
        return ret_dict

    def set_save_path(self, save_path):
        self._save_path = save_path
        if save_path is not None:
            os.makedirs(save_path, exist_ok=self.save_path_exists_ok)
            os.makedirs(os.path.join(save_path, 'policy'),
                        exist_ok=self.save_path_exists_ok)
            for idx in range(len(self.qnets)):
                os.makedirs(os.path.join(save_path, 'qnet%d' % idx),
                            exist_ok=self.save_path_exists_ok)
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
