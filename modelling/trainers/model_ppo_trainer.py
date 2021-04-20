"""
AWAC Trainer for policy.
"""
from collections import OrderedDict
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from modelling.utils.torch_utils import set_cuda_device, torch_to, unroll,\
        IteratedDataLoader, torchify_to
from util import dict_append


class ModelPPOTrainer(object):

    def __init__(
            self,
            policy,
            vf,
            model_unroller,
            gamma: float = 0.99,
            lmbda: float = 0.95,
            epsilon: float = 0.2,
            learning_rate: float = 1e-3,
            weight_decay: float = 0.0,
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
        set_cuda_device(cuda_device)
        # Put networks on correct device.
        self.policy = torch_to(policy)
        self.policy.train(False)
        self.vf = torch_to(vf)
        self.vf.train(False)
        model_unroller.model = [torch_to(m) for m in model_unroller.model]
        for m in model_unroller.model:
            m.train(False)
        self.model_unroller = model_unroller
        # Set up optimizer for policy.
        to_opt = {'Model': self.policy}
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
            self.optimizers.update({'%s' % netname: torch.optim.Adam(
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
            self._writer = SummaryWriter(
                log_dir=os.path.join(save_path, 'tensorboard'))
        else:
            self._writer = None
        self._save_freq = save_freq
        self._silent = silent
        self._pbar = None
        self._gamma = gamma
        self._lmbda = lmbda
        self._epsilon = epsilon
        self._total_epochs = 0
        self._stats = OrderedDict()
        self._tr_stats = OrderedDict()
        if track_stats is None:
            track_stats = ['Returns', 'Loss', 'Ratio', 'Advantage']
        self._track_stats = track_stats
        self._env = env
        self._max_ep_len = max_ep_len
        self._num_eval_eps = num_eval_eps

    def train(
            self,
            dataset: DataLoader,
            inner_loops: int,
            horizon: int,
            start_unroll_batches_per_epoch: int,
            num_policy_batch_update_per_epoch: int,
            ppo_update_batch_size: int,
            outer_loops: int = 1,
    ) -> None:
        """Args:
            dataset: Is a dataloader with start states to initialize at.
        """
        start_replay = IteratedDataLoader(dataset)
        # For each epoch...
        for ep in range(outer_loops):
            # Collect data from the model environment.
            if not self._silent:
                print('Collecting imaginary data...')
            ppo_data = self.collect_ppo_data(
                start_replay=start_replay,
                horizon=horizon,
                start_unroll_batches_per_epoch=start_unroll_batches_per_epoch,
                ppo_update_batch_size=ppo_update_batch_size,
            )
            # Train the policy for a number of epochs.
            if not self._silent:
                print('Performing policy updates...')
            self.ppo_updates(
                    ppo_data,
                    inner_loops,
                    num_policy_batch_update_per_epoch,
            )
        # Save off the updated policy
        spath = os.path.join(self._save_path, 'model.pt')
        self.policy.save_model(spath)
        self._pbar = None

    def collect_ppo_data(
            self,
            start_replay: IteratedDataLoader,
            horizon: int,
            start_unroll_batches_per_epoch: int,
            ppo_update_batch_size: int,
    ) -> DataLoader:
        observations, actions, advantages, logpis =\
                [[] for _ in range(4)]
        for _ in tqdm(range(start_unroll_batches_per_epoch)):
            # Collect trajectories from the model.
            obs, acts, rews, terminals, infos =\
                    self.model_unroller.model_unroll(
                            start_states=start_replay.next()[0],
                            policy=self.policy,
                            horizon=horizon,
                    )
            # Calculate the advantages.
            with torch.no_grad():
                vals = self.vf(torchify_to(obs.reshape(-1, obs.shape[-1])))
            vals = vals.cpu().numpy().reshape(obs.shape[:-1])
            is_past = np.cumsum(terminals, axis=1) > 1
            deltas = (rews * (1 - is_past)
                      + self._gamma * vals[:, 1:] * (1 - terminals)
                      - vals[:, :-1])
            # Add the information
            for ridx, rollout in enumerate(obs):
                curradv = 0
                for hidx in range(horizon - 1, -1, -1):
                    if is_past[ridx, hidx]:
                        continue
                    curradv = (deltas[ridx, hidx]
                               + self._gamma * self._lmbda * curradv)
                    observations.append(obs[ridx, hidx])
                    actions.append(acts[ridx, hidx])
                    advantages.append(curradv)
                    logpis.append(infos['logpis'][ridx, hidx])
        return IteratedDataLoader(DataLoader(
            TensorDataset(
                torch.Tensor(observations),
                torch.Tensor(actions),
                torch.Tensor(advantages),
                torch.Tensor(logpis),
            ),
            batch_size=ppo_update_batch_size,
            shuffle=True,
        ))

    def ppo_updates(
        self,
        dataset: IteratedDataLoader,
        epochs: int,
        num_policy_batch_update_per_epoch: int,
    ):
        if not self._silent:
            self._pbar = tqdm(total=epochs)
        for ep in range(epochs):
            self._set_train(True)
            for _ in range(num_policy_batch_update_per_epoch):
                batch = dataset.next()
                model_out = self.policy.forward(batch[:2])
                model_out['advantage'] = torch_to(batch[2])
                model_out['oldlogpi'] = torch_to(batch[3])
                losses, bstats = self.policy.ppo_loss(model_out,
                                                      epsilon=self._epsilon)
                for k, v in bstats.items():
                    dict_append(self._tr_stats, k, v)
                for loss_name, loss in losses.items():
                    self.optimizers[loss_name].zero_grad()
                    loss.backward()
                    self.optimizers[loss_name].step()
            self._set_train(False)
            # Create statistics.
            self._evaluate_policy()
            self.end_epoch()
        if not self._silent:
            self._pbar.close()

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
