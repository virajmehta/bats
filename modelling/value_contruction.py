"""
Value function construction.
"""
import os

import numpy as np
import torch
from torch.nn.functional import tanh
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from modelling.policy_construction import get_qlearning_dataloader
from modelling.models import ValueFunction
from modelling.trainers import ValueFunctionTrainer
from util import s2i


def train_value_function(
    dataset,
    epochs,
    save_path,
    hidden_sizes='64,64',
    gamma=0.99,
    batch_size=256,
    batch_updates_per_epoch=100,
    learning_rate=1e-3,
    weight_decay=0,
    cuda_device='',
    save_freq=-1,
    silent=False,
):
    use_gpu = cuda_device != ''
    obs_dim = dataset['observations'].shape[1]
    dataloader = get_qlearning_dataloader(dataset, cuda_device, batch_size,
                                          shuffle=True)
    vf = get_value_function(obs_dim, hidden_sizes)
    vf_target = get_value_function(obs_dim, hidden_sizes)
    trainer = ValueFunctionTrainer(
        vf=vf,
        vf_target=vf_target,
        gamma=gamma,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        cuda_device=cuda_device,
        save_freq=save_freq,
        silent=silent,
        save_path=save_path,
    )
    trainer.train(dataloader, epochs,
                  batch_updates_per_epoch=batch_updates_per_epoch)
    return vf


def load_value_function(
    load_dir,
    obs_dim,
    hidden_sizes='64,64',
    cuda_device='',
):
    vf = get_value_function(obs_dim, hidden_sizes)
    device = 'cpu' if cuda_device == '' else 'cuda:' + cuda_device
    vf.load_model(os.path.join(load_dir, 'model.pt'), map_location=device)
    return vf


def get_value_function(
    obs_dim,
    hidden_sizes='64,64',
):
    return ValueFunction(
        state_dim=obs_dim,
        hidden_sizes=s2i(hidden_sizes),
    )
