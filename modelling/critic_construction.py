"""
Learn a critic from the graph.
"""
from collections import OrderedDict
import os
import pickle as pkl

import numpy as np
import torch
from torch.nn.functional import tanh
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from modelling.models import Critic
from modelling.trainers import ModelTrainer
from util import s2i


def tdlambda_critic(
    dataset,
    save_dir,
    epochs,
    hidden_sizes='256,256',
    batch_size=256,
    batch_updates_per_epoch=50, # If None then epoch is going through dataset.
    learning_rate=1e-3,
    weight_decay=0,
    cuda_device='',
    save_freq=-1,
    silent=False,
):
    use_gpu = cuda_device != ''
    # Get data into trainable form.
    obs_dim = dataset['observations'].shape[1]
    vf = get_value_function(obs_dim, hidden_sizes)
    trainer = ModelTrainer(
            model=vf,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            cuda_device=cuda_device,
            save_freq=save_freq,
            save_path=save_dir,
    )
    with open(os.path.join(save_dir, 'params.pkl'), 'wb') as f:
        pkl.dump(OrderedDict(hidden_sizes=hidden_sizes, obs_dim=obs_dim)
    tr_data = DataLoader(
        TensorDataset(**[torch.Tensor(dataset[k] for k in ['observations',
            'next_observations', 'rewards', 'nsteps', 'terminals'])),
        batch_size=batch_size,
        shuffle=True,
    )
    trainer.fit(tr_data, epochs)
    return net

def supervise_critic(
    # A dictionary containing 'observations', 'actions', 'values', or 'advantage
    dataset,
    save_dir,
    epochs,
    hidden_sizes='256,256',
    od_wait=None,  # Epochs of no validation improvement before break.
    val_size=0,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=0,
    cuda_device='',
    save_freq=-1,
    silent=False,
):
    use_gpu = cuda_device != ''
    # Get data into trainable form.
    x_data = torch.Tensor(dataset['observations'])
    y_data = torch.Tensor(dataset['actions'])
    val_key = 'advantages' if 'advantages' in dataset else 'values'
    val_data = torch.Tensor(dataset[val_key])
    tensor_data = [x_data, y_data, val_data]
    has_weights = 'weights' in dataset
    if has_weights:
        tensor_data.append(torch.Tensor(dataset['weights']))
    tr_data, val_data = split_supervised_data(
            tensor_data, val_size, batch_size, use_gpu)
    net = get_critic(x_data.shape[1], y_data.shape[1], hidden_sizes)
    trainer = ModelTrainer(
            model=net,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            cuda_device=cuda_device,
            save_freq=save_freq,
            save_path=save_dir,
    )
    trainer.fit(tr_data, epochs, val_data, od_wait,
                last_column_is_weights=has_weights)
    net.load_model(os.path.join(save_dir, 'model.pt'))
    return net


def sarsa_learn_critic(
    # A dictionary containing the qlearning standards.
    dataset,
    policy,
    save_dir,
    epochs,
    num_qs=1,
    hidden_sizes='256,256',
    gamma=0.99,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=0,
    cuda_device='',
    save_freq=-1,
    silent=False,
):
    use_gpu = cuda_device != ''
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]
    dataloader = get_qlearning_dataloader(dataset, cuda_device, batch_size,
                                          shuffle=True)
    qnets = [get_critic(obs_dim, act_dim, hidden_sizes) for _ in range(num_qs)]
    qts = [get_critic(obs_dim, act_dim, hidden_sizes) for _ in range(num_qs)]
    trainer = ActorCriticTrainer(
        policy=policy,
        qnets=qnets,
        qtargets=qts,
        gamma=gamma,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        cuda_device=cuda_device,
        save_freq=save_freq,
        save_path=save_dir,
    )
    trainer.train(dataloader, epochs, train_policy=False)
    return qnets


def load_critic(
    load_dir,
    obs_dim,
    act_dim,
    hidden_sizes='256,256',
    cuda_device='',
):
    critic = get_critic(obs_dim, act_dim, hidden_sizes)
    device = 'cpu' if cuda_device == '' else 'cuda:' + cuda_device
    critic.load_model(os.path.join(load_dir, 'model.pt'), map_location=device)
    return critic


def get_critic(
    obs_dim,
    act_dim,
    hidden_sizes='256,256',
):
    return Critic(
        state_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=s2i(hidden_sizes),
    )


def load_value_function(
    load_dir,
    cuda_device='',
):
    with open(os.path.join(load_dir, 'params.pkl'), 'rb') as f:
        params = pkl.load(f)
    vf = get_value_function(**params)
    device = 'cpu' if cuda_device == '' else 'cuda:' + cuda_device
    vf.load_model(os.path.join(load_dir, 'model.pt'), map_location=device)
    return vf


def get_value_function(
    obs_dim,
    hidden_sizes='256,256',
):
    return ValueFunction(
        state_dim=obs_dim,
        hidden_sizes=s2i(hidden_sizes),
    )
