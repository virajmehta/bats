"""
Policy construction through behavior cloning.
"""
import os

import torch
from torch.nn.functional import tanh
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from modelling.models import ModelTrainer, RegressionNN
from util import s2i


def train_policy(
    dataset,  # Dataset is a dictionary containing 'observations' and 'actions'
    save_dir,
    epochs,
    hidden_sizes='256,256',
    standardize_targets=False,
    od_wait=25,  # Epochs of no validation improvement before break.
    val_size=1000,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=0,
    cuda_device='',
    silent=False,
):
    use_gpu = cuda_device != ''
    # Get data into trainable form.
    x_data = torch.Tensor(dataset['observations'])
    y_data = torch.Tensor(dataset['actions'])
    tr_x, val_x, tr_y, val_y = train_test_split(x_data, y_data,
                                                test_size=val_size)
    tr_data = DataLoader(
        TensorDataset(tr_x, tr_y),
        batch_size=batch_size,
        shuffle=not use_gpu,
        pin_memory=use_gpu,
    )
    val_data = DataLoader(
        TensorDataset(tr_x, tr_y),
        batch_size=batch_size,
        shuffle=not use_gpu,
        pin_memory=use_gpu,
    )
    policy = get_policy(x_data.shape[1], y_data.shape[1],
                        hidden_sizes, standardize_targets)
    policy.set_standardization([
            (torch.mean(x_data, dim=0), torch.std(x_data, dim=0)),
            (torch.mean(y_data, dim=0), torch.std(y_data, dim=0)),
    ])
    trainer = ModelTrainer(
            model=policy,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            cuda_device=cuda_device,
            save_path=save_dir,
    )
    trainer.fit(tr_data, epochs, val_data, od_wait)
    policy.load_model(os.path.join(save_dir, 'model.pt'))
    return policy


def load_policy(
    load_dir,
    obs_dim,
    act_dim,
    hidden_sizes='256,256',
    standardize_targets=False,
    cuda_device='',
):
    policy = get_policy(obs_dim, act_dim, hidden_sizes, standardize_targets)
    policy.load_model(os.path.join(load_dir, 'model.pt'))
    return policy


def get_policy(
        obs_dim,
        act_dim,
        hidden_sizes='256,256',
        standardize_targets=False,
):
    return RegressionNN(
        input_dim=obs_dim,
        output_dim=act_dim,
        hidden_sizes=s2i(hidden_sizes),
        output_activation=tanh,
        standardize_targets=standardize_targets,
    )
