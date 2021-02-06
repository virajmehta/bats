"""
Policy construction through behavior cloning.
"""
import os

import torch
from torch.nn.functional import tanh
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from modelling.models import ModelTrainer, DeterministicPolicy,\
        StochasticPolicy
from util import s2i


def behavior_clone(
    dataset,  # Dataset is a dictionary containing 'observations' and 'actions'
    save_dir,
    epochs,
    hidden_sizes='256,256',
    deterministic=False, # Whether we have deterministic policy.
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
    tensor_data = [x_data, y_data]
    has_weights = 'weights' in dataset
    if has_weights:
        tensor_data.append(torch.Tensor(dataset['weights']))
    if val_size > 0:
        tensor_data = train_test_split(*tensor_data, test_size=val_size)
        tensor_data = ([tensor_data[i] for i in range(0, len(tensor_data), 2)]
                    + [tensor_data[i] for i in range(1, len(tensor_data), 2)])
        val_data = DataLoader(
            TensorDataset(*tensor_data[2 + has_weights:]),
            batch_size=batch_size,
            shuffle=not use_gpu,
            pin_memory=use_gpu,
        )
    else:
        val_data = None
    tr_data = DataLoader(
        TensorDataset(*tensor_data[:2 + has_weights]),
        batch_size=batch_size,
        shuffle=not use_gpu,
        pin_memory=use_gpu,
    )
    policy = get_policy(x_data.shape[1], y_data.shape[1],
                        hidden_sizes, standardize_targets)
    standardizers = [(torch.mean(x_data, dim=0), torch.std(x_data, dim=0)),
                     (torch.mean(y_data, dim=0), torch.std(y_data, dim=0))]
    policy.set_standardization(standardizers)
    trainer = ModelTrainer(
            model=policy,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            cuda_device=cuda_device,
            save_path=save_dir,
    )
    trainer.fit(tr_data, epochs, val_data, od_wait,
                last_column_is_weights=has_weights)
    policy.load_model(os.path.join(save_dir, 'model.pt'))
    return policy


def load_policy(
    load_dir,
    obs_dim,
    act_dim,
    deterministic=False,
    hidden_sizes='256,256',
    standardize_targets=False,
    cuda_device='',
):
    policy = get_policy(obs_dim, act_dim, hidden_sizes, deterministic,
                        standardize_targets)
    device = 'cpu' if cuda_device == '' else 'cuda:' + cuda_device
    policy.load_model(os.path.join(load_dir, 'model.pt'), map_location=device)
    return policy


def get_policy(
        obs_dim,
        act_dim,
        hidden_sizes='256,256',
        deterministic=False,
        standardize_targets=False,
):
    if deterministic:
        return DeterministicPolicy(
            input_dim=obs_dim,
            output_dim=act_dim,
            hidden_sizes=s2i(hidden_sizes),
            output_activation=tanh,
            standardize_targets=standardize_targets,
        )
    else:
        last_comma_idx = -1 * hidden_sizes[::-1].find(',') - 1
        return StochasticPolicy(
            input_dim=obs_dim,
            output_dim=act_dim,
            encoder_hidden_sizes=s2i(hidden_sizes[:last_comma_idx]),
            latent_dim=int(hidden_sizes[last_comma_idx + 1:]),
            mean_hidden_sizes=[],
            logvar_hidden_sizes=[],
            tanh_transform=True,
            standardize_targets=standardize_targets
        )
