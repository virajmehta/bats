"""
Policy construction through behavior cloning.
"""
import os

import numpy as np
import torch
from torch.nn.functional import tanh
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from modelling.models import ModelTrainer, DeterministicPolicy,\
        StochasticPolicy, AWACTrainer, QNet
from util import s2i


def learn_awac_policy(
    dataset, # Dictionary containing sarsd and optionally 'values'.
    save_path,
    epochs,
    policy_hidden_sizes='256,256',
    qnet_hidden_sizes='256,256',
    n_qnets=2,
    batch_size=256,
    learning_rate=1e-3,
    soft_tau=5e-3, # Weight for target net soft update.
    weight_decay=0,
    cuda_device='',
    silent=False,
    save_freq=-1,
    qnets=None, # List of pretrained qnets to use.
    env = None, # Gym environment for doing evaluations.
    max_ep_len: int = 1000,
    num_eval_eps: int = 10,
):
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]
    # Initialize networks and make trainer.
    policy = get_policy(obs_dim, act_dim, policy_hidden_sizes, False)
    if qnets is None:
        qnets, qtargets = [[get_qnet(obs_dim, act_dim, qnet_hidden_sizes)
                            for _ in range(n_qnets)] for _ in range(2)]
    else:
        qtargets = []
        for qnet in qnets:
            qtarg = get_qnet(obs_dim, act_dim, qnet_hidden_sizes)
            qtarg.load_state_dict(qnet.state_dict())
            qtargets.append(qtarg)
    trainer = AWACTrainer(
        policy=policy,
        qnets=qnets,
        qtargets=qtargets,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        soft_tau=soft_tau,
        cuda_device=cuda_device,
        save_path=save_path,
        save_freq=save_freq,
        silent=silent,
        env=env,
        max_ep_len=max_ep_len,
        num_eval_eps=num_eval_eps,
    )
    # Organize data and train.
    rand_idx = np.arange(dataset['observations'].shape[0])
    np.random.shuffle(rand_idx)
    tensor_data = [
        torch.Tensor(dataset['observations'][rand_idx]),
        torch.Tensor(dataset['actions'][rand_idx]),
        torch.Tensor(dataset['rewards'][rand_idx]),
        torch.Tensor(dataset['next_observations'][rand_idx]),
        torch.Tensor(dataset['terminals'][rand_idx]),
    ]
    if 'values' in dataset:
        tensor_data.append(dataset['values'][rand_idx])
    use_gpu = cuda_device != ''
    trainer.train(
        dataset=DataLoader(
            TensorDataset(*tensor_data),
            batch_size=batch_size,
            pin_memory=use_gpu,
            shuffle=not use_gpu,
        ),
        epochs=epochs
    )
    return policy, qnets


def behavior_clone(
    dataset,  # Dataset is a dictionary containing 'observations' and 'actions'
    save_dir,
    epochs,
    hidden_sizes='256,256',
    deterministic=False, # Whether we have deterministic policy.
    standardize_targets=False,
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
    tensor_data = [x_data, y_data]
    has_weights = 'weights' in dataset
    if has_weights:
        tensor_data.append(torch.Tensor(dataset['weights']))
    tr_data, val_data = split_supervised_data(
            tensor_data, val_size, batch_size, use_gpu)
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
            save_freq=save_freq,
            track_stats=['MSE', 'LogPis'],
    )
    trainer.fit(tr_data, epochs, val_data, od_wait,
                last_column_is_weights=has_weights)
    policy.load_model(os.path.join(save_dir, 'model.pt'))
    return policy


def supervise_qlearning(
    dataset,  # A dictionary containing 'observations', 'actions', 'values'
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
    val_data = torch.Tensor(dataset['values'])
    tensor_data = [x_data, y_data, val_data]
    has_weights = 'weights' in dataset
    if has_weights:
        tensor_data.append(torch.Tensor(dataset['weights']))
    tr_data, val_data = split_supervised_data(
            tensor_data, val_size, batch_size, use_gpu)
    qnet = get_qnet(x_data.shape[1], y_data.shape[1], hidden_sizes)
    trainer = ModelTrainer(
            model=qnet,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            cuda_device=cuda_device,
            save_freq=save_freq,
            save_path=save_dir,
    )
    trainer.fit(tr_data, epochs, val_data, od_wait,
                last_column_is_weights=has_weights)
    qnet.load_model(os.path.join(save_dir, 'model.pt'))
    return qnet


def split_supervised_data(tensor_data, val_size, batch_size, use_gpu):
    td_size = len(tensor_data)
    if val_size > 0:
        tensor_data = train_test_split(*tensor_data, test_size=val_size)
        tensor_data = ([tensor_data[i] for i in range(0, td_size, 2)]
                       + [tensor_data[i] for i in range(1, td_size, 2)])
        val_data = DataLoader(
            TensorDataset(*tensor_data[td_size:]),
            batch_size=batch_size,
            shuffle=not use_gpu,
            pin_memory=use_gpu,
        )
    else:
        val_data = None
    tr_data = DataLoader(
        TensorDataset(*tensor_data[:td_size]),
        batch_size=batch_size,
        shuffle=not use_gpu,
        pin_memory=use_gpu,
    )
    return tr_data, val_data


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
            target_entropy=-3,
            standardize_targets=standardize_targets,
        )


def load_qnet(
    load_dir,
    obs_dim,
    act_dim,
    hidden_sizes='256,256',
    cuda_device='',
):
    qnet = get_qnet(obs_dim, act_dim, hidden_sizes)
    device = 'cpu' if cuda_device == '' else 'cuda:' + cuda_device
    qnet.load_model(os.path.join(load_dir, 'model.pt'), map_location=device)
    return qnet


def get_qnet(
    obs_dim,
    act_dim,
    hidden_sizes='256,256',
):
    return QNet(
        state_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=s2i(hidden_sizes),
    )
