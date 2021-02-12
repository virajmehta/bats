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
        StochasticPolicy, AWACTrainer, Critic
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
    train_loops_per_epoch: int = 1,
):
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]
    # Initialize networks and make trainer.
    policy = get_policy(obs_dim, act_dim, policy_hidden_sizes, False)
    if qnets is None:
        qnets, qtargets = [[get_critic(obs_dim, act_dim, qnet_hidden_sizes)
                            for _ in range(n_qnets)] for _ in range(2)]
    else:
        qtargets = []
        for qnet in qnets:
            qtarg = get_critic(obs_dim, act_dim, qnet_hidden_sizes)
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
        train_loops_per_epoch=train_loops_per_epoch,
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
        tensor_data.append(torch.Tensor(dataset['values'][rand_idx]))
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

def advantage_weighted_regression(
    dataset,  # Dataset is a dictionary with obs, acts, and optionally adv.
    save_dir,
    epochs,
    advnet=None, # Network that predicts advantages.
    adv_weight=1,
    max_weighting=20,
    cuda_device='',
    **kwargs
):
    """This is the same as behavior cloning but computes weights for you."""
    if advnet is not None:
        advdata = DataLoader(
            TensorDataset(torch.Tensor(dataset['observations']),
                          torch.Tensor(dataset['actions'])),
            batch_size=1024,
            shuffle=False,
            pin_memory=cuda_device != '',
        )
        advantages = []
        for batch in advdata:
            with torch.no_grad():
                advantages.append(advnet(batch[0], batch[1]))
        dataset['weights'] = torch.clamp(
                (torch.cat(advantages, dim=0) / adv_weight).exp(),
                max=max_weighting)
    elif 'advantages' in dataset:
        dataset['weights'] = np.minimum(
            np.exp(dataset['advantages'] / adv_weight), max_weighting)
    else:
        raise ValueError('Either need advantages in dataset or advnet.')
    behavior_clone(
        dataset=dataset,
        save_dir=save_dir,
        epochs=epochs,
        cuda_device=cuda_device,
        **kwargs
    )


def behavior_clone(
    dataset,  # Dataset is a dictionary containing 'observations' and 'actions'
    save_dir,
    epochs,
    hidden_sizes='256,256',
    deterministic=False, # Whether we have deterministic policy.
    standardize_targets=False,
    od_wait=None,  # Epochs of no validation improvement before break.
    val_size=0,
    val_dataset=None, # Dataset w same structure as dataset but for validation.
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=0,
    cuda_device='',
    save_freq=-1,
    silent=False,
    env = None, # Gym environment for doing evaluations.
    max_ep_len: int = 1000,
    num_eval_eps: int = 10,
    train_loops_per_epoch: int = 1,
):
    use_gpu = cuda_device != ''
    # Get data into trainable form.
    headers = ['observations', 'actions']
    has_weights = 'weights' in dataset
    if has_weights:
        headers.append('weights')
    shuff_idxs = np.arange(len(dataset['observations']))
    np.random.shuffle(shuff_idxs)
    tensor_data = [torch.Tensor(dataset[h][shuff_idxs]) for h in headers]
    # Train-validation split.
    if val_dataset is None:
        tr_data, val_data = split_supervised_data(
                tensor_data, val_size, batch_size, use_gpu)
    else:
        tr_data = DataLoader(
            TensorDataset(*tensor_data),
            batch_size=batch_size,
            shuffle=not use_gpu,
            pin_memory=use_gpu,
        )
        shuff_idxs = np.arange(len(val_dataset['observations']))
        np.random.shuffle(shuff_idxs)
        val_data = DataLoader(
            TensorDataset(*[torch.Tensor(val_dataset[h][shuff_idxs])
                            for h in headers]),
            batch_size=batch_size,
            shuffle=not use_gpu,
            pin_memory=use_gpu,
        )
    # Initialize model.
    tr_x, tr_y = tensor_data[:2]
    policy = get_policy(tr_x.shape[1], tr_y.shape[1],
                        hidden_sizes, standardize_targets)
    standardizers = [(torch.mean(tr_x, dim=0), torch.std(tr_x, dim=0)),
                     (torch.mean(tr_y, dim=0), torch.std(tr_y, dim=0))]
    policy.set_standardization(standardizers)
    trainer = ModelTrainer(
            model=policy,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            cuda_device=cuda_device,
            save_path=save_dir,
            save_freq=save_freq,
            env=env,
            max_ep_len=max_ep_len,
            num_eval_eps=num_eval_eps,
            train_loops_per_epoch=train_loops_per_epoch,
    )
    trainer.fit(tr_data, epochs, val_data, od_wait,
                last_column_is_weights=has_weights)
    policy.load_model(os.path.join(save_dir, 'model.pt'))
    return policy


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


def split_supervised_data(tensor_data, val_size, batch_size, use_gpu):
    td_size = len(tensor_data)
    if val_size > 0:
        tensor_data = train_test_split(*tensor_data, test_size=val_size)
        tensor_data = ([tensor_data[i] for i in range(0, len(tensor_data), 2)]
                    + [tensor_data[i] for i in range(1, len(tensor_data), 2)])
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
            train_alpha_entropy=True,
            add_entropy_bonus=True,
            standardize_targets=standardize_targets,
        )


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
