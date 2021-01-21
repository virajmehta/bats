"""
Functions for training and loading in dyamics models.
"""
import os

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

from modelling.models import ModelTrainer, PNN, RegressionNN
from modelling.utils.torch_utils import swish
from util import s2i


def train_ensemble(
        dataset,
        n_members,
        n_elites,
        save_dir,
        epochs,
        od_wait=25,  # Epochs of no validation improvement before break.
        val_size=1000,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=0,
        model_type='PNN',
        model_params={},
        cuda_device='',
        silent=False,
):
    # Set devices and get the data.
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]
    x_data, y_data = prepare_data_for_dyn_training(dataset)
    tr_data, val_data = get_tr_val_data(x_data, y_data,
                                        batch_size=batch_size,
                                        val_size=val_size,
                                        use_gpu=cuda_device != '')
    # Do training for all models.
    ensemble, ensemble_scores = [], []
    for ens_idx in range(1, n_members + 1):
        if not silent:
            print('=========ENSEMBLE MEMBER %d========' % ens_idx)
        model, optimizer = get_model(obs_dim, act_dim, model_type,
                                     model_params)
        model.set_standardization([
            (torch.mean(x_data, dim=0), torch.std(x_data, dim=0)),
            # Set y standardization so nothing happens to output space.
            (torch.zeros(y_data.shape[1]), torch.ones(y_data.shape[1])),
        ])
        model_save = os.path.join(save_dir, 'member_%d' % ens_idx)
        tune_metric = 'MSE' if model_type == 'PNN' else 'Loss'
        trainer = ModelTrainer(
                model=model,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                cuda_device=cuda_device,
                save_path=model_save,
                silent=silent,
                optimizer=optimizer,
                validation_tune_metric=tune_metric,
        )
        trainer.fit(tr_data, epochs, val_data, od_wait)
        model.load_model(os.path.join(model_save, 'model.pt'))
        ensemble.append(model)
        ensemble_scores.append(trainer._best_val_loss)
    elite_idxs = np.argsort(ensemble_scores)[:n_elites]
    elite_str = ','.join(['member_%d' % (ei + 1) for ei in elite_idxs])
    if not silent:
        print('Elite Members: ', elite_str)
    with open(os.path.join(save_dir, 'elites.txt'), 'w') as f:
        f.write(elite_str)
    return [ensemble[idx] for idx in elite_idxs]


def load_ensemble(
        load_dir,
        obs_dim,
        act_dim,
        model_type='PNN',
        model_params={},
        cuda_device='',
):
    ensemble = []
    if os.path.exists(os.path.join(load_dir, 'elites.txt')):
        with open(os.path.join(load_dir, 'elites.txt'), 'r') as f:
            member_names = [n for n in f.readlines()[0].split(',')
                            if 'member' in n]
    else:
        member_names = [f for f in os.listdir(load_dir) if 'member' in f]
    device = 'cpu' if cuda_device == '' else 'cuda:' + cuda_device
    for memb_dir in member_names:
        memb_path = os.path.join(load_dir, memb_dir, 'model.pt')
        member = get_model(obs_dim, act_dim, model_type, model_params)[0]
        member.load_model(memb_path, map_location=device)
        ensemble.append(member)
    return ensemble


def get_model(obs_dim, act_dim, model_type,
              model_params={}, learning_rate=1e-3, weight_decay=0):
    if model_type == 'PNN':
        model = get_pnn(obs_dim, act_dim, **model_params)
        optimizer = get_pnn_optimizer(model, learning_rate=learning_rate)
    elif model_type == 'NN':
        model = get_nn(obs_dim, act_dim, **model_params)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
    else:
        raise ValueError('Unknown model type: %s' % model_type)
    return model, optimizer


def get_pnn(
        obs_dim,
        act_dim,
        encoder_hidden='200,200,200',
        latent_dim=200,
        mean_hidden='',
        logvar_hidden='',
        logvar_lower=-10,
        logvar_upper=0.5,
        bound_loss_coef=1e-2,
        activation=swish,
):
    return PNN(
        input_dim=obs_dim + act_dim,
        encoder_hidden_sizes=s2i(encoder_hidden),
        latent_dim=latent_dim,
        mean_hidden_sizes=s2i(mean_hidden),
        logvar_hidden_sizes=s2i(logvar_hidden),
        output_dim=obs_dim + 1,
        logvar_bounds=[logvar_lower, logvar_upper],
        bound_loss_coef=bound_loss_coef,
        hidden_activation=activation,
        standardize_targets=False,
    )


def get_pnn_optimizer(pnn, learning_rate=1e-3):
    # Decays taken from MOPO repo.
    decays = [0.000025, 0.00005, 0.000075, 0.000075]
    head_decay = 0.0001
    custom_params = []
    for layer_idx in range(pnn._gaussian_net.encode_net.n_layers):
        param_dict = {'params': getattr(pnn._gaussian_net.encode_net,
                                        'linear_%d' % layer_idx).parameters()}
        if layer_idx < len(decays):
            param_dict['weight_decay'] = decays[layer_idx]
        custom_params.append(param_dict)
    for layer_idx in range(pnn._gaussian_net.mean_net.n_layers):
        param_dict = {'params': getattr(pnn._gaussian_net.mean_net,
                                        'linear_%d' % layer_idx).parameters(),
                      'weight_decay': head_decay}
        custom_params.append(param_dict)
    for layer_idx in range(pnn._gaussian_net.logvar_net.n_layers):
        param_dict = {'params': getattr(pnn._gaussian_net.logvar_net,
                                        'linear_%d' % layer_idx).parameters(),
                      'weight_decay': head_decay}
        custom_params.append(param_dict)
    custom_params.append({'params': [pnn._min_logvar, pnn._max_logvar]})
    return torch.optim.Adam(custom_params, lr=learning_rate)


def get_nn(
        obs_dim,
        act_dim,
        hidden_sizes='200,200,200,200',
        activation=swish,
):
    return RegressionNN(
        input_dim=obs_dim + act_dim,
        output_dim=obs_dim + 1,
        hidden_sizes=s2i(hidden_sizes),
        hidden_activation=activation,
        standardize_targets=False,
    )


def prepare_data_for_dyn_training(dataset):
    # Make data into input, outputs.
    st = dataset['observations']
    st1 = dataset['next_observations']
    acts = dataset['actions']
    rews = dataset['rewards'].reshape(-1, 1)
    x_data = torch.Tensor(np.hstack([st, acts]))
    y_data = torch.cat([
        torch.Tensor(rews),
        torch.Tensor(st1 - st),
    ], dim=1)
    return x_data, y_data


def prepare_model_inputs(obs, actions):
    return torch.Tensor(np.hstack([obs, actions]))


def get_tr_val_data(
        x_data,
        y_data,
        batch_size=256,
        val_size=1000,
        use_gpu=False,
):
    # Spit data.
    tr_x, val_x, tr_y, val_y = train_test_split(
        x_data,
        y_data,
        test_size=val_size,
    )
    # Make dataloaders.
    tr_data = DataLoader(
        TensorDataset(tr_x, tr_y),
        batch_size=batch_size,
        shuffle=not use_gpu,
        pin_memory=use_gpu,
    )
    val_data = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=batch_size,
        shuffle=not use_gpu,
        pin_memory=use_gpu,
    )
    return tr_data, val_data
