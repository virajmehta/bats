"""
Construction for bisumulation modelling.
"""
import os
from collections import OrderedDict
from copy import deepcopy
import pickle as pkl

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

from modelling.dynamics_construction import get_pnn_optimizer
from modelling.models import BisimulationModel
from modelling.trainers import ModelTrainer
from modelling.utils.torch_utils import swish
from util import s2i


DEFAULT_VARIANT = OrderedDict(
    num_dyn_nets=5,
    encoder_architecture='256,128,64',
    dyn_architecture='200,200,200',
    dyn_latent_dim=200,
    mean_head_architecture='',
    logvar_head_architecture='',
    logvar_lower=-10,
    logvar_upper=0.5,
    bound_loss_coef=1e-2,
    pnn_activation=swish,
)


def train_bisim(
        latent_dim,
        epochs,
        dataset,
        n_members,
        save_dir,
        od_wait=None,  # Epochs of no validation improvement before break.
        val_size=1000,
        batch_size=256,
        bisim_params={},
        cuda_device='',
        save_freq=-1,
        batch_updates_per_epoch=None,
        validation_batches_per_epoch=None,
        model=None, # If there is a pre-existing model use this.
        train_loops_per_epoch=1,
):
    # Set devices and get the data.
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]
    tr_data, val_data = get_tr_val_data(dataset,
                                        batch_size=batch_size,
                                        val_size=val_size,
                                        use_gpu=cuda_device != '')
    # Initialize model.
    params = deepcopy(DEFAULT_VARIANT)
    params.update(bisim_params)
    params['num_dyn_nets'] = n_members
    params['obs_dim'] = obs_dim
    params['act_dim'] = act_dim
    params['latent_dim'] = latent_dim
    if model is None:
        model = get_bisim(**params)
    # Create trainer and save off params.
    trainer = make_trainer(model,
                           n_members,
                           save_dir)
    with open(os.path.join(save_dir, 'params.pkl'), 'wb') as f:
        pkl.dump(params, f)
    # Do training.
    trainer.fit(tr_data, epochs, val_data, od_wait,
                batch_updates_per_epoch=batch_updates_per_epoch)
    return model, trainer


def make_trainer(model,
                 n_members,
                 save_dir,
                 learning_rate=1e-3,
                 weight_decay=0,
                 cuda_device='',
                 silent=False,
                 train_loops_per_epoch=1,
                 save_freq=-1):
    # Create optimizers.
    optimizers = {'Encoder': torch.optim.Adam(model.encoder.parameters(),
                                              lr=learning_rate,
                                              weight_decay=weight_decay)}
    for i in range(n_members):
        pnn = getattr(model, 'pnn_%d' % i)
        optimizers['PNN%d' % i] = get_pnn_optimizer(pnn,
                                                    learning_rate=learning_rate)
    trainer = ModelTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        cuda_device=cuda_device,
        save_path=save_dir,
        silent=silent,
        optimizers=optimizers,
        track_stats=['EncoderLoss', 'PNN0_MSE'],
        save_best_model=False,
        save_freq=save_freq,
        train_loops_per_epoch=train_loops_per_epoch,
    )
    # Do training.
    return trainer


def fine_tune_bisim(
        trainer,
        epochs,
        dataset,
        od_wait=None,  # Epochs of no validation improvement before break.
        val_size=1000,
        batch_size=256,
        cuda_device='',
        batch_updates_per_epoch=None,
):
    # Set devices and get the data.
    tr_data, val_data = get_tr_val_data(dataset,
                                        batch_size=batch_size,
                                        val_size=val_size,
                                        use_gpu=cuda_device != '')
    trainer.fit(tr_data, epochs, val_data, od_wait,
                batch_updates_per_epoch=batch_updates_per_epoch)
    return trainer


def get_bisim(
        obs_dim,
        act_dim,
        latent_dim,
        num_dyn_nets=5,
        encoder_architecture='256,128,64',
        dyn_architecture='200,200,200',
        dyn_latent_dim=200,
        mean_head_architecture='',
        logvar_head_architecture='',
        logvar_lower=-10,
        logvar_upper=0.5,
        bound_loss_coef=1e-2,
        pnn_activation=swish,
):
    return BisimulationModel(
        obs_dim=obs_dim,
        act_dim=act_dim,
        latent_dim=latent_dim,
        num_dyn_nets=num_dyn_nets,
        encoder_architecture=s2i(encoder_architecture),
        dyn_architecture=s2i(dyn_architecture),
        dyn_latent_dim=dyn_latent_dim,
        mean_head_architecture=s2i(mean_head_architecture),
        logvar_head_architecture=s2i(logvar_head_architecture),
        logvar_bounds=[logvar_lower, logvar_upper],
        bound_loss_coef=bound_loss_coef,
        pnn_hidden_activation=pnn_activation,
    )


def get_tr_val_data(
        dataset,
        batch_size=256,
        val_size=1000,
        use_gpu=False,
        shuffle=True,
):
    # Spit data.
    datas = []
    for k in ['observations', 'actions', 'rewards', 'next_observations']:
        to_add = torch.Tensor(dataset[k])
        if len(to_add.shape) == 1:
            to_add = to_add.reshape(-1, 1)
        datas.append(to_add)
    split_data = train_test_split(*datas, test_size=val_size)
    train_data = [sd for idx, sd in enumerate(split_data) if idx % 2 == 0]
    val_data = [sd for idx, sd in enumerate(split_data) if idx % 2 == 1]
    # Make dataloaders.
    tr_data = DataLoader(
        TensorDataset(*train_data),
        batch_size=batch_size,
        shuffle=shuffle or not use_gpu,
        pin_memory=not shuffle and use_gpu,
    )
    val_data = DataLoader(
        TensorDataset(*val_data),
        batch_size=batch_size,
        shuffle=not use_gpu,
        pin_memory=use_gpu,
    )
    return tr_data, val_data


def load_bisim(load_dir):
    with open(load_dir / 'params.pkl', 'rb') as f:
        params = pkl.load(f)
    model = get_bisim(**params)
    model.load_model(load_dir / 'model.pt', map_location=torch.device('cpu'))
    return model
