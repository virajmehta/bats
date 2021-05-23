"""
Configs for maze experiments.
"""
from collections import OrderedDict
from pathlib import Path
from copy import deepcopy

base_config = OrderedDict(
    epsilon_planning=0.1,
    # epsilon_neighbors=0.225,
    num_cpus=6,
    num_stitching_iters=20,
    stitching_chunk_size=100,
    bc_epochs=25,
    bc_every_iter=True,
    top_percent_starts=None,
    temperature=0.1,
    od_wait=None,
    # k_neighbors=10,
    max_stitch_length=2,
)

dataset_base_path = Path('/spin/datasets/d4rl/mountaincar')
model_base_path = Path('models/')
ENV_NAME = 'MountainCarContinuous-v0'
# For any additional configurations, add them here.
MOUNTAINCAR_CONFIGS = OrderedDict()

MOUNTAINCAR_CONFIGS['mountaincar-1k'] = deepcopy(base_config)
MOUNTAINCAR_CONFIGS['mountaincar-1k']['env_name'] = ENV_NAME
MOUNTAINCAR_CONFIGS['mountaincar-1k']['offline_dataset_path'] = dataset_base_path / 'mountaincar-random-1000.hdf5'
MOUNTAINCAR_CONFIGS['mountaincar-1k']['load_model'] = model_base_path / '64,64,64_mc_random_1k'
MOUNTAINCAR_CONFIGS['mountaincar-1k']['load_bisim_model'] = model_base_path / '64,64,64_mc_random_1k_bisim'
MOUNTAINCAR_CONFIGS['mountaincar-1k']['dynamics_encoder_hidden'] = '64,64'
MOUNTAINCAR_CONFIGS['mountaincar-1k']['dynamics_latent_dim'] = 64
MOUNTAINCAR_CONFIGS['mountaincar-10k'] = deepcopy(base_config)
MOUNTAINCAR_CONFIGS['mountaincar-10k']['env_name'] = ENV_NAME
MOUNTAINCAR_CONFIGS['mountaincar-10k']['offline_dataset_path'] = dataset_base_path / 'mountaincar-random-10000.hdf5'
MOUNTAINCAR_CONFIGS['mountaincar-10k']['load_model'] = model_base_path / '64,64,64_mc_random_10k'
MOUNTAINCAR_CONFIGS['mountaincar-10k']['load_bisim_model'] = model_base_path / '64,64,64_mc_random_10k_bisim'
MOUNTAINCAR_CONFIGS['mountaincar-10k']['dynamics_encoder_hidden'] = '64,64'
MOUNTAINCAR_CONFIGS['mountaincar-10k']['dynamics_latent_dim'] = 64
MOUNTAINCAR_CONFIGS['mountaincar-5'] = deepcopy(base_config)
MOUNTAINCAR_CONFIGS['mountaincar-5']['env_name'] = ENV_NAME
MOUNTAINCAR_CONFIGS['mountaincar-5']['offline_dataset_path'] = dataset_base_path / 'mountaincar_5.hdf5'
MOUNTAINCAR_CONFIGS['mountaincar-5']['load_model'] = model_base_path / 'mc_5'
MOUNTAINCAR_CONFIGS['mountaincar-5']['load_bisim_model'] = None
MOUNTAINCAR_CONFIGS['mountaincar-5']['dynamics_encoder_hidden'] = '64,64'
MOUNTAINCAR_CONFIGS['mountaincar-5']['dynamics_latent_dim'] = 64
MOUNTAINCAR_CONFIGS['mountaincar-mixed'] = deepcopy(base_config)
MOUNTAINCAR_CONFIGS['mountaincar-mixed']['env_name'] = ENV_NAME
MOUNTAINCAR_CONFIGS['mountaincar-mixed']['offline_dataset_path'] = dataset_base_path / 'mountaincar-mixed-v2.hdf5'
MOUNTAINCAR_CONFIGS['mountaincar-mixed']['load_model'] = model_base_path / 'mc_mixed'
MOUNTAINCAR_CONFIGS['mountaincar-mixed']['load_bisim_model'] = None
MOUNTAINCAR_CONFIGS['mountaincar-mixed']['dynamics_encoder_hidden'] = '64,64'
MOUNTAINCAR_CONFIGS['mountaincar-mixed']['dynamics_latent_dim'] = 64

to_add = OrderedDict()
for k, v in MOUNTAINCAR_CONFIGS.items():
    task_type = k[k.index('-') + 1:]
    config = deepcopy(v)
    config['use_all_planning_itrs'] = True
    # config['continue_after_no_advantage'] = True
    # config['num_stitching_iters'] = 25
    # For umaze dataset edge distance = 0.11 +- 0.03
    config['epsilon_planning'] = 1.5
    config['verbose'] = True
    to_add[k + '-tune'] = config
MOUNTAINCAR_CONFIGS.update(to_add)
