"""
Configs for maze experiments.
"""
from collections import OrderedDict
from pathlib import Path
from copy import deepcopy

base_config = OrderedDict(
    epsilon_planning=0.01,
    epsilon_neighbors=0.5,
    num_cpus=6,
    num_stitching_iters=200,
    stitching_chunk_size=1000,
    bc_epochs=50,
    bc_every_iter=True,
    od_wait=None,
    temperature=0.1,
    rollout_stitch_temperature=0.2,
    max_stitch_length=3,
    use_all_planning_itrs=True,
    return_threshold=-1000,
    starts_from_dataset=True,
    reward_offset=25,
)

dataset_base_path = Path('/spin/datasets/d4rl/pendulum')
model_base_path = Path('models/')
ENV_NAME = 'Pendulum-v0'
# For any additional configurations, add them here.
PENDULUM_CONFIGS = OrderedDict()

PENDULUM_CONFIGS['pendulum-1k'] = deepcopy(base_config)
PENDULUM_CONFIGS['pendulum-1k']['env_name'] = ENV_NAME
PENDULUM_CONFIGS['pendulum-1k']['offline_dataset_path'] = 'datasets/pendulum/pendulum-random-1000.hdf5'
PENDULUM_CONFIGS['pendulum-1k']['load_model'] = 'models/pendulum/1k_large'
# PENDULUM_CONFIGS['pendulum-1k']['load_bisim_model'] = model_base_path / '64,64,64_pendulum_random_1k_bisim'

PENDULUM_CONFIGS['pendulum-10k'] = deepcopy(base_config)
PENDULUM_CONFIGS['pendulum-10k']['env_name'] = ENV_NAME
PENDULUM_CONFIGS['pendulum-10k']['offline_dataset_path'] = 'datasets/pendulum/pendulum-random-10k-10starts.hdf5'
PENDULUM_CONFIGS['pendulum-10k']['load_model'] = 'models/pendulum/10k_10starts'
PENDULUM_CONFIGS['pendulum-10k']['epsilon_neighbors'] = 0.1
PENDULUM_CONFIGS['pendulum-10k']['epsilon_planning'] = 0.1


to_add = OrderedDict()
for k, v in PENDULUM_CONFIGS.items():
    task_type = k[k.index('-') + 1:]
    if 'maze' not in task_type:
        task_type = task_type + 'maze'
    config = deepcopy(v)
    config['use_all_planning_itrs'] = True
    config['continue_after_no_advantage'] = True
    # config['num_stitching_iters'] = 25
    # For umaze dataset edge distance = 0.11 +- 0.03
    config['planning_quantile'] = 0.4
    config['epsilon_planning'] = 1.5
    config['verbose'] = True
    to_add[k + '-tune'] = config
PENDULUM_CONFIGS.update(to_add)
