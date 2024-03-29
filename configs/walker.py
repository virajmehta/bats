"""
Configs for walker experiments.
"""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

base_config = OrderedDict(
    epsilon_planning=0.2,
    epsilon_neighbors=1.3,
    num_cpus=40,
    stitching_chunk_size=50000,
    normalize_obs=True,
    ni=10,
)

# For any additional configurations, add them here.
WALKER_CONFIGS = OrderedDict()

WALKER_CONFIGS['walker-expert'] = deepcopy(base_config)
WALKER_CONFIGS['walker-expert']['env_name'] = 'walker2d-expert-v0'

WALKER_CONFIGS['walker-medexp'] = deepcopy(base_config)
WALKER_CONFIGS['walker-medexp']['env_name'] =\
    'walker2d-medium-expert-v0'

WALKER_CONFIGS['walker-random'] = deepcopy(base_config)
WALKER_CONFIGS['walker-random']['env_name'] = 'walker2d-random-v0'

WALKER_CONFIGS['walker-mixed'] = deepcopy(base_config)
WALKER_CONFIGS['walker-mixed']['env_name'] =\
    'walker2d-medium-replay-v0'
WALKER_CONFIGS['walker-mixed']['load_model'] = Path('~/bats/models/wkv0-mixed').expanduser()

WALKER_CONFIGS['walker-adv'] = deepcopy(base_config)
WALKER_CONFIGS['walker-adv']['env_name'] =\
    'walker2d-medium-replay-v0'
WALKER_CONFIGS['walker-adv']['load_model'] = Path('~/bats/models/walker_adv').expanduser()
WALKER_CONFIGS['walker-adv']['offline_dataset_path'] = Path('~/.d4rl/datasets/walker-adv-mixed.hdf5').expanduser()

WALKER_CONFIGS['walker-medium'] = deepcopy(base_config)
WALKER_CONFIGS['walker-medium']['env_name'] = 'walker2d-medium-v0'

to_add = OrderedDict()
for k, v in WALKER_CONFIGS.items():
    task_type = k[k.index('-') + 1:]
    config = deepcopy(v)
    config['use_all_planning_itrs'] = True
    config['continue_after_no_advantage'] = True
    config['num_stitching_iters'] = 40
    config['stitching_chunk_size'] = 5000
    # For mixed dataset edge distance = 1.83 +- 1.34
    # config['epsilon_neighbors'] = 0.2
    config['planning_quantile'] = 0.4
    config['epsilon_planning'] = 10
    config['verbose'] = True
    config['k_neighbors'] = 25
    config['max_stitch_length'] = 1
    to_add[k + '-tune'] = config
WALKER_CONFIGS.update(to_add)
