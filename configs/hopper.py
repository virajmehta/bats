"""
Configs for hopper experiments.
"""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

base_config = OrderedDict(
    epsilon_planning=0.07,
    epsilon_neighbors=0.3,
    num_cpus=40,
    stitching_chunk_size=50000,
    normalize_obs=True,
)

# For any additional configurations, add them here.
HOPPER_CONFIGS = OrderedDict()

HOPPER_CONFIGS['hopper-expert'] = deepcopy(base_config)
HOPPER_CONFIGS['hopper-expert']['env_name'] = 'hopper-expert-v0'

HOPPER_CONFIGS['hopper-medexp'] = deepcopy(base_config)
HOPPER_CONFIGS['hopper-medexp']['env_name'] =\
    'hopper-medium-expert-v0'

HOPPER_CONFIGS['hopper-random'] = deepcopy(base_config)
HOPPER_CONFIGS['hopper-random']['env_name'] = 'hopper-random-v0'

HOPPER_CONFIGS['hopper-mixed'] = deepcopy(base_config)
HOPPER_CONFIGS['hopper-mixed']['env_name'] =\
    'hopper-medium-replay-v0'
HOPPER_CONFIGS['hopper-mixed']['load_model'] = Path('~/base/shared/models/hpv0-mixed')

HOPPER_CONFIGS['hopper-medium'] = deepcopy(base_config)
HOPPER_CONFIGS['hopper-medium']['env_name'] = 'hopper-medium-v0'

to_add = OrderedDict()
for k, v in HOPPER_CONFIGS.items():
    task_type = k[k.index('-') + 1:]
    config = deepcopy(v)
    config['use_all_planning_itrs'] = True
    config['continue_after_no_advantage'] = True
    config['num_stitching_iters'] = 20
    config['stitching_chunk_size'] = 5000
    # For mixed dataset edge distance = 0.726 +- 0.632
    # config['epsilon_neighbors'] = 0.3
    config['planning_quantile'] = 0.4
    config['epsilon_planning'] = 5
    config['verbose'] = True
    config['k_neighbors'] = 25
    config['max_stitch_length'] = 1
    to_add[k + '-tune'] = config
HOPPER_CONFIGS.update(to_add)
