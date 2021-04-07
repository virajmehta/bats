"""
Configs for walker experiments.
"""
from collections import OrderedDict
from copy import deepcopy

base_config = OrderedDict(
    epsilon_planning=0.2,
    epsilon_neighbors=1.3,
    num_cpus=84,
    stitching_chunk_size=50000,
    normalize_obs=True,
    ni=10,
)

# For any additional configurations, add them here.
WALKER_CONFIGS = OrderedDict()

WALKER_CONFIGS['walker-expert'] = deepcopy(base_config)
WALKER_CONFIGS['walker-expert']['env_name'] = 'walker2d-expert-v2'

WALKER_CONFIGS['walker-medium-expert'] = deepcopy(base_config)
WALKER_CONFIGS['walker-medium-expert']['env_name'] =\
    'walker2d-medium-expert-v2'

WALKER_CONFIGS['walker-random'] = deepcopy(base_config)
WALKER_CONFIGS['walker-random']['env_name'] = 'walker2d-random-v2'

WALKER_CONFIGS['walker-mixed'] = deepcopy(base_config)
WALKER_CONFIGS['walker-mixed']['env_name'] =\
    'walker2d-medium-replay-v2'
WALKER_CONFIGS['walker-mixed']['num_stitching_iters'] = 2

WALKER_CONFIGS['walker-medium'] = deepcopy(base_config)
WALKER_CONFIGS['walker-medium']['env_name'] = 'walker2d-medium-v2'
