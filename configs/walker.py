"""
Configs for walker experiments.
"""
from collections import OrderedDict
from copy import deepcopy

base_config = OrderedDict(
    ep=0.2,
    en=1.3,
    ncpu=84,
    scs=50000,
    normalize_obs=True,
)

# For any additional configurations, add them here.
WALKER_CONFIGS = OrderedDict()

WALKER_CONFIGS['walker-expert'] = deepcopy(base_config)
WALKER_CONFIGS['walker-expert']['env'] = 'walker2d-expert-v2'

WALKER_CONFIGS['walker-medium-expert'] = deepcopy(base_config)
WALKER_CONFIGS['walker-medium-expert']['env'] =\
    'walker2d-medium-expert-v2'

WALKER_CONFIGS['walker-random'] = deepcopy(base_config)
WALKER_CONFIGS['walker-random']['env'] = 'walker2d-random-v2'

WALKER_CONFIGS['walker-mixed'] = deepcopy(base_config)
WALKER_CONFIGS['walker-mixed']['env'] =\
    'walker2d-medium-replay-v2'
WALKER_CONFIGS['walker-mixed']['num_stitching_iters'] = 2

WALKER_CONFIGS['walker-medium'] = deepcopy(base_config)
WALKER_CONFIGS['walker-medium']['env'] = 'walker2d-medium-v2'
