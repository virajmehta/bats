"""
Configs for walker experiments.
"""
from collections import OrderedDict
from copy import deepcopy

base_config = OrderedDict(
    ep=0.1,
    en=0.2,
    ncpu=128,
    scs=50000,
    nvi=250,
    nvie=1000,
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

WALKER_CONFIGS['walker-medium'] = deepcopy(base_config)
WALKER_CONFIGS['walker-medium']['env'] = 'walker2d-medium-v2'
