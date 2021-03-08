"""
Configs for hopper experiments.
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
HOPPER_CONFIGS = OrderedDict()

HOPPER_CONFIGS['hopper-expert'] = deepcopy(base_config)
HOPPER_CONFIGS['hopper-expert']['env'] = 'hopper-expert-v2'

HOPPER_CONFIGS['hopper-medium-expert'] = deepcopy(base_config)
HOPPER_CONFIGS['hopper-medium-expert']['env'] =\
    'hopper-medium-expert-v2'

HOPPER_CONFIGS['hopper-random'] = deepcopy(base_config)
HOPPER_CONFIGS['hopper-random']['env'] = 'hopper-random-v2'

HOPPER_CONFIGS['hopper-mixed'] = deepcopy(base_config)
HOPPER_CONFIGS['hopper-mixed']['env'] =\
    'hopper-medium-replay-v2'

HOPPER_CONFIGS['hopper-medium'] = deepcopy(base_config)
HOPPER_CONFIGS['hopper-medium']['env'] = 'hopper-medium-v2'
