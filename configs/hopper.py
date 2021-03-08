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
HOPPER_CONFIGS['hopper-expert']['env_name'] = 'hopper-expert-v2'

HOPPER_CONFIGS['hopper-medium-expert'] = deepcopy(base_config)
HOPPER_CONFIGS['hopper-medium-expert']['env_name'] =\
    'hopper-medium-expert-v2'

HOPPER_CONFIGS['hopper-random'] = deepcopy(base_config)
HOPPER_CONFIGS['hopper-random']['env_name'] = 'hopper-random-v2'

HOPPER_CONFIGS['hopper-mixed'] = deepcopy(base_config)
HOPPER_CONFIGS['hopper-mixed']['env_name'] =\
    'hopper-medium-replay-v2'

HOPPER_CONFIGS['hopper-medium'] = deepcopy(base_config)
HOPPER_CONFIGS['hopper-medium']['env_name'] = 'hopper-medium-v2'
