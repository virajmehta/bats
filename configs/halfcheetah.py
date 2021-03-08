"""
Configs for halfcheetah experiments.
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
HALFCHEETAH_CONFIGS = OrderedDict()

HALFCHEETAH_CONFIGS['halfcheetah-expert'] = deepcopy(base_config)
HALFCHEETAH_CONFIGS['halfcheetah-expert']['env'] = 'halfcheetah-expert-v2'

HALFCHEETAH_CONFIGS['halfcheetah-medium-expert'] = deepcopy(base_config)
HALFCHEETAH_CONFIGS['halfcheetah-medium-expert']['env'] =\
    'halfcheetah-medium-expert-v2'

HALFCHEETAH_CONFIGS['halfcheetah-random'] = deepcopy(base_config)
HALFCHEETAH_CONFIGS['halfcheetah-random']['env'] = 'halfcheetah-random-v2'

HALFCHEETAH_CONFIGS['halfcheetah-mixed'] = deepcopy(base_config)
HALFCHEETAH_CONFIGS['halfcheetah-mixed']['env'] =\
    'halfcheetah-medium-replay-v2'

HALFCHEETAH_CONFIGS['halfcheetah-medium'] = deepcopy(base_config)
HALFCHEETAH_CONFIGS['halfcheetah-medium']['env'] = 'halfcheetah-medium-v2'
