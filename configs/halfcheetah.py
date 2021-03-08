"""
Configs for halfcheetah experiments.
"""
from collections import OrderedDict
from copy import deepcopy

base_config = OrderedDict(
    ep=0.5,
    en=1.3,
    ncpu=84,
    scs=50000,
    nvi=250,
    nvie=1000,
    normalize_obs=True,
)

# For any additional configurations, add them here.
HALFCHEETAH_CONFIGS = OrderedDict()

HALFCHEETAH_CONFIGS['halfcheetah-expert'] = deepcopy(base_config)
HALFCHEETAH_CONFIGS['halfcheetah-expert']['env_name'] = 'halfcheetah-expert-v2'

HALFCHEETAH_CONFIGS['halfcheetah-medium-expert'] = deepcopy(base_config)
HALFCHEETAH_CONFIGS['halfcheetah-medium-expert']['env_name'] =\
    'halfcheetah-medium-expert-v2'

HALFCHEETAH_CONFIGS['halfcheetah-random'] = deepcopy(base_config)
HALFCHEETAH_CONFIGS['halfcheetah-random']['env_name'] = 'halfcheetah-random-v2'

HALFCHEETAH_CONFIGS['halfcheetah-mixed'] = deepcopy(base_config)
HALFCHEETAH_CONFIGS['halfcheetah-mixed']['env_name'] =\
    'halfcheetah-medium-replay-v2'

HALFCHEETAH_CONFIGS['halfcheetah-medium'] = deepcopy(base_config)
HALFCHEETAH_CONFIGS['halfcheetah-medium']['env_name'] = 'halfcheetah-medium-v2'
