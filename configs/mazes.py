"""
Configs for maze experiments.
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
MAZE_CONFIGS = OrderedDict()

MAZE_CONFIGS['maze-umaze'] = deepcopy(base_config)
MAZE_CONFIGS['maze-umaze']['env_name'] = 'maze2d-umaze-v1'

MAZE_CONFIGS['maze-medium'] = deepcopy(base_config)
MAZE_CONFIGS['maze-medium']['env_name'] = 'maze2d-medium-v1'

MAZE_CONFIGS['maze-large'] = deepcopy(base_config)
MAZE_CONFIGS['maze-large']['env_name'] = 'maze2d-large-v1'
