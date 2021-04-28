"""
Configs for halfcheetah experiments.
"""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

base_config = OrderedDict(
    epsilon_planning=0.4,
    epsilon_neighbors=1.3,
    num_cpus=80,
    stitching_chunk_size=50000,
    normalize_obs=True,
    num_stitching_iterations=20,
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
HALFCHEETAH_CONFIGS['halfcheetah-mixed']['load_bisim_model'] = Path("/zfsauton/project/public/ichar/models/bisimulation/halfcheetah")  # NOQA
HALFCHEETAH_CONFIGS['halfcheetah-mixed']['use_bisimulation'] = True
HALFCHEETAH_CONFIGS['halfcheetah-mixed']['penalize_stitches'] = True
HALFCHEETAH_CONFIGS['halfcheetah-mixed']['num_stitching_iterations'] = 10
HALFCHEETAH_CONFIGS['halfcheetah-mixed']['bc_every_iter'] = True
HALFCHEETAH_CONFIGS['halfcheetah-mixed']['epsilon_neighbors'] = 0.25


HALFCHEETAH_CONFIGS['halfcheetah-medium'] = deepcopy(base_config)
HALFCHEETAH_CONFIGS['halfcheetah-medium']['env_name'] = 'halfcheetah-medium-v2'
