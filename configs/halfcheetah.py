"""
Configs for halfcheetah experiments.
"""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

base_config = OrderedDict(
    epsilon_planning=0.4,
    epsilon_neighbors=1.3,
    num_cpus=40,
    stitching_chunk_size=50000,
    normalize_obs=True,
    num_stitching_iters=20,
)

# For any additional configurations, add them here.
HALFCHEETAH_CONFIGS = OrderedDict()

HALFCHEETAH_CONFIGS['halfcheetah-expert'] = deepcopy(base_config)
HALFCHEETAH_CONFIGS['halfcheetah-expert']['env_name'] = 'halfcheetah-expert-v0'

HALFCHEETAH_CONFIGS['halfcheetah-medexp'] = deepcopy(base_config)
HALFCHEETAH_CONFIGS['halfcheetah-medexp']['env_name'] =\
    'halfcheetah-medium-expert-v0'

HALFCHEETAH_CONFIGS['halfcheetah-random'] = deepcopy(base_config)
HALFCHEETAH_CONFIGS['halfcheetah-random']['env_name'] = 'halfcheetah-random-v0'

HALFCHEETAH_CONFIGS['halfcheetah-mixed'] = deepcopy(base_config)
HALFCHEETAH_CONFIGS['halfcheetah-mixed']['env_name'] =\
    'halfcheetah-medium-replay-v0'
# HALFCHEETAH_CONFIGS['halfcheetah-mixed']['load_bisim_model'] = Path("/zfsauton/project/public/ichar/models/bisimulation/halfcheetah")  # NOQA
HALFCHEETAH_CONFIGS['halfcheetah-mixed']['use_bisimulation'] = False
HALFCHEETAH_CONFIGS['halfcheetah-mixed']['penalize_stitches'] = False
HALFCHEETAH_CONFIGS['halfcheetah-mixed']['bc_every_iter'] = False
HALFCHEETAH_CONFIGS['halfcheetah-mixed']['epsilon_neighbors'] = 0.25
HALFCHEETAH_CONFIGS['halfcheetah-mixed']['load_model'] = Path('~/bats/models/hcv0-mixed').expanduser()

HALFCHEETAH_CONFIGS['halfcheetah-medium'] = deepcopy(base_config)
HALFCHEETAH_CONFIGS['halfcheetah-medium']['env_name'] = 'halfcheetah-medium-v0'


to_add = OrderedDict()
for k, v in HALFCHEETAH_CONFIGS.items():
    task_type = k[k.index('-') + 1:]
    config = deepcopy(v)
    config['use_all_planning_itrs'] = True
    config['continue_after_no_advantage'] = True
    config['num_stitching_iters'] = 40
    config['stitching_chunk_size'] = 5000
    # For mixed dataset edge distance = 4.48265 +- 1.07
    # config['epsilon_neighbors'] = 1.3
    config['bc_every_iter'] = False
    config['planning_quantile'] = 0.4
    config['epsilon_planning'] = 10
    config['verbose'] = True
    config['k_neighbors'] = 25
    config['max_stitch_length'] = 5
    to_add[k + '-tune'] = config
HALFCHEETAH_CONFIGS.update(to_add)
