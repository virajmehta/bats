"""
Configs for hopper experiments.
"""
from collections import OrderedDict
from copy import deepcopy

base_config = OrderedDict(
    epsilon_planning=0.07,
    epsilon_neighbors=0.3,
    num_cpus=84,
    stitching_chunk_size=50000,
)

# For any additional configurations, add them here.
HOPPER_CONFIGS = OrderedDict()

HOPPER_CONFIGS['hopper-expert'] = deepcopy(base_config)
HOPPER_CONFIGS['hopper-expert']['env_name'] = 'hopper-expert-v2'

HOPPER_CONFIGS['hopper-medexp'] = deepcopy(base_config)
HOPPER_CONFIGS['hopper-medexp']['env_name'] =\
    'hopper-medium-expert-v2'

HOPPER_CONFIGS['hopper-random'] = deepcopy(base_config)
HOPPER_CONFIGS['hopper-random']['env_name'] = 'hopper-random-v2'

HOPPER_CONFIGS['hopper-mixed'] = deepcopy(base_config)
HOPPER_CONFIGS['hopper-mixed']['env_name'] =\
    'hopper-medium-replay-v2'
HOPPER_CONFIGS['hopper-mixed']['load_model'] = 'experiments/hopper_medium_replay_bolt'

HOPPER_CONFIGS['hopper-medium'] = deepcopy(base_config)
HOPPER_CONFIGS['hopper-medium']['env_name'] = 'hopper-medium-v2'

to_add = OrderedDict()
for k, v in HOPPER_CONFIGS.items():
    task_type = k[k.index('-') + 1:]
    config = deepcopy(v)
    config['use_all_planning_itrs'] = True
    config['continue_after_no_advantage'] = True
    config['num_stitching_iters'] = 25
    # For mixed dataset edge distance = 0.726 +- 0.632
    config['epsilon_neighbors'] = 0.3
    config['planning_quantile'] = 0.4
    config['epsilon_planning'] = 10
    config['load_model'] = ('zfsauton/project/public/ichar/'
                            'd4rl_models/hopper/hp_%s' % task_type)
    to_add[k + '-tune'] = config
HOPPER_CONFIGS.update(to_add)
