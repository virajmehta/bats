"""
Configs for walker experiments.
"""
from collections import OrderedDict
from copy import deepcopy

base_config = OrderedDict(
    epsilon_planning=0.2,
    epsilon_neighbors=1.3,
    num_cpus=84,
    stitching_chunk_size=50000,
    normalize_obs=True,
    ni=10,
)

# For any additional configurations, add them here.
WALKER_CONFIGS = OrderedDict()

WALKER_CONFIGS['walker-expert'] = deepcopy(base_config)
WALKER_CONFIGS['walker-expert']['env_name'] = 'walker2d-expert-v2'

WALKER_CONFIGS['walker-medexp'] = deepcopy(base_config)
WALKER_CONFIGS['walker-medexp']['env_name'] =\
    'walker2d-medium-expert-v2'

WALKER_CONFIGS['walker-random'] = deepcopy(base_config)
WALKER_CONFIGS['walker-random']['env_name'] = 'walker2d-random-v2'

WALKER_CONFIGS['walker-mixed'] = deepcopy(base_config)
WALKER_CONFIGS['walker-mixed']['env_name'] =\
    'walker2d-medium-replay-v2'
WALKER_CONFIGS['walker-mixed']['num_stitching_iters'] = 2

WALKER_CONFIGS['walker-medium'] = deepcopy(base_config)
WALKER_CONFIGS['walker-medium']['env_name'] = 'walker2d-medium-v2'

to_add = OrderedDict()
for k, v in WALKER_CONFIGS.items():
    task_type = k[k.index('-') + 1:]
    config = deepcopy(v)
    config['use_all_planning_itrs'] = True
    config['continue_after_no_advantage'] = True
    config['num_stitching_iters'] = 25
    # For mixed dataset edge distance = 1.83 +- 1.34
    config['epsilon_neighbors'] = 0.2
    config['planning_quantile'] = 0.4
    config['epsilon_planning'] = 10
    config['load_model'] = ('/zfsauton/project/public/ichar/'
                            'd4rl_models/walker/wk_%s' % task_type)
    to_add[k + '-tune'] = config
WALKER_CONFIGS.update(to_add)
