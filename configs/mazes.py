"""
Configs for maze experiments.
"""
from collections import OrderedDict
from copy import deepcopy

base_config = OrderedDict(
    epsilon_planning=0.1,
    epsilon_neighbors=0.225,
    num_cpus=80,
    stitching_chunk_size=50000,
)

# For any additional configurations, add them here.
MAZE_CONFIGS = OrderedDict()

MAZE_CONFIGS['maze-umaze'] = deepcopy(base_config)
MAZE_CONFIGS['maze-umaze']['env_name'] = 'maze2d-umaze-v1'

MAZE_CONFIGS['maze-medium'] = deepcopy(base_config)
MAZE_CONFIGS['maze-medium']['env_name'] = 'maze2d-medium-v1'

MAZE_CONFIGS['maze-large'] = deepcopy(base_config)
MAZE_CONFIGS['maze-large']['env_name'] = 'maze2d-large-v1'

to_add = OrderedDict()
for k, v in MAZE_CONFIGS.items():
    task_type = k[k.index('-') + 1:]
    if 'maze' not in task_type:
        task_type = task_type + 'maze'
    config = deepcopy(v)
    config['use_all_planning_itrs'] = True
    config['continue_after_no_advantage'] = True
    config['num_stitching_iters'] = 25
    # For umaze dataset edge distance = 0.11 +- 0.03
    config['planning_quantile'] = 0.4
    config['epsilon_planning'] = 1.5
    config['load_model'] = ('zfsauton/project/public/ichar/'
                            'd4rl_models/mazes/%s' % task_type)
    to_add[k + '-tune'] = config
MAZE_CONFIGS.update(to_add)
