"""
Configs for maze experiments.
"""
from collections import OrderedDict
from copy import deepcopy

base_config = OrderedDict(
    epsilon_planning=10,
    k_neighbors=25,
    num_cpus=30,
    stitching_chunk_size=20000,
    bc_epochs=25,
    od_wait=-1,
    top_percent_starts=None,
    temperature=0.1,
    normalize_obs=True,
)

# For any additional configurations, add them here.
ANTMAZE_CONFIGS = OrderedDict()

ANTMAZE_CONFIGS['antmaze-umaze'] = deepcopy(base_config)
ANTMAZE_CONFIGS['antmaze-umaze']['env_name'] = 'maze2d-umaze-v1'

# MAZE_CONFIGS['maze-medium'] = deepcopy(base_config)
# MAZE_CONFIGS['maze-medium']['env_name'] = 'maze2d-medium-v1'
# MAZE_CONFIGS['maze-medium']['planning_quantile'] = 0.8
# MAZE_CONFIGS['maze-medium']['epsilon_planning'] = 0.425
# MAZE_CONFIGS['maze-medium']['num_stitching_iters'] = 10
# MAZE_CONFIGS['maze-medium']['max_stitch_length'] = 1
# MAZE_CONFIGS['maze-medium']['load_model'] = ('/zfsauton/project/public/ichar/'
#                             'd4rl_models/mazes/mediummaze')
# MAZE_CONFIGS['maze-medium']['verbose'] = True
# 
# MAZE_CONFIGS['maze-large'] = deepcopy(base_config)
# MAZE_CONFIGS['maze-large']['env_name'] = 'maze2d-large-v1'
# MAZE_CONFIGS['maze-large']['epsilon_neighbors'] = 0.15

# to_add = OrderedDict()
# for k, v in MAZE_CONFIGS.items():
#     task_type = k[k.index('-') + 1:]
#     if 'maze' not in task_type:
#         task_type = task_type + 'maze'
#     config = deepcopy(v)
#     config['use_all_planning_itrs'] = True
#     config['continue_after_no_advantage'] = True
#     config['stitching_chunk_size'] = 10000
#     config['num_stitching_iters'] = 50
#     # For umaze dataset edge distance = 0.11 +- 0.03
#     config['planning_quantile'] = 0.4
#     config['epsilon_planning'] = 1.5
#     config['load_model'] = ('/zfsauton/project/public/ichar/'
#                             'd4rl_models/mazes/%s' % task_type)
#     config['verbose'] = True
#     config['k_neighbors'] = 25
#     config['max_stitch_length'] = 5
#     to_add[k + '-tune'] = config
# ANTMAZE_CONFIGS.update(to_add)
