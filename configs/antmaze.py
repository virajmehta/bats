"""
Configs for maze experiments.
"""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

base_config = OrderedDict(
    epsilon_planning=10,
    # k_neighbors=25,
    num_cpus=90,
    stitching_chunk_size=8000,
    bc_epochs=25,
    # use_bisimulation=True,
    # penalize_stitches=True,
    # bc_every_iter=True,
    top_percent_starts=None,
    temperature=0.1,
    normalize_obs=True,
)

# For any additional configurations, add them here.
ANTMAZE_CONFIGS = OrderedDict()

ANTMAZE_CONFIGS['antmaze-umaze'] = deepcopy(base_config)
ANTMAZE_CONFIGS['antmaze-umaze']['env_name'] = 'antmaze-umaze-v0'
ANTMAZE_CONFIGS['antmaze-umaze']['load_model'] = Path('~/bats/models/antmaze').expanduser()

ANTMAZE_CONFIGS['antmaze-medium'] = deepcopy(base_config)
ANTMAZE_CONFIGS['antmaze-medium']['env_name'] = 'antmaze2d-medium-v1'
ANTMAZE_CONFIGS['antmaze-medium']['planning_quantile'] = 0.8
ANTMAZE_CONFIGS['antmaze-medium']['epsilon_planning'] = 0.425
ANTMAZE_CONFIGS['antmaze-medium']['num_stitching_iters'] = 10
ANTMAZE_CONFIGS['antmaze-medium']['max_stitch_length'] = 1
ANTMAZE_CONFIGS['antmaze-medium']['load_model'] = ('/zfsauton/project/public/ichar/'
                            'd4rl_models/mazes/mediummaze')
ANTMAZE_CONFIGS['antmaze-medium']['verbose'] = True

ANTMAZE_CONFIGS['antmaze-large'] = deepcopy(base_config)
ANTMAZE_CONFIGS['antmaze-large']['env_name'] = 'antmaze2d-large-v1'
ANTMAZE_CONFIGS['antmaze-large']['epsilon_neighbors'] = 0.15

to_add = OrderedDict()
for k, v in ANTMAZE_CONFIGS.items():
    task_type = k[k.index('-') + 1:]
    if 'antmaze' not in task_type:
        task_type = task_type + 'antmaze'
    config = deepcopy(v)
    config['use_all_planning_itrs'] = True
    config['continue_after_no_advantage'] = True
    config['num_stitching_iters'] = 25
    # For umaze dataset edge distance = 0.11 +- 0.03
    config['planning_quantile'] = 0.8
    config['epsilon_planning'] = 5
    # config['load_model'] = ('/zfsauton/project/public/ichar/'
                            # 'd4rl_models/mazes/%s' % task_type)
    config['verbose'] = True
    config['epsilon_neighbors'] = 0.1
    config['max_stitch_length'] = 5
    to_add[k + '-tune'] = config
ANTMAZE_CONFIGS.update(to_add)
