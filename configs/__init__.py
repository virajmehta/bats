from collections import OrderedDict

from configs.halfcheetah import HALFCHEETAH_CONFIGS
from configs.hopper import HOPPER_CONFIGS
from configs.mazes import MAZE_CONFIGS
from configs.walker import WALKER_CONFIGS

CONFIGS = OrderedDict()
CONFIGS.update(HALFCHEETAH_CONFIGS)
CONFIGS.update(HOPPER_CONFIGS)
CONFIGS.update(MAZE_CONFIGS)
CONFIGS.update(WALKER_CONFIGS)
