from collections import OrderedDict

from configs.halfcheetah import HALFCHEETAH_CONFIGS
from configs.hopper import HOPPER_CONFIGS
from configs.mazes import MAZE_CONFIGS
from configs.walker import WALKER_CONFIGS
from configs.mountaincar import MOUNTAINCAR_CONFIGS
from configs.pendulum import PENDULUM_CONFIGS

CONFIGS = OrderedDict()
CONFIGS.update(HALFCHEETAH_CONFIGS)
CONFIGS.update(HOPPER_CONFIGS)
CONFIGS.update(MAZE_CONFIGS)
CONFIGS.update(WALKER_CONFIGS)
CONFIGS.update(PENDULUM_CONFIGS)
CONFIGS.update(MOUNTAINCAR_CONFIGS)
