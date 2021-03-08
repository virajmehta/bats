"""
Baselines on the D4RL tasks to compare to.
"""
from collections import OrderedDict

import d4rl

BASELINES = OrderedDict()


def get_raw_score(task_name, score):
    min_score = d4rl.infos.REF_MIN_SCORE[task_name]
    max_score = d4rl.infos.REF_MAX_SCORE[task_name]
    print(min_score, max_score)
    return (score / 100 * (max_score - min_score)) + min_score

BASELINES['halfcheetah-random'] = OrderedDict(
    SAC=30.5,
    BC=2.1,
    BEAR=25.5,
    BRACp=23.5,
    BRACv=28.1,
    CQL=35.4,
    MOPO=35.4,
    COMBO=38.8,
)

BASELINES['hopper-random'] = OrderedDict(
    SAC=11.3,
    BC=9.8,
    BEAR=9.5,
    BRACp=11.1,
    BRACv=12.0,
    CQL=10.8,
    MOPO=11.7,
    COMBO=17.9
)

BASELINES['walker2d-random'] = OrderedDict(
    SAC=4.1,
    BC=1.6,
    BEAR=6.7,
    BRACp=0.8,
    BRACv=0.5,
    CQL=7.0,
    MOPO=13.6,
    COMBO=7.0,
)

BASELINES['halfcheetah-medium'] = OrderedDict(
    SAC=-4.3,
    BC=36.1,
    BEAR=38.6,
    BRACp=44.0,
    BRACv=45.5,
    CQL=44.4,
    MOPO=42.3,
    COMBO=54.2,
)

BASELINES['hopper-medium'] = OrderedDict(
    SAC=0.8,
    BC=29.0,
    BEAR=47.6,
    BRACp=31.2,
    BRACv=32.3,
    CQL=86.6,
    MOPO=28.0,
    COMBO=94.9,
)

BASELINES['walker2d-medium'] = OrderedDict(
    SAC=0.9,
    BC=6.6,
    BEAR=33.2,
    BRACp=72.7,
    BRACv=81.3,
    CQL=74.5,
    MOPO=17.8,
    COMBO=75.5,
)

BASELINES['halfcheetah-expert'] = OrderedDict(
    SAC=-1.9,
    BC=107.0,
    BEAR=108.2,
    BRACp=3.8,
    BRACv=-1.1,
    CQL=104.8,
)

BASELINES['hopper-expert'] = OrderedDict(
    SAC=0.7,
    BC=109.0,
    BEAR=110.3,
    BRACp=6.6,
    BRACv=3.7,
    CQL=109.9,
)

BASELINES['walker2d-expert'] = OrderedDict(
    SAC=-0.3,
    BC=125.7,
    BEAR=106.1,
    BRACp=-0.2,
    BRACv=0.0,
    CQL=153.9,
)

BASELINES['halfcheetah-medium-expert'] = OrderedDict(
    SAC=1.8,
    BC=35.8,
    BEAR=51.7,
    BRACp=43.8,
    BRACv=45.3,
    CQL=62.4,
    MOPO=63.3,
    COMBO=90.0,
)

BASELINES['hopper-medium-expert'] = OrderedDict(
    SAC=1.6,
    BC=111.9,
    BEAR=4.0,
    BRACp=1.1,
    BRACv=0.8,
    CQL=110.0,
    MOPO=23.7,
    COMBO=111.1,
)

BASELINES['walker2d-medium-expert'] = OrderedDict(
    SAC=1.9,
    BC=11.3,
    BEAR=10.8,
    BRACp=-0.3,
    BRACv=0.9,
    CQL=98.7,
    MOPO=44.6,
    COMBO=96.1,
)

BASELINES['halfcheetah-random-expert'] = OrderedDict(
    SAC=53.0,
    BC=1.3,
    BEAR=24.6,
    BRACp=30.2,
    BRACv=2.2,
    CQL=92.5,
)

BASELINES['hopper-random-expert'] = OrderedDict(
    SAC=5.6,
    BC=10.1,
    BEAR=10.1,
    BRACp=5.8,
    BRACv=11.1,
    CQL=110.5,
)

BASELINES['walker2d-random-expert'] = OrderedDict(
    SAC=0.8,
    BC=0.7,
    BEAR=1.9,
    BRACp=0.2,
    BRACv=2.7,
    CQL=91.1,
)

BASELINES['halfcheetah-mixed'] = OrderedDict(
    SAC=-2.4,
    BC=38.4,
    BEAR=36.2,
    BRACp=45.6,
    BRACv=45.9,
    CQL=46.2,
    MOPO=53.1,
    COMBO=55.1,
)
BASELINES['halfcheetah-medium-replay'] = BASELINES['halfcheetah-mixed']

BASELINES['hopper-mixed'] = OrderedDict(
    SAC=3.5,
    BC=11.8,
    BEAR=25.3,
    BRACp=0.7,
    BRACv=0.8,
    CQL=48.6,
    MOPO=67.5,
    COMBO=73.1,
)
BASELINES['hopper-medium-replay'] = BASELINES['hopper-mixed']

BASELINES['walker2d-mixed'] = OrderedDict(
    SAC=1.9,
    BC=11.3,
    BEAR=10.8,
    BRACp=-0.3,
    BRACv=0.9,
    CQL=26.7,
    MOPO=39.0,
    COMBO=56.0,
)
BASELINES['walker2d-medium-replay'] = BASELINES['walker2d-mixed']

BASELINES['maze2d-umaze'] = OrderedDict(
    SAC_ONLINE=62.7,
    BC=3.8,
    SAC=88.2,
    BEAR=3.4,
    BRACp=4.7,
    BRACv=-16.0,
    AWR=1.0,
    BCQ=12.89,
    aDICE=-15.7,
    CQL=5.7,
)

BASELINES['maze2d-medium'] = OrderedDict(
    SAC_ONLINE=21.3,
    BC=30.3,
    SAC=26.1,
    BEAR=29.0,
    BRACp=32.4,
    BRACv=-33.8,
    AWR=7.6,
    BCQ=8.3,
    aDICE=10.0,
    CQL=5.0,
)

BASELINES['maze2d-large'] = OrderedDict(
    SAC_ONLINE=2.7,
    BC=5.0,
    SAC=-1.9,
    BEAR=4.6,
    BRACp=10.4,
    BRACv=40.6,
    AWR=23.7,
    BCQ=6.2,
    aDICE=-0.1,
    CQL=12.5,
)
