"""
Configs for fusion experiments.
"""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

FUSION_CONFIG = OrderedDict(
    epsilon_planning=1,
    num_cpus=5,
    stitching_chunk_size=500,
    normalize_obs=True,
    num_stitching_iterations=25,
    k_neighbors=25,
    cb_plan=True,
    offline_dataset_path='/zfsauton/project/public/ichar/FusionData/bats_data/qset.hdf5',
    verbose=True,
    max_stitch_length=1,
    bc_every_iter=False,
    env_name=None,
)
