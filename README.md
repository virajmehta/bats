# bats
best-action trajectory stitching

## Install instructions
1. Install the conda environment included with environment.yml.
2. Install d4rl with pip.
3. Run 
```
pip install -e .
mkdir experiments
```

## Running Experiments

### Training dynamics and bisimulation models.
To train dynamics:
```
python scripts/train_d4rl_dynamics.py --save_dir <save_dir> --env <env_name>
```

To train a bisimulation model:
```
python scripts/train_d4rl_bisiumulation.py --save_dir <save_dir> --env <env_name>

```

### Running BATS.

* For mazes:
```
python run_maze.py <experiment_name> --config maze-<maze_type> --num_cpus <num_cpus> --load_model <path_to_model>
```

* For mujoco + mountaincar:
```
python run.py <experiment_name> --config <task>-mixed --num_cpus <num_cpus> --load_model <path_to_model>
```

### Penalizing graph.

To add penalty to a previously stitched graph run the following:
```
python scripts/add_penalty_to_graph.py --env <env_name> --graph_dir <path_to_graph_dir> --save_dir <path_to_save_dir> --planerr_coef <float> --epsilon_planning <float>
```
Here:
* planerr_coef is the coeficient to multiply on the planning error distance which will be subtracted from the reward as penalty.
* epsilon_planning is the threshold of planning error to allow for any stitch.

### Running behavior cloning.

To run behavior cloning run:
```
python scripts/boltclone_graph.py --env <env_name> --graph_dir <path_to_graph_dir> --save_dir <path_to_save_dir> --unpenalized_rewards --return_threshold <threshold> --all_starts_once
```
