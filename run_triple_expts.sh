all_args=("$@")
first_arg=$1
rest_args=("${all_args[@]:2}")
sbatch psc_run.sbatch "${first_arg}_s0" "${rest_args[@]}"
sbatch psc_run.sbatch "${first_arg}_s1" "${rest_args[@]}"
sbatch psc_run.sbatch "${first_arg}_s2" "${rest_args[@]}"
