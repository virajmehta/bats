all_args=("$@")
first_arg=$1
rest_args=("${all_args[@]:1}")
echo "sbatch psc_run.sbatch ${first_arg}_s0" "${rest_args[@]}"
# cp overlay.img "overlays/${first_arg}_s0.img"
./container/make_overlay.sh "overlays/${first_arg}_s0.img"
sbatch psc_run.sbatch "${first_arg}_s0" "${rest_args[@]}" --runseed 0
echo "sbatch psc_run.sbatch ${first_arg}_s1" "${rest_args[@]}"
# cp overlay.img "overlays/${first_arg}_s1.img"
./container/make_overlay.sh "overlays/${first_arg}_s1.img"
sbatch psc_run.sbatch "${first_arg}_s1" "${rest_args[@]}" --runseed 1
echo "sbatch psc_run.sbatch ${first_arg}_s2" "${rest_args[@]}"
# cp overlay.img "overlays/${first_arg}_s2.img"
./container/make_overlay.sh "overlays/${first_arg}_s2.img"
sbatch psc_run.sbatch "${first_arg}_s2" "${rest_args[@]}" --runseed 2
