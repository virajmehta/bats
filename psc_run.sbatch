#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM
#SBATCH -t 24:00:00
# #SBATCH --EGRESS
cp overlay.img overlays/$1.img
singularity exec --overlay overlays/$1.img --nv batcave_latest.sif python run.py $@
