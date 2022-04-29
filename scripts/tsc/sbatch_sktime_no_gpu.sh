#!/bin/bash
#

#SBATCH --job-name=tsc-test
#SBATCH --ntasks=1
#SBATCH -M ukko
##SBATCH -p test
#SBATCH -c 1

# Memory request per core
##SBATCH --mem-per-cpu=2048   

# Time limit
##SBATCH -t 1-00:00:00
#SBATCH -t 01:00:00
##SBATCH -t 10:00

#SBATCH --chdir=/wrk/users/hajaalin/output/TSC
#SBATCH --output=tsc-%j.out


# Load the Conda module
module use /proj/hajaalin/LMUModules/
module --ignore-cache load Miniconda3/4.11.0
source activate tsc

which python

cmd="srun python /proj/hajaalin/Projects/n_track_ML/scripts/tsc/cv_sktime.py --config_paths /proj/hajaalin/Projects/n_track_ML/scripts/tsc/paths.yml --config_tsc /proj/hajaalin/Projects/n_track_ML/scripts/tsc/config.yml"
echo ${cmd}
${cmd}




