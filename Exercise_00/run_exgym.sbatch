#!/bin/bash

#SBATCH --job-name=SDC_Ex00_rh

#SBATCH --cpus-per-task=4

#SBATCH --partition=test

#SBATCH --mem-per-cpu=3G

#SBATCH --gres=gpu:1

#SBATCH --time=5:00

#SBATCH --error=job.%J.err

#SBATCH --output=job.%J.out

#SBATCH --mail-type=ALL

#SBATCH --mail-user=robert.herzig@student.uni-tuebingen.de

singularity exec ~/sdc_gym.simg python3 exercise_gym.py

echo FINISHED

