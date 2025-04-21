#!/bin/bash

# Submit this script with: sbatch batchTrain.sh

#SBATCH --ntasks 1           # number of tasks
#SBATCH --cpus-per-task 4   # number of cpu cores per task
#SBATCH --time 12:00:00      # walltime
#SBATCH --mem 60gb           # amount of memory per CPU core (Memory per Task / Cores per Task)
#SBATCH --nodes 1            # number of nodes
#SBATCH --gpus-per-task v100:1 # gpu model and amount requested
#SBATCH --job-name "DeRainDrop_CleanNS" # job name
# Created with the RCD Docs Job Builder
#
# Visit the following link to edit this job:
# https://docs.rcd.clemson.edu/palmetto/job_management/job_builder/?num_cores=10&num_mem=40gb&use_gpus=yes&gpu_model=a100&walltime=12%3A00%3A00&job_name=testTrain

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

cd /scratch/cdauche/DeRaindrop/DeRaindrop #Path to CRN repo

#Load Modules
module load anaconda3
module load cuda/11.8.0

source activate DeRain2
echo "Starting"
srun python predict.py --mode demo 
echo "Draining Complete"
