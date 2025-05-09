#!/bin/bash

# Submit this script with: sbatch batchTrain.sh

#SBATCH --ntasks 1           # number of tasks
#SBATCH --cpus-per-task 12   # number of cpu cores per task
#SBATCH --time 2:00:00      # walltime
#SBATCH --mem 64gb           # amount of memory per CPU core (Memory per Task / Cores per Task)
#SBATCH --nodes 1            # number of nodes
#SBATCH --gpus-per-task a100:2 # gpu model and amount requested
#SBATCH --job-name "Eval_2H" # job name

#SBATCH -C gpu_a100_80gb
# Created with the RCD Docs Job Builder
#
# Visit the following link to edit this job:
# https://docs.rcd.clemson.edu/palmetto/job_management/job_builder/?num_cores=10&num_mem=40gb&use_gpus=yes&gpu_model=a100&walltime=12%3A00%3A00&job_name=testTrain

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

cd /scratch/cdauche/CRN #Path to CRN repo

#Load Modules
module load anaconda3
module load cuda/11.8.0

source activate CRN

#Show some basic job info in the output log
echo "12 cores, 64 gb ram, 2x A100 80gb GPU"

echo "Starting"
srun python ./exps/det/CRN_r50_256x704_128x128_4key_CDD.py --ckpt_path ./outputs/det/CRN_r50_256x704_128x128_4key/lightning_logs/version_2560023/checkpoints/epoch=23-step=21096.ckpt -e -b 16 --gpus 2
echo "Evaluation Complete"