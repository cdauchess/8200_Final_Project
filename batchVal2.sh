#!/bin/bash

# Submit this script with: sbatch FILENAME

#SBATCH --ntasks 1           # number of tasks
#SBATCH --cpus-per-task 12   # number of cpu cores per task
#SBATCH --time 2:00:00      # walltime
#SBATCH --mem 48gb           # amount of memory per CPU core (Memory per Task / Cores per Task)
#SBATCH --nodes 1            # number of nodes
#SBATCH --gpus-per-task a100:1 # gpu model and amount requested
#SBATCH --job-name "CRN_VAL-3modes" # job name
# Created with the RCD Docs Job Builder
#
# Visit the following link to edit this job:
# https://docs.rcd.clemson.edu/palmetto/job_management/job_builder/?num_cores=10&num_mem=40gb&use_gpus=yes&gpu_model=a100&walltime=12%3A00%3A00&job_name=testTrain

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

cd /project/bli4/maps/CD_RC_SL_8200Project/8200_Final_Project/ #Path to CRN repo

#Load Modules
module load anaconda3
module load cuda/11.8.0

source activate CRN


echo "Starting DeRaindrop Validation"
srun python ./exps/det/CRN_r50_256x704_128x128_4key_CDD_DRD.py --ckpt_path ./models/CRN_r50_256x704_128x128_4key.pth -e -b 1 --gpus 1

echo "Starting Utility IR Validation"
srun python ./exps/det/CRN_r50_256x704_128x128_4key_CDD_UtilityIR.py --ckpt_path ./models/CRN_r50_256x704_128x128_4key.pth -e -b 1 --gpus 1

echo "Evaluation Complete"
