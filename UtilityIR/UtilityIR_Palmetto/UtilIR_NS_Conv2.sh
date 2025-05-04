#!/bin/bash

# Submit this script with: sbatch batchTrain.sh

#SBATCH --ntasks 1           # number of tasks
#SBATCH --cpus-per-task 12   # number of cpu cores per task
#SBATCH --time 8:00:00      # walltime
#SBATCH --mem 48gb           # amount of memory per CPU core (Memory per Task / Cores per Task)
#SBATCH --nodes 1            # number of nodes
#SBATCH --gpus-per-task a100:1 # gpu model and amount requested
#SBATCH --job-name "Utility_IR_CleanNS, EVAL" # job name

#SBATCH -C gpu_a100_80gb
# Created with the RCD Docs Job Builder
#

# Visit the following link to edit this job:
# https://docs.rcd.clemson.edu/palmetto/job_management/job_builder/?num_cores=10&num_mem=40gb&use_gpus=yes&gpu_model=a100&walltime=12%3A00%3A00&job_name=testTrain

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

cd /project/bli4/maps/CD_RC_SL_8200Project/8200_Final_Project/UtilityIR/UtilityIR_Palmetto #Utility IR Location

#Load Modules
module load anaconda3
module load cuda/11.8.0

#Setup for Utility IR Cleansing
source activate DeRain2

echo "Starting Cleansing, Model 2"
srun python test.py --out_dir results --testing NS_INFERENCE --model 2
echo "Draining Complete - Utility IR2"

#Change to CRN
cd /project/bli4/maps/CD_RC_SL_8200Project/8200_Final_Project/ #Path to CRN repo
source activate CRN

echo "Starting Utility IR Validation, model 2"
srun python ./exps/det/CRN_r50_256x704_128x128_4key_CDD_UtilityIR2.py --ckpt_path ./models/CRN_r50_256x704_128x128_4key.pth -e -b 1 --gpus 1

echo "Evaluation Complete"


