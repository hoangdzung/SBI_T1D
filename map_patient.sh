#!/bin/bash
#SBATCH --job-name="replay"
#SBATCH --time=1-00:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=out_%x_%j.txt
#SBATCH -e err_%x_%j.txt
#SBATCH --partition=epyc2,bdw
#SBATCH --qos=job_cpu

module load Anaconda3
module load Workspace_Home
eval "$(conda shell.bash hook)"
conda activate my_mpi_env_new

python map_patient.py --method map --test_data $1  --save_folder map_simglucose