#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="train"
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=300G
#SBATCH --output=slurm_outputs/train_%j.out
#SBATCH --cpus-per-task=8

# ------------------------
# Debug info
# ------------------------
echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "DATE:                $(date)"
echo "---------------------------------------------"

# ------------------------
# Activate environment
# ------------------------
conda activate .venv
export TOKENIZERS_PARALLELISM=false

# ------------------------
# Run generator
# ------------------------
python /u/scandussio/stlenc/train.py 
