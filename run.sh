#!/bin/bash
#SBATCH --job-name=bert_test
#SBATCH --output=bert_test.out
#SBATCH --error=bert_test.err
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --ntasks=1

##SBATCH --partition=gpu

module load cesga/2025

cd $STORE/HPC_TOOLS/HPCAI # Yo lo tengo organizado asi en el FT3
source ./venv/bin/activate

echo "Starting job..."
python main.py
