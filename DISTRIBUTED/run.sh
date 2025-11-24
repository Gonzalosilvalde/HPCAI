#!/bin/bash
#SBATCH --job-name=bert_test
#SBATCH --output=bert_test_a100.out
#SBATCH --error=bert_test_a100.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --ntasks=1

source ./.env
module load cesga/2025

cd $PROJECT_ROUTE
source ./venv/bin/activate

mkdir runs 
mkdir runs/profile

echo "Starting job..."
python main.py
