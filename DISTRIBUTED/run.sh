#!/bin/bash
#SBATCH --job-name=bert_test_distributed
#SBATCH --output=bert_test_a100_distributed.out
#SBATCH --error=bert_test_a100_distributed.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=2
#SBATCH --mem=32G
#SBATCH --nodes=2

source ./.env
module load cesga/2025

cd $PROJECT_ROUTE
source ./venv/bin/activate

mkdir runs 
mkdir runs/profile

echo "Starting job..."
srun python main.py