#!/bin/bash
#SBATCH --job-name=bert_test
#SBATCH --output=bert_test_cpu.out
#SBATCH --error=bert_test_cpu.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclusive
##SBATCH --gres=gpu:1

source ./.env
module load cesga/2025

cd $PROJECT_ROUTE
source ./venv/bin/activate

mkdir runs 
mkdir runs/profile

echo "Starting job..."
python main.py
