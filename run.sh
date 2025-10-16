#!/bin/bash
#SBATCH --job-name=bert_test
#SBATCH --output=bert_test.out
#SBATCH --error=bert_test.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --ntasks=1

##SBATCH --partition=gpu

source ./.env
module load cesga/2025

cd $PROJECT_ROUTE
source ./venv/bin/activate

mkdir runs profile

echo "Starting job..."
python main.py
