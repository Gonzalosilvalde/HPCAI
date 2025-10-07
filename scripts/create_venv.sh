#!/bin/bash
source ./.env

cd $PROJECT_ROUTE

module load cesga/2025

rm -rf venv

python -m venv venv

source ./venv/bin/activate

# Los he sacado del ejemplo del CFR24
echo "Installing modules..."

pip install -r requirements.txt
