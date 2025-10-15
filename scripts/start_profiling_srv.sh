#!/bin/bash

module load cesga/2025
echo "Starting profiling server..."
tensorboard --logdir ./profile --port 6006
