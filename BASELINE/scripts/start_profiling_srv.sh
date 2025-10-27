#!/bin/bash

module load cesga/2025
echo "Starting profiling server..."
tensorboard --logdir ./results --port 6006
