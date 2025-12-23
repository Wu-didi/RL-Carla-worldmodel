#!/bin/bash

# ============================================
# CARLA RL Training Startup Script
# ============================================

# Set your CARLA installation path here
CARLA_ROOT="/home/wudi/carla"

# Setup PYTHONPATH for CARLA
unset PYTHONPATH
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
# Add agents path for GlobalRoutePlanner (if available)
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export CARLA_ROOT=${CARLA_ROOT}

echo "CARLA_ROOT: ${CARLA_ROOT}"
echo "PYTHONPATH: ${PYTHONPATH}"

# Train NoCrash benchmark
python RL/train_nocrash.py \
  --run_name nocrash_train \
  --nocrash_strict \
  --town Town01 \
  --nocrash_weather_group training \
  --nocrash_scenarios empty,regular,dense \
  --network Attention_SAC \
  --max_episodes 3000 \
  --save_interval_episodes 50
