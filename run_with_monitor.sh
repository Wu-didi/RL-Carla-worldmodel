#!/bin/bash

# ============================================
# CARLA RL Training with Memory Monitor
# Auto-restart on OOM
# ============================================

CARLA_ROOT="/home/wudi/carla"

# Setup PYTHONPATH
unset PYTHONPATH
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export CARLA_ROOT=${CARLA_ROOT}

# Training parameters
RUN_NAME="nocrash_train"
MAX_RETRIES=10
RETRY_COUNT=0

echo "Starting training with auto-restart on failure..."

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo ""
    echo "=========================================="
    echo "Attempt $((RETRY_COUNT + 1)) / $MAX_RETRIES"
    echo "=========================================="

    # Run training
    python RL/train_nocrash.py \
        --run_name $RUN_NAME \
        --nocrash_strict \
        --town Town01 \
        --nocrash_weather_group training \
        --nocrash_scenarios empty,regular,dense \
        --network Attention_SAC \
        --max_episodes 3000 \
        --save_interval_episodes 50 \
        --pretrained_agent_path "RL/runs/${RUN_NAME}/checkpoints/agent/last.pt" \
        --pretrained_world_model_path "RL/runs/${RUN_NAME}/checkpoints/world_model/last.pt"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully!"
        exit 0
    elif [ $EXIT_CODE -eq 137 ] || [ $EXIT_CODE -eq 9 ]; then
        # 137 = SIGKILL (OOM), 9 = killed
        echo "[WARN] Process killed (likely OOM). Restarting in 30 seconds..."
        echo "Consider reducing traffic density or increasing system RAM."
        RETRY_COUNT=$((RETRY_COUNT + 1))
        sleep 30
    else
        echo "[ERROR] Training failed with exit code $EXIT_CODE"
        RETRY_COUNT=$((RETRY_COUNT + 1))
        sleep 10
    fi
done

echo "[FATAL] Max retries ($MAX_RETRIES) exceeded. Check logs and system resources."
exit 1
