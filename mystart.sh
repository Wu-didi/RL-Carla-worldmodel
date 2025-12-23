unset PYTHONPATH
# conda activate carla13
export PYTHONPATH=$PYTHONPATH:/home/wudi/carla/PythonAPI
export PYTHONPATH=$PYTHONPATH:/home/wudi/carla/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/home/wudi/carla/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
export CARLA_ROOT=/home/wudi/carla




# train 
python RL/train_nocrash.py \
  --run_name nocrash_train \
  --nocrash_strict \
  --town Town01 \
  --nocrash_weather_group training \
  --nocrash_scenarios empty,regular,dense \
  --network Attention_SAC \
  --max_episodes 3000 \
  --save_interval_episodes 50