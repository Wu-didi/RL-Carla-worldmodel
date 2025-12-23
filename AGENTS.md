# Repository Guidelines

## Project Structure & Modules
- `carla_env/`: Environment wrappers; `carla_env.py` is the base CARLA Gym env, `nocrash_env.py` wraps it for NoCrash-style scenarios.
- `RL/`: Training and evaluation entrypoints plus agents and utilities.
  - `train.py`: Baseline SAC+world-model training.
  - `train_nocrash.py`, `eval_nocrash.py`: NoCrash-specific train/eval.
  - `agents/`: SAC variants, latent world model (`world_model.py`), diffusion/GAIL helpers.
  - `utils/`: Dynamic potential field, logging helpers.
- `docs/`: Benchmark notes (e.g., `NOCRASH.md`).
- `runs/`: Outputs (checkpoints, TensorBoard logs, configs). Treat as artifacts, not source.
- `requirements.txt`: Python dependencies (PyTorch, Gym, CARLA client).

## Build, Test, and Development Commands
- Create venv and install deps:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Run baseline training (requires CARLA server running on the configured port):
  - `python RL/train.py --run_name my_run`
- Run NoCrash training/eval:
  - `python RL/train_nocrash.py --run_name exp_nocrash`
  - `python RL/eval_nocrash.py --run_name exp_nocrash --episodes_per_scenario 5`
- Logs/checkpoints land under `RL/runs/<run_name>/`.

## Coding Style & Naming Conventions
- Python 3, PEP8-ish, 4-space indent. Prefer type hints where practical.
- Modules and files are lowercase with underscores; classes are `CamelCase`; functions/variables `snake_case`.
- Keep comments brief and functional; avoid redundant narration.
- Align with existing patterns in `RL/agents/` and `carla_env/`.

## Testing Guidelines
- No formal test suite is present. When adding features, include minimal reproducible scripts (e.g., small smoke run) or assertions where feasible.
- For env changes, sanity-check a short rollout: `python RL/train.py --max_episodes 1 --save_interval_episodes 0`.
- Keep new test/demo scripts out of `runs/`; place them in a dedicated `examples/` or inline docstring if small.

## Commit & Pull Request Guidelines
- No established history yet; use clear, imperative commits (e.g., `Add goal distance shaping to NoCrash`, `Fix lidar sensor cleanup`).
- In PRs, include: purpose, key changes, how to run, and risks/regressions. Add screenshots/metrics for behavior or reward changes when relevant.
- Do not commit large artifacts in `runs/`; treat them as generated outputs.

## Security & Configuration Tips
- CARLA client connects to `localhost` with a configurable port (default `2000`); avoid hardcoding external addresses.
- Keep credentials/tokens out of the codebase; use environment variables for any private endpoints.
