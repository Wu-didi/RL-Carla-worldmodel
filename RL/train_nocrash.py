# -*- coding: utf-8 -*-
"""
NoCrash Benchmark Training Script

Based on: "Exploring the Limitations of Behavior Cloning for Autonomous Driving"
Codevilla et al., ICCV 2019

Supports:
- Strict NoCrash benchmark mode (25 predefined routes, original traffic/weather)
- Simplified mode for faster iteration
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import traceback
import random
import time
import gc
import collections
import json
from dataclasses import asdict
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from agents.SAC import SAC_setup
from agents.world_model import DreamerWorldModel
from carla_env.nocrash_env import (
    NoCrashEnv,
    NoCrashScenario,
    WeatherGroup,
    make_default_scenarios,
    make_nocrash_scenarios,
    create_nocrash_training_scenarios,
    create_nocrash_newweather_scenarios,
    create_nocrash_newtown_scenarios,
)
from config import Config
from train import (
    set_seed,
    run_dir,
    log_dir,
    dump_config,
    make_agent,
    build_proprio_vector,
    log_losses,
    load_checkpoint_if_any,
    save_checkpoint,
)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, s, a, r, ns, d) -> None:
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*transitions)
        return np.array(s), a, r, np.array(ns), d

    def size(self) -> int:
        return len(self.buffer)


def make_nocrash_env(cfg: Config) -> NoCrashEnv:
    """
    Create NoCrash environment based on configuration.

    Supports two modes:
    - Strict mode (nocrash_strict=True): Uses original 25 routes, correct traffic/weather
    - Simplified mode: Uses custom scenarios for faster iteration
    """
    params = dict(
        number_of_vehicles=cfg.number_of_vehicles,
        number_of_walkers=cfg.number_of_walkers,
        dt=cfg.dt,
        ego_vehicle_filter=cfg.ego_vehicle_filter,
        surrounding_vehicle_spawned_randomly=cfg.surrounding_vehicle_spawned_randomly,
        port=cfg.port,
        town=cfg.town,
        max_time_episode=cfg.max_time_episode,
        max_waypoints=cfg.max_waypoints,
        visualize_waypoints=cfg.visualize_waypoints,
        desired_speed=cfg.desired_speed,
        max_ego_spawn_times=cfg.max_ego_spawn_times,
        view_mode=cfg.view_mode,
        traffic=cfg.traffic,
        lidar_max_range=cfg.lidar_max_range,
        max_nearby_vehicles=cfg.max_nearby_vehicles,
        use_camera=cfg.use_camera,
        use_lidar=cfg.use_lidar,
    )

    # Check for strict NoCrash mode
    nocrash_strict = getattr(cfg, 'nocrash_strict', False)
    nocrash_weather_group = getattr(cfg, 'nocrash_weather_group', 'training')

    if nocrash_strict:
        # Strict NoCrash benchmark mode
        print(f"[NoCrash] Using STRICT benchmark mode")
        print(f"[NoCrash] Town: {cfg.town}")
        print(f"[NoCrash] Weather group: {nocrash_weather_group}")
        print(f"[NoCrash] Traffic levels: {cfg.nocrash_scenarios}")

        weather_group = WeatherGroup.TRAINING if nocrash_weather_group == 'training' else WeatherGroup.NEW

        scenarios = make_nocrash_scenarios(
            town=cfg.town,
            traffic_levels=cfg.nocrash_scenarios,
            weather_group=weather_group,
            routes_per_condition=25,
        )
        use_predefined_routes = True
    else:
        # Simplified mode (backward compatible)
        print(f"[NoCrash] Using simplified mode (custom scenarios)")
        scenarios = make_default_scenarios(cfg.nocrash_scenarios, cfg.nocrash_weathers)
        use_predefined_routes = False

    print(f"[NoCrash] Total scenarios: {len(scenarios)}")

    env = NoCrashEnv(
        base_params=params,
        scenarios=scenarios,
        max_steps=cfg.nocrash_max_steps,
        goal_threshold=cfg.nocrash_goal_threshold,
        min_goal_distance=cfg.nocrash_min_goal_distance,
        use_predefined_routes=use_predefined_routes,
    )
    return env


def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    tb_dir = log_dir(cfg)
    os.makedirs(tb_dir, exist_ok=True)
    dump_config(cfg)

    writer = SummaryWriter(tb_dir)
    device = torch.device(cfg.device)
    env = make_nocrash_env(cfg)
    agent = make_agent(cfg)
    world_model = DreamerWorldModel(
        image_size=cfg.image_size,
        latent_dim=cfg.latent_dim,
        action_dim=cfg.action_dim,
        proprio_dim=cfg.proprio_dim,
        rssm_hidden_dim=cfg.rssm_hidden_dim,
        lr=cfg.world_model_lr,
        device=device,
    )
    load_checkpoint_if_any(cfg, agent, world_model)

    replay_buffer = ReplayBuffer(cfg.buffer_size)
    gstep = 0
    consecutive_failures = 0
    max_consecutive_failures = 5

    try:
        for ep in range(cfg.max_episodes):
            # Reset with error handling
            try:
                obs = env.reset()
                consecutive_failures = 0  # Reset failure counter on success
            except Exception as e:
                consecutive_failures += 1
                print(f"[Error] Reset failed ({consecutive_failures}/{max_consecutive_failures}): {e}")
                if consecutive_failures >= max_consecutive_failures:
                    print("[FATAL] Too many consecutive failures, check CARLA server!")
                    raise
                time.sleep(2.0)  # Wait before retry
                continue

            done = False
            ep_reward = 0.0
            ep_step = 0
            info = {}  # Initialize info to avoid undefined variable
            wm_state = world_model.init_state(batch_size=1)
            scenario_name = getattr(env.cur_scenario, "name", "unknown")
            step_failures = 0

            while not done:
                proprio = build_proprio_vector(obs)
                latent = world_model.encode(obs['rgb'], proprio)
                agent_state = world_model.agent_state(latent, proprio)[0]
                action = agent.take_action(agent_state)

                try:
                    next_obs, reward, cost, done, info = env.step(action)
                    step_failures = 0  # Reset on success
                except Exception as e:
                    step_failures += 1
                    traceback.print_exc()
                    print(f"[Error] Step failed ({step_failures}/3): {e}")
                    if step_failures >= 3:
                        print("[WARN] Too many step failures, forcing episode end...")
                        done = True
                        break
                    time.sleep(0.5)
                    continue

                next_proprio = build_proprio_vector(next_obs)
                pred_latent, next_wm_state = world_model.predict(latent, action, wm_state)
                target_latent = world_model.encode(next_obs['rgb'], next_proprio)
                with torch.no_grad():
                    wm_cos = float(F.cosine_similarity(pred_latent, target_latent, dim=-1).mean().item())
                wm_loss = world_model.optimize_world(pred_latent, target_latent)
                wm_state = next_wm_state.detach()

                agent_next_state = world_model.agent_state(target_latent.detach(), next_proprio)[0]
                replay_buffer.add(agent_state, action, reward, agent_next_state, done)

                if replay_buffer.size() > cfg.minimal_size and cfg.train_every_step:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(cfg.batch_size)
                    batch = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                    loss_dict = agent.update(batch)
                    loss_dict['world_model_loss'] = wm_loss
                    log_losses(writer, loss_dict, gstep, cfg.log_attention_image)
                writer.add_scalar("world_model/loss_step", wm_loss, env.env.total_step)
                writer.add_scalar("world_model/cosine", wm_cos, env.env.total_step)
                writer.add_scalar("env/reward_step", float(reward), env.env.total_step)

                obs = next_obs
                ep_reward += reward
                ep_step += 1
                gstep += 1

            writer.add_scalar("Episode Reward", float(ep_reward), ep)
            writer.add_scalar("Episode Steps", float(ep_step), ep)
            writer.add_scalar("Episode/avg_reward_per_step", float(ep_reward / max(ep_step, 1)), ep)
            if info:
                writer.add_scalar("Episode/dist_to_goal", float(info.get("dist_to_goal", 0.0)), ep)
                writer.add_scalar("Episode/success_flag", float(info.get("success", False)), ep)
                writer.add_scalar("Episode/collision_flag", float(info.get("collision", False)), ep)

            if (ep + 1) % cfg.save_interval_episodes == 0:
                save_checkpoint(cfg, agent, world_model, gstep, ep)

            print(f"[NoCrash Ep {ep:03d}] Scenario={scenario_name} Reward={ep_reward:.2f} Steps={ep_step} Success={info.get('success', False)}")

            # Periodic memory cleanup to prevent memory leaks
            if (ep + 1) % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                writer.flush()  # Flush TensorBoard data to disk

    finally:
        try:
            env.clear_all_actors([
                'sensor.other.collision',
                'sensor.lidar.ray_cast',
                'sensor.camera.rgb',
                'vehicle.*',
                'controller.ai.walker',
                'walker.*',
                'traffic.*',
            ])
            print("Cleared all Carla actors.")
        except Exception:
            traceback.print_exc()
        writer.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NoCrash benchmark trainer for CARLA")

    # Basic settings
    p.add_argument('--run_name', type=str, default=Config.run_name)
    p.add_argument('--run_root', type=str, default=Config.run_root)
    p.add_argument('--network', type=str, default=Config.network, choices=['SAC', 'Attention_SAC'])
    p.add_argument('--max_episodes', type=int, default=Config.max_episodes)
    p.add_argument('--save_interval_episodes', type=int, default=Config.save_interval_episodes)
    p.add_argument('--seed', type=int, default=Config.seed)

    # NoCrash benchmark settings
    p.add_argument('--nocrash_strict', action='store_true',
                   help="Use strict NoCrash benchmark (25 routes, original traffic/weather)")
    p.add_argument('--town', type=str, default="Town01",
                   help="CARLA town (Town01 for training, Town02 for testing)")
    p.add_argument('--nocrash_weather_group', type=str, default='training',
                   choices=['training', 'new'],
                   help="Weather group: 'training' (4 weathers) or 'new' (2 unseen weathers)")
    p.add_argument('--nocrash_max_steps', type=int, default=1000,
                   help="Max steps per episode (dynamic timeout based on route if strict)")
    p.add_argument('--nocrash_goal_threshold', type=float, default=2.0,
                   help="Distance to goal for success (meters)")
    p.add_argument('--nocrash_min_goal_distance', type=float, default=50.0,
                   help="Minimum goal distance for random routes (meters)")
    p.add_argument('--nocrash_scenarios', type=str, default="empty,regular,dense",
                   help="Comma-separated traffic levels")
    p.add_argument('--nocrash_weathers', type=str, default="ClearNoon,WetNoon,HardRainNoon,ClearSunset",
                   help="Comma-separated weather names (only used in non-strict mode)")

    # Pretrained model paths
    p.add_argument('--pretrained_agent_path', type=str, default=None)
    p.add_argument('--pretrained_world_model_path', type=str, default=None)

    return p


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    cfg.run_name = args.run_name
    cfg.run_root = args.run_root
    cfg.network = args.network
    cfg.max_episodes = args.max_episodes
    cfg.save_interval_episodes = args.save_interval_episodes
    cfg.seed = args.seed
    cfg.town = args.town
    cfg.nocrash_strict = args.nocrash_strict
    cfg.nocrash_weather_group = args.nocrash_weather_group
    cfg.nocrash_max_steps = args.nocrash_max_steps
    cfg.nocrash_goal_threshold = args.nocrash_goal_threshold
    cfg.nocrash_min_goal_distance = args.nocrash_min_goal_distance
    cfg.nocrash_scenarios = [s.strip() for s in args.nocrash_scenarios.split(",") if s.strip()]
    cfg.nocrash_weathers = [w.strip() for w in args.nocrash_weathers.split(",") if w.strip()]
    cfg.pretrained_agent_path = args.pretrained_agent_path
    cfg.pretrained_world_model_path = args.pretrained_world_model_path
    return cfg


def main() -> None:
    args = build_argparser().parse_args()
    cfg = apply_overrides(Config(), args)
    print("[Config]", asdict(cfg))
    train(cfg)


if __name__ == '__main__':
    main()
