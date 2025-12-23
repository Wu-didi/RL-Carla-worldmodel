# -*- coding: utf-8 -*-
"""
NoCrash Benchmark Evaluation Script

Based on: "Exploring the Limitations of Behavior Cloning for Autonomous Driving"
Codevilla et al., ICCV 2019

[FIX #4] Properly aggregate results by (traffic, weather) according to NoCrash standard.
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agents.SAC import SAC_setup
from agents.world_model import DreamerWorldModel
from carla_env.nocrash_env import NoCrashEnv, make_default_scenarios
from config import Config
from train import (
    set_seed,
    run_dir,
    log_dir,
    dump_config,
    make_agent,
    build_proprio_vector,
    load_checkpoint_if_any,
)
from train_nocrash import make_nocrash_env


def evaluate(cfg: Config) -> Dict[str, Any]:
    """
    Run NoCrash evaluation following the original benchmark protocol.

    In strict mode:
    - 25 routes per (traffic, weather) combination
    - Results aggregated by traffic level (Empty/Regular/Dense)
    - Each route is run once (episodes_per_scenario is ignored in strict mode)
    """
    set_seed(cfg.seed)
    tb_dir = os.path.join(log_dir(cfg), "eval")
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

    # [FIX #4] Aggregate by (traffic, weather) combination
    # stats[traffic][weather] = list of success flags
    stats: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    collision_stats: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    timeout_stats: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    total_scenarios = len(env.scenarios)
    print(f"[NoCrash Eval] Total scenarios: {total_scenarios}")
    print(f"[NoCrash Eval] Town: {cfg.town}, Weather group: {cfg.nocrash_weather_group}")
    print(f"[NoCrash Eval] Traffic levels: {cfg.nocrash_scenarios}")
    print("-" * 60)

    try:
        for ep in range(total_scenarios):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            ep_step = 0

            # Get scenario info
            scenario = env.cur_scenario
            traffic = scenario.traffic.name
            weather = scenario.weather_name
            route_idx = scenario.route_idx

            while not done:
                proprio = build_proprio_vector(obs)
                latent = world_model.encode(obs['rgb'], proprio)
                agent_state = world_model.agent_state(latent, proprio)[0]
                action = agent.take_action(agent_state)
                obs, reward, cost, done, info = env.step(action)
                ep_reward += reward
                ep_step += 1

            success = int(info.get("success", False))
            collision = int(info.get("collision", False))
            timeout = int(info.get("timeout", False))

            # Record by (traffic, weather)
            stats[traffic][weather].append(success)
            collision_stats[traffic][weather].append(collision)
            timeout_stats[traffic][weather].append(timeout)

            # TensorBoard logging
            writer.add_scalar(f"eval/{traffic}/{weather}/success", success, route_idx)
            writer.add_scalar(f"eval/{traffic}/{weather}/collision", collision, route_idx)
            writer.add_scalar(f"eval/{traffic}_{weather}/reward", ep_reward, route_idx)

            status = "✓ Success" if success else ("✗ Collision" if collision else ("⏱ Timeout" if timeout else "? Other"))
            print(f"[{ep+1:03d}/{total_scenarios}] {traffic}/{weather}/route{route_idx:02d}: {status} (steps={ep_step}, reward={ep_reward:.1f})")

    finally:
        writer.close()

    # Build summary in NoCrash standard format
    summary = _build_nocrash_summary(stats, collision_stats, timeout_stats, cfg)

    # Save JSON
    out_path = os.path.join(run_dir(cfg), "nocrash_eval.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved eval summary → {out_path}")

    # Print table
    _print_nocrash_table(summary, cfg)

    return summary


def _build_nocrash_summary(
    stats: Dict[str, Dict[str, List[int]]],
    collision_stats: Dict[str, Dict[str, List[int]]],
    timeout_stats: Dict[str, Dict[str, List[int]]],
    cfg: Config,
) -> Dict[str, Any]:
    """Build NoCrash standard summary format."""
    summary = {
        "config": {
            "town": cfg.town,
            "weather_group": cfg.nocrash_weather_group,
            "traffic_levels": cfg.nocrash_scenarios,
            "strict_mode": cfg.nocrash_strict,
        },
        "success_rate": {},
        "collision_rate": {},
        "timeout_rate": {},
        "by_traffic": {},  # Aggregated by traffic level (standard format)
        "by_traffic_weather": {},  # Detailed by (traffic, weather)
    }

    # Aggregate by traffic level (NoCrash standard)
    for traffic in ["empty", "regular", "dense"]:
        if traffic not in stats:
            continue

        all_success = []
        all_collision = []
        all_timeout = []

        for weather, results in stats[traffic].items():
            all_success.extend(results)
            all_collision.extend(collision_stats[traffic][weather])
            all_timeout.extend(timeout_stats[traffic][weather])

            # Store detailed results
            key = f"{traffic}_{weather}"
            n = len(results)
            summary["by_traffic_weather"][key] = {
                "success_rate": float(np.mean(results)) if n > 0 else 0.0,
                "collision_rate": float(np.mean(collision_stats[traffic][weather])) if n > 0 else 0.0,
                "timeout_rate": float(np.mean(timeout_stats[traffic][weather])) if n > 0 else 0.0,
                "episodes": n,
            }

        # Aggregate for traffic level
        n_total = len(all_success)
        if n_total > 0:
            summary["by_traffic"][traffic] = {
                "success_rate": float(np.mean(all_success)),
                "collision_rate": float(np.mean(all_collision)),
                "timeout_rate": float(np.mean(all_timeout)),
                "episodes": n_total,
            }
            summary["success_rate"][traffic] = float(np.mean(all_success))
            summary["collision_rate"][traffic] = float(np.mean(all_collision))
            summary["timeout_rate"][traffic] = float(np.mean(all_timeout))

    return summary


def _print_nocrash_table(summary: Dict[str, Any], cfg: Config):
    """Print NoCrash results in standard table format."""
    print("\n" + "=" * 60)
    print(f"NoCrash Results: {cfg.town} / {cfg.nocrash_weather_group} weathers")
    print("=" * 60)

    # Header
    print(f"{'Traffic':<12} {'Success%':<12} {'Collision%':<12} {'Timeout%':<12} {'Episodes':<10}")
    print("-" * 60)

    # Rows
    for traffic in ["empty", "regular", "dense"]:
        if traffic in summary["by_traffic"]:
            data = summary["by_traffic"][traffic]
            sr = data["success_rate"] * 100
            cr = data["collision_rate"] * 100
            tr = data["timeout_rate"] * 100
            ep = data["episodes"]
            print(f"{traffic.capitalize():<12} {sr:<12.1f} {cr:<12.1f} {tr:<12.1f} {ep:<10}")

    print("=" * 60)

    # Overall
    all_sr = [summary["success_rate"].get(t, 0) for t in ["empty", "regular", "dense"] if t in summary["success_rate"]]
    if all_sr:
        print(f"{'Overall':<12} {np.mean(all_sr)*100:<12.1f}")
    print()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NoCrash benchmark evaluation for CARLA")

    # Basic settings
    p.add_argument('--run_name', type=str, default=Config.run_name)
    p.add_argument('--run_root', type=str, default=Config.run_root)

    # NoCrash benchmark settings
    p.add_argument('--nocrash_strict', action='store_true',
                   help="Use strict NoCrash benchmark (25 routes, original traffic/weather)")
    p.add_argument('--town', type=str, default="Town01",
                   help="CARLA town (Town01 for training, Town02 for testing)")
    p.add_argument('--nocrash_weather_group', type=str, default='training',
                   choices=['training', 'new'],
                   help="Weather group: 'training' (4 weathers) or 'new' (2 unseen weathers)")
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
    cfg.town = args.town
    cfg.nocrash_strict = args.nocrash_strict
    cfg.nocrash_weather_group = args.nocrash_weather_group
    cfg.nocrash_scenarios = [s.strip() for s in args.nocrash_scenarios.split(",") if s.strip()]
    cfg.nocrash_weathers = [w.strip() for w in args.nocrash_weathers.split(",") if w.strip()]
    cfg.pretrained_agent_path = args.pretrained_agent_path
    cfg.pretrained_world_model_path = args.pretrained_world_model_path
    return cfg


def main() -> None:
    args = build_argparser().parse_args()
    cfg = apply_overrides(Config(), args)
    print("[Config]", asdict(cfg))
    evaluate(cfg)


if __name__ == '__main__':
    main()
