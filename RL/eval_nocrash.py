# -*- coding: utf-8 -*-
"""
NoCrash Benchmark Evaluation Script

Based on: "Exploring the Limitations of Behavior Cloning for Autonomous Driving"
Codevilla et al., ICCV 2019
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
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


def evaluate(cfg: Config, episodes_per_scenario: int = 5) -> Dict[str, Any]:
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

    stats: Dict[str, List[float]] = {}

    try:
        for ep in range(episodes_per_scenario * len(env.scenarios)):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            ep_step = 0
            scenario_name = getattr(env.cur_scenario, "name", f"ep{ep}")
            while not done:
                proprio = build_proprio_vector(obs)
                latent = world_model.encode(obs['rgb'], proprio)
                agent_state = world_model.agent_state(latent, proprio)[0]
                action = agent.take_action(agent_state)
                obs, reward, cost, done, info = env.step(action)
                ep_reward += reward
                ep_step += 1

            success = float(info.get("success", False))
            collision = float(info.get("collision", False))
            off_road = float(info.get("off_road", False))
            timeout = float(info.get("timeout", False))

            writer.add_scalar(f"eval/{scenario_name}/success", success, ep)
            writer.add_scalar(f"eval/{scenario_name}/collision", collision, ep)
            writer.add_scalar(f"eval/{scenario_name}/off_road", off_road, ep)
            writer.add_scalar(f"eval/{scenario_name}/timeout", timeout, ep)
            writer.add_scalar(f"eval/{scenario_name}/reward", ep_reward, ep)
            writer.add_scalar(f"eval/{scenario_name}/steps", ep_step, ep)

            if scenario_name not in stats:
                stats[scenario_name] = []
            stats[scenario_name].append(success)

            print(f"[Eval {ep:03d}] Scenario={scenario_name} Success={bool(success)} Collision={bool(collision)} Timeout={bool(timeout)} Reward={ep_reward:.2f}")

    finally:
        writer.close()

    summary = {
        "success_rate": {k: float(np.mean(v)) for k, v in stats.items()},
        "episodes": episodes_per_scenario,
    }
    out_path = os.path.join(run_dir(cfg), "nocrash_eval.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved eval summary → {out_path}")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NoCrash benchmark evaluation for CARLA")

    # Basic settings
    p.add_argument('--run_name', type=str, default=Config.run_name)
    p.add_argument('--run_root', type=str, default=Config.run_root)
    p.add_argument('--episodes_per_scenario', type=int, default=25,
                   help="Episodes per scenario (25 for full NoCrash eval)")

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
    evaluate(cfg, episodes_per_scenario=args.episodes_per_scenario)


if __name__ == '__main__':
    main()
