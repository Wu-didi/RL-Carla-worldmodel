# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse
import traceback
import random
import collections
from dataclasses import asdict
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import json

# ----- 外部依赖（与你原项目一致） -----
from agents.SAC import SAC_setup
from agents.world_model import DreamerWorldModel
from carla_env.carla_env import CarlaEnv
from config import Config



# ===================== 实用工具 =====================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def run_dir(cfg: Config) -> str:
    return os.path.join(project_root(), cfg.run_root, cfg.run_name)


def log_dir(cfg: Config) -> str:
    return os.path.join(run_dir(cfg), cfg.log_subdir)


def agent_ckpt_dir(cfg: Config) -> str:
    return os.path.join(run_dir(cfg), cfg.agent_ckpt_subdir)


def world_ckpt_dir(cfg: Config) -> str:
    return os.path.join(run_dir(cfg), cfg.world_ckpt_subdir)


def dump_config(cfg: Config) -> None:
    os.makedirs(run_dir(cfg), exist_ok=True)
    cfg_path = os.path.join(run_dir(cfg), cfg.config_filename)
    with open(cfg_path, 'w') as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"Saved config → {cfg_path}")


# ===================== 观测向量化 =====================

def build_proprio_vector(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Extract low-dimensional proprioceptive inputs (no lidar / nearby vehicles)."""
    return np.concatenate([
        obs_dict['ego_state'].flatten(),        # 9
        obs_dict['lane_info'].flatten(),        # 2
        obs_dict['waypoints'].flatten(),        # 3 * max_waypoints
    ]).astype(np.float32)


# ===================== 经验回放池 =====================
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


# ===================== 日志器 =====================
class CarlaLogger:
    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        self.ep_actions: List[np.ndarray] = []

    def step(self, action: np.ndarray) -> None:
        self.ep_actions.append(action)

    def episode_done(self, gstep: int) -> None:
        if not self.ep_actions:
            return
        ep_actions = np.stack(self.ep_actions)
        self.ep_actions.clear()
        names = ["throttle", "steer", "brake"]
        for i, name in enumerate(names):
            self.writer.add_scalar(f"action/{name}_mean", float(ep_actions[:, i].mean()), gstep)
            self.writer.add_scalar(f"action/{name}_std", float(ep_actions[:, i].std()), gstep)
        self.writer.flush()

# ===================== 构建环境与智能体 =====================

def make_env(cfg: Config) -> CarlaEnv:
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
    # 兼容你原始脚本中需要的 model_id（用于统计与可视化）
    return CarlaEnv(params=params)


def make_agent(cfg: Config) -> SAC_setup:
    device = torch.device(cfg.device)
    agent = SAC_setup(
        state_dim=cfg.state_dim,
        hidden_dim=cfg.hidden_dim,
        action_dim=cfg.action_dim,
        action_bound=cfg.action_bound,
        device=device,
        gamma=cfg.gamma,
        tau=cfg.tau,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        alpha_lr=cfg.alpha_lr,
        target_entropy=cfg.target_entropy,
        network=cfg.network,
        latent_dim=cfg.latent_dim,
        proprio_dim=cfg.proprio_dim,
    )
    return agent


# ===================== 训练核心 =====================
def log_losses(writer: SummaryWriter, loss_dict: Dict[str, Any], gstep: int, log_attention_image: bool) -> None:
    for key, value in loss_dict.items():
        if key == 'attention_img' and log_attention_image:
            writer.add_image('attention/alpha_heatmap', loss_dict['attention_img'], global_step=gstep)
        elif key != 'attention_img':
            # value 可能为标量或标量张量
            scalar = float(value) if not isinstance(value, (float, int)) else value
            writer.add_scalar(key, scalar, global_step=gstep)


def save_checkpoint(cfg: Config, agent: SAC_setup, world_model: DreamerWorldModel, gstep: int, ep: int) -> None:
    agent_dir = agent_ckpt_dir(cfg)
    wm_dir = world_ckpt_dir(cfg)
    os.makedirs(agent_dir, exist_ok=True)
    os.makedirs(wm_dir, exist_ok=True)

    # Save agent checkpoint
    agent_ckpt_id = f"sac_ep{ep:05d}_gs{gstep:06d}"
    agent.save_model(agent_dir, id=agent_ckpt_id)
    agent_ckpt_file = f"sac_ckpt_{agent_ckpt_id}.pt"

    # Save world model checkpoint
    wm_ckpt_name = f"world_ep{ep:05d}_gs{gstep:06d}.pt"
    world_model.save(os.path.join(wm_dir, wm_ckpt_name))

    # Create symlinks to last checkpoint (like many popular projects do)
    agent_last = os.path.join(agent_dir, "last.pt")
    wm_last = os.path.join(wm_dir, "last.pt")

    # Remove old symlinks if exist
    if os.path.islink(agent_last):
        os.remove(agent_last)
    if os.path.islink(wm_last):
        os.remove(wm_last)

    # Create new symlinks (relative path for portability)
    os.symlink(agent_ckpt_file, agent_last)
    os.symlink(wm_ckpt_name, wm_last)

    print(f"[Checkpoint] agent→{agent_ckpt_file}, world→{wm_ckpt_name} (last.pt updated)")


def load_checkpoint_if_any(cfg: Config, agent: SAC_setup, world_model: DreamerWorldModel) -> None:
    if cfg.pretrained_agent_path:
        try:
            if os.path.isfile(cfg.pretrained_agent_path):
                agent.load_model_path(cfg.pretrained_agent_path)
                print(f"[OK] Loaded agent model {cfg.pretrained_agent_path}")
            else:
                print(f"[WARN] Agent checkpoint not found at {cfg.pretrained_agent_path}, skip loading.")
        except Exception as e:
            print(f"[WARN] Load agent model failed ({cfg.pretrained_agent_path}): {e}")
    if cfg.pretrained_world_model_path:
        try:
            world_model.load(cfg.pretrained_world_model_path)
            print(f"[OK] Loaded world model {cfg.pretrained_world_model_path}")
        except Exception as e:
            print(f"[WARN] Load world model failed ({cfg.pretrained_world_model_path}): {e}")


# ===================== 主训练流程 =====================
def train(cfg: Config) -> None:
    set_seed(cfg.seed)

    # 目录与配置保存
    tb_dir = log_dir(cfg)
    os.makedirs(tb_dir, exist_ok=True)
    dump_config(cfg)

    # Writer & 日志器
    writer = SummaryWriter(tb_dir)
    print(f"TensorBoard logs → {tb_dir}")
    action_logger = CarlaLogger(writer)

    # 环境 / agent / 世界模型 / 回放池
    device = torch.device(cfg.device)
    env = make_env(cfg)
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

    # 可选加载预训练模型 
    load_checkpoint_if_any(cfg, agent, world_model)
    replay_buffer = ReplayBuffer(cfg.buffer_size)

    gstep = 0  # 全局步数计数（按 episode 累加也可）

    try:
        for ep in range(cfg.max_episodes):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            ep_step = 0
            wm_state = world_model.init_state(batch_size=1)
            while not done:
                proprio = build_proprio_vector(obs)
                latent = world_model.encode(obs['rgb'], proprio)
                agent_state = world_model.agent_state(latent, proprio)[0]
                action = agent.take_action(agent_state)

                try:
                    next_obs, reward, cost, done, info = env.step(action)
                except Exception:
                    traceback.print_exc()
                    print("[Error] Carla step failed; resetting env...")
                    obs = env.reset()
                    continue

                next_proprio = build_proprio_vector(next_obs)
                pred_latent, next_wm_state = world_model.predict(latent, action, wm_state)
                target_latent = world_model.encode(next_obs['rgb'], next_proprio)
                with torch.no_grad():
                    wm_cos = float(F.cosine_similarity(pred_latent, target_latent, dim=-1).mean().item())
                    wm_pred_norm = float(pred_latent.norm(dim=-1).mean().item())
                    wm_tgt_norm = float(target_latent.norm(dim=-1).mean().item())
                    wm_hid_norm = float(next_wm_state.norm(dim=-1).mean().item())
                wm_loss = world_model.optimize_world(pred_latent, target_latent)
                wm_state = next_wm_state.detach()

                agent_next_state = world_model.agent_state(target_latent.detach(), next_proprio)[0]

                # ===== 经验入池 =====
                replay_buffer.add(agent_state, action, reward, agent_next_state, done)

                # ===== 训练步 =====
                if replay_buffer.size() > cfg.minimal_size and cfg.train_every_step:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(cfg.batch_size)
                    batch = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                    loss_dict = agent.update(batch)
                    loss_dict['world_model_loss'] = wm_loss
                    log_losses(writer, loss_dict, gstep, cfg.log_attention_image)
                writer.add_scalar("world_model/loss_step", wm_loss, env.total_step)
                writer.add_scalar("world_model/cosine", wm_cos, env.total_step)
                writer.add_scalar("world_model/pred_norm", wm_pred_norm, env.total_step)
                writer.add_scalar("world_model/target_norm", wm_tgt_norm, env.total_step)
                writer.add_scalar("world_model/rssm_hidden_norm", wm_hid_norm, env.total_step)
                writer.add_scalar("env/reward_step", float(reward), env.total_step)
                writer.add_scalar("env/done_flag", float(done), env.total_step)
                writer.add_scalar("replay/size", replay_buffer.size(), env.total_step)
                writer.add_scalar("action/abs_mean_step", float(np.abs(action).mean()), env.total_step)
                writer.add_scalar("action/max_abs_step", float(np.abs(action).max()), env.total_step)

                # 汇总
                obs = next_obs
                ep_reward += reward
                ep_step += 1

            # ===== Episode 结束处理 =====
            writer.add_scalar("Episode Reward", float(ep_reward), gstep)
            writer.add_scalar("Episode Steps", float(ep_step), gstep)
            if ep_step > 0:
                writer.add_scalar("Episode/avg_reward_per_step", float(ep_reward / ep_step), gstep)
            action_logger.episode_done(gstep)

            # 保存 checkpoint（这里按 episode 保存；可改为间隔保存）
            if (ep + 1) % cfg.save_interval_episodes == 0:
                save_checkpoint(cfg, agent, world_model, gstep, ep)

            print(f"[Episode {ep:03d}] Reward={ep_reward:.2f} Steps={ep_step} gstep={gstep}")
            gstep += 1

    finally:
        # 清理 CARLA actors（与你原脚本一致）
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


# ===================== CLI =====================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean RL trainer for carla")
    # 只暴露常用可变项；其余请直接改 Config 默认值或追加参数
    p.add_argument('--town', type=str, default=Config.town)
    p.add_argument('--run_name', type=str, default=Config.run_name, help="Run name used to build paths under run_root")
    p.add_argument('--run_root', type=str, default=Config.run_root, help="Root directory for runs/logs/checkpoints")
    p.add_argument('--pretrained_agent_path', type=str, default=None, help="Path to pretrained SAC checkpoint")
    p.add_argument('--pretrained_world_model_path', type=str, default=None, help="Path to pretrained world model checkpoint")
    p.add_argument('--network', type=str, default=Config.network, choices=['SAC', 'Attention_SAC'])  # TODO: add more networks  e.g., 'ppo', 'dqn'
    p.add_argument('--max_episodes', type=int, default=Config.max_episodes)
    p.add_argument('--save_interval_episodes', type=int, default=Config.save_interval_episodes)
    p.add_argument('--seed', type=int, default=Config.seed)
    return p


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    cfg.run_name = args.run_name
    cfg.run_root = args.run_root
    cfg.town = args.town  # Fix: apply --town argument
    cfg.pretrained_agent_path = args.pretrained_agent_path
    cfg.pretrained_world_model_path = args.pretrained_world_model_path
    cfg.network = args.network
    cfg.max_episodes = args.max_episodes
    cfg.save_interval_episodes = args.save_interval_episodes
    cfg.seed = args.seed
    return cfg


def main() -> None:
    args = build_argparser().parse_args()
    cfg = apply_overrides(Config(), args)
    print("[Config]", asdict(cfg))
    train(cfg)


if __name__ == '__main__':
    main()
