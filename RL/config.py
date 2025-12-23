import torch
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Tuple, List, Optional

@dataclass
class Config:
    # Run / IO 路径
    run_root: str = 'runs'
    run_name: str = 'dreamer_latent'
    log_subdir: str = 'logs'
    agent_ckpt_subdir: str = 'checkpoints/agent'
    world_ckpt_subdir: str = 'checkpoints/world_model'
    config_filename: str = 'config.json'
    pretrained_agent_path: Optional[str] = None
    pretrained_world_model_path: Optional[str] = None

    # Env
    number_of_vehicles: int = 140
    number_of_walkers: int = 0
    dt: float = 0.1
    ego_vehicle_filter: str = 'vehicle.tesla.model3'
    surrounding_vehicle_spawned_randomly: bool = True
    port: int = 2000
    town: str = 'Town05'
    max_time_episode: int = 1000
    max_waypoints: int = 12
    visualize_waypoints: bool = True
    desired_speed: float = 14.0
    max_ego_spawn_times: int = 200
    view_mode: str = 'follow'  # 'top' or 'follow'
    traffic: str = 'off'  # 'on' or 'off'
    lidar_max_range: float = 50.0
    max_nearby_vehicles: int = 5
    use_camera: bool = True
    use_lidar: bool = False

    # SAC 网络与训练
    state_dim: int = 287
    hidden_dim: int = 540
    action_dim: int = 3
    action_bound: float = 1.0
    gamma: float = 0.99
    tau: float = 0.01
    actor_lr: float = 1e-4
    critic_lr: float = 4e-4
    alpha_lr: float = 1e-4
    target_entropy: float = -3.0
    network: str = 'SAC'  # 'SAC' or 'Attention_SAC'

    # Replay Buffer & 训练节拍
    buffer_size: int = 600_000
    minimal_size: int = 500
    batch_size: int = 100
    max_episodes: int = 10000
    save_interval_episodes: int = 1
    train_every_step: bool = True

    # 记录 / 模型路径与 ID
    tensorboard_id: int = 1026_3  # 保留旧字段兼容，但新的路径使用 run_root/run_name
    model_load_id: int = 1026     # 保留旧字段兼容
    model_save_id: int = 1026_3   # 保留旧字段兼容
    expert_save_id: int =  1026_3 # 保留旧字段兼容

    # 注意力图记录（仅 Attention SAC 使用）
    log_attention_image: bool = True

    # 滤波与起步控制（训练建议关闭，测试可打开）
    enable_action_filter: bool = False
    ramp_time: float = 1.5
    lpf_alpha: float = 0.8

    # 随机种子
    seed: int = 42

    # 设备
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # World model (Dreamer-style latent)
    image_size: Tuple[int, int] = (128, 128)
    latent_dim: int = 128
    rssm_hidden_dim: int = 256
    world_model_lr: float = 3e-4
    proprio_dim: int = 47

    # NoCrash 相关
    nocrash_scenarios: List[str] = field(default_factory=lambda: ["empty", "regular", "dense"])
    nocrash_weathers: List[str] = field(
        default_factory=lambda: ["ClearNoon", "WetNoon", "HardRainNoon", "ClearSunset"]
    )
    nocrash_max_steps: int = 1000
    nocrash_goal_threshold: float = 2.0
    nocrash_min_goal_distance: float = 50.0
    nocrash_strict: bool = False
    nocrash_weather_group: str = "training"  # training | test/new

    def __post_init__(self):
        self.proprio_dim = 9 + 2 + self.max_waypoints * 3
        self.state_dim = self.latent_dim + self.proprio_dim
