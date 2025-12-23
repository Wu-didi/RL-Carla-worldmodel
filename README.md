# 项目概览
基于 CARLA 的驾驶强化学习示例，使用 Dreamer 风格的图像编码 + RSSM 预测 latent，SAC 以 "latent + 本车低维信息" 作为输入。已移除激光雷达与周车观测，所有决策依赖 RGB 相机与自监督的 latent。

## 运行前置
- CARLA 0.9.13（或兼容版本）已安装并可启动 `CarlaUE4.sh`
- Python 依赖：`pip install -r requirements.txt`

## 快速开始
1. 启动 CARLA：
   `./CarlaUE4.sh -RenderOffScreen -quality_level=Low -prefernvidia`
2. 训练：
   ```bash
   python RL/train.py \
     --run_name exp_dreamer1 \
     --run_root runs \
     --network Attention_SAC \
     --max_episodes 200 \
     --save_interval_episodes 5
   ```
3. TensorBoard：`tensorboard --logdir runs/exp_dreamer1/logs`

## 目录与输出
以 `run_root/run_name` 为根（默认 `runs/dreamer_latent`）：
- `logs/`：TensorBoard 事件文件。
- `checkpoints/agent/`：SAC 权重，命名 `sac_ep{ep}_gs{gstep}.pt`。
- `checkpoints/world_model/`：世界模型权重，命名 `world_ep{ep}_gs{gstep}.pt`。
- `config.json`：训练时保存的完整配置快照。

---

## NoCrash 基准测试

基于论文 [Exploring the Limitations of Behavior Cloning for Autonomous Driving](https://arxiv.org/abs/1904.08980) (Codevilla et al., ICCV 2019)

### 基准规范

| 项目 | 规范 |
|------|------|
| 路线 | 每个城镇 25 条预定义路线 |
| 城镇 | Town01 (训练), Town02 (测试泛化) |
| 交通条件 | Empty (0车/0人), Regular, Dense |
| 天气条件 | 4 训练天气 + 2 新天气 (测试泛化) |

**交通密度：**

| 城镇 | Empty | Regular | Dense |
|------|-------|---------|-------|
| Town01 | 0车/0人 | 20车/50人 | 100车/250人 |
| Town02 | 0车/0人 | 15车/50人 | 70车/150人 |

**天气条件：**
- 训练天气：ClearNoon, WetNoon, HardRainNoon, ClearSunset
- 新天气（测试）：WetSunset, SoftRainSunset

### 严格模式训练（论文发表推荐）

```bash
python RL/train_nocrash.py \
  --run_name nocrash_train \
  --nocrash_strict \
  --town Town01 \
  --nocrash_weather_group training \
  --nocrash_scenarios empty,regular,dense \
  --network Attention_SAC \
  --max_episodes 3000 \
  --save_interval_episodes 50
```

### 严格模式评估

```bash
# Town01 + 训练天气
python RL/eval_nocrash.py \
  --run_name nocrash_eval_t1_train \
  --nocrash_strict \
  --town Town01 \
  --nocrash_weather_group training \
  --pretrained_agent_path runs/nocrash_train/checkpoints/agent/best.pt \
  --pretrained_world_model_path runs/nocrash_train/checkpoints/world_model/best.pt

# Town01 + 新天气（泛化测试）
python RL/eval_nocrash.py \
  --run_name nocrash_eval_t1_new \
  --nocrash_strict \
  --town Town01 \
  --nocrash_weather_group new \
  --pretrained_agent_path runs/nocrash_train/checkpoints/agent/best.pt \
  --pretrained_world_model_path runs/nocrash_train/checkpoints/world_model/best.pt

# Town02 + 训练天气（城镇泛化测试）
python RL/eval_nocrash.py \
  --run_name nocrash_eval_t2_train \
  --nocrash_strict \
  --town Town02 \
  --nocrash_weather_group training \
  --pretrained_agent_path runs/nocrash_train/checkpoints/agent/best.pt \
  --pretrained_world_model_path runs/nocrash_train/checkpoints/world_model/best.pt
```

### 简化模式（快速调试）

```bash
python RL/train_nocrash.py \
  --run_name exp_nocrash_debug \
  --nocrash_scenarios empty,regular \
  --nocrash_weathers ClearNoon,WetSunset \
  --nocrash_max_steps 600 \
  --network Attention_SAC
```

### NoCrash 评估结果格式

论文中标准的结果表格：

| 条件 | Town01 训练天气 | Town01 新天气 | Town02 训练天气 |
|------|-----------------|---------------|-----------------|
| Empty | SR% | SR% | SR% |
| Regular | SR% | SR% | SR% |
| Dense | SR% | SR% | SR% |

SR% = Success Rate (成功率)

评估输出：
- TensorBoard：`logs/eval/` 下按场景记录 success/collision/off_road/timeout
- JSON 汇总：`runs/<run_name>/nocrash_eval.json`

### NoCrash 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--nocrash_strict` | False | 启用严格 NoCrash 基准模式 |
| `--town` | Town01 | CARLA 城镇 |
| `--nocrash_weather_group` | training | 天气组: `training` 或 `new` |
| `--nocrash_scenarios` | empty,regular,dense | 交通密度级别 |
| `--nocrash_max_steps` | 1000 | 最大步数 |
| `--nocrash_goal_threshold` | 2.0 | 成功判定距离 (米) |

详细说明见 [docs/NOCRASH.md](docs/NOCRASH.md)

---

## 主要配置字段

`RL/config.py` 或 CLI 覆盖：

- `run_root`/`run_name`：输出根目录与实验名
- `pretrained_agent_path` / `pretrained_world_model_path`：预训练权重路径
- `save_interval_episodes`：模型保存间隔
- SAC：`state_dim`、`hidden_dim`、`action_dim`、`gamma`、`tau`、学习率等
- 世界模型：`image_size`、`latent_dim`、`rssm_hidden_dim`、`world_model_lr`
- 环境：`number_of_vehicles`、`town`、`max_waypoints` 等

## 预训练加载

启动时指定 `--pretrained_agent_path` 或 `--pretrained_world_model_path`，会尝试加载对应权重；加载失败会给出警告但继续训练。

## 训练指标

TensorBoard 记录：
- 世界模型：`world_model/loss_step`、`cosine`、`pred_norm`、`target_norm`
- SAC：actor/critic/alpha 损失、Q 值、entropy
- 环境：step reward、episode 奖励、步数

## 重要实现摘要

- **世界模型**：Dreamer 风格 CNN 编码 + RSSM 预测，MSE 自监督
- **观测**：RGB 相机 + 本车状态 + 航路点
- **SAC 输入**：`[latent, proprio]` 连接，Attention 版对两段 token 做加性注意力

## 引用

如果使用 NoCrash 基准，请引用：

```bibtex
@inproceedings{codevilla2019exploring,
  title={Exploring the limitations of behavior cloning for autonomous driving},
  author={Codevilla, Felipe and Santana, Eder and L{\'o}pez, Antonio M and Gaidon, Adrien},
  booktitle={ICCV},
  year={2019}
}
```
