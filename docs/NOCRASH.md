# NoCrash 基准测试说明

基于论文: ["Exploring the Limitations of Behavior Cloning for Autonomous Driving"](https://arxiv.org/abs/1904.08980) (Codevilla et al., ICCV 2019)

官方实现参考: [carla-simulator/driving-benchmarks](https://github.com/carla-simulator/driving-benchmarks)

## 目录
- [基准规范](#基准规范)
- [快速开始](#快速开始)
- [严格模式 vs 简化模式](#严格模式-vs-简化模式)
- [评估设置](#评估设置)
- [配置参数](#配置参数)
- [输出指标](#输出指标)

---

## 基准规范

### 原始 NoCrash 基准 (Codevilla et al., 2019)

| 项目 | 规范 |
|------|------|
| **路线** | 每个城镇 25 条预定义路线 |
| **城镇** | Town01 (训练), Town02 (测试/泛化) |
| **交通条件** | Empty, Regular, Dense |
| **天气条件** | 4 训练天气 + 2 新天气 (测试) |
| **成功标准** | 到达目标点且无碰撞 |
| **失败条件** | 碰撞任何物体 |

### 交通密度 (原始数值)

**Town01 (训练城镇):**
| 级别 | 车辆数 | 行人数 |
|------|--------|--------|
| Empty | 0 | 0 |
| Regular | 20 | 50 |
| Dense | 100 | 250 |

**Town02 (测试城镇):**
| 级别 | 车辆数 | 行人数 |
|------|--------|--------|
| Empty | 0 | 0 |
| Regular | 15 | 50 |
| Dense | 70 | 150 |

### 天气条件

**训练天气 (Training Weathers):**
- ClearNoon (晴朗正午)
- WetNoon (湿润正午)
- HardRainNoon (大雨正午)
- ClearSunset (晴朗日落)

**新天气 (New Weathers) - 测试用:**
- WetSunset (湿润日落)
- SoftRainSunset (小雨日落)

### 25 条预定义路线

路线定义为生成点索引对 `[start_idx, end_idx]`：

**Town01:**
```python
[[105, 29], [27, 130], [102, 87], [132, 27], [25, 44],
 [4, 64], [34, 67], [54, 30], [140, 134], [105, 9],
 [148, 129], [65, 18], [21, 16], [147, 97], [134, 49],
 [30, 41], [81, 89], [69, 45], [102, 95], [18, 145],
 [111, 64], [79, 45], [84, 69], [73, 31], [37, 81]]
```

**Town02:**
```python
[[19, 66], [79, 14], [19, 57], [39, 53], [60, 26],
 [53, 76], [42, 13], [31, 71], [59, 35], [47, 16],
 [10, 61], [66, 3], [20, 79], [14, 56], [26, 69],
 [79, 19], [2, 29], [16, 14], [5, 57], [77, 68],
 [70, 73], [46, 67], [34, 77], [61, 49], [21, 12]]
```

---

## 快速开始

### 严格 NoCrash 训练 (推荐用于论文)

```bash
# Town01 训练天气 (标准训练设置)
python RL/train_nocrash.py \
  --run_name nocrash_strict_train \
  --nocrash_strict \
  --town Town01 \
  --nocrash_weather_group training \
  --nocrash_scenarios empty,regular,dense \
  --network Attention_SAC \
  --max_episodes 3000
```

### 严格 NoCrash 评估

```bash
# Town01 训练天气评估
python RL/eval_nocrash.py \
  --run_name nocrash_strict_train \
  --nocrash_strict \
  --town Town01 \
  --nocrash_weather_group training \
  --pretrained_agent_path runs/nocrash_strict_train/checkpoints/agent/best.pt \
  --pretrained_world_model_path runs/nocrash_strict_train/checkpoints/world_model/best.pt

# Town01 新天气评估 (泛化测试)
python RL/eval_nocrash.py \
  --run_name nocrash_newweather_eval \
  --nocrash_strict \
  --town Town01 \
  --nocrash_weather_group new \
  --pretrained_agent_path runs/nocrash_strict_train/checkpoints/agent/best.pt

# Town02 评估 (城镇泛化测试)
python RL/eval_nocrash.py \
  --run_name nocrash_newtown_eval \
  --nocrash_strict \
  --town Town02 \
  --nocrash_weather_group training \
  --pretrained_agent_path runs/nocrash_strict_train/checkpoints/agent/best.pt
```

---

## 严格模式 vs 简化模式

### 严格模式 (`--nocrash_strict`)

适用于**论文发表**，完全复现原始 NoCrash 基准：

- ✅ 使用 25 条预定义路线
- ✅ 原始交通密度数值
- ✅ 正确的训练/测试天气划分
- ✅ 基于路线距离的动态超时
- ✅ 支持 Town01/Town02

```bash
python RL/train_nocrash.py \
  --nocrash_strict \
  --town Town01 \
  --nocrash_weather_group training
```

### 简化模式 (默认)

适用于**快速实验**和**调试**：

- 随机选择起点/终点
- 自定义天气组合
- 灵活的场景配置

```bash
python RL/train_nocrash.py \
  --nocrash_scenarios empty,regular \
  --nocrash_weathers ClearNoon,WetSunset \
  --nocrash_max_steps 600
```

---

## 评估设置

### 标准 NoCrash 评估表格

论文中通常报告以下评估结果：

| 条件 | Town01 训练天气 | Town01 新天气 | Town02 训练天气 |
|------|-----------------|---------------|-----------------|
| Empty | SR% | SR% | SR% |
| Regular | SR% | SR% | SR% |
| Dense | SR% | SR% | SR% |

SR% = Success Rate (成功率)

### 完整评估脚本

```bash
#!/bin/bash
MODEL_PATH="runs/your_experiment/checkpoints/agent/best.pt"
WORLD_PATH="runs/your_experiment/checkpoints/world_model/best.pt"

# Town01 + Training Weathers
python RL/eval_nocrash.py --nocrash_strict --town Town01 \
  --nocrash_weather_group training \
  --pretrained_agent_path $MODEL_PATH \
  --pretrained_world_model_path $WORLD_PATH \
  --run_name eval_t1_train

# Town01 + New Weathers
python RL/eval_nocrash.py --nocrash_strict --town Town01 \
  --nocrash_weather_group new \
  --pretrained_agent_path $MODEL_PATH \
  --pretrained_world_model_path $WORLD_PATH \
  --run_name eval_t1_new

# Town02 + Training Weathers
python RL/eval_nocrash.py --nocrash_strict --town Town02 \
  --nocrash_weather_group training \
  --pretrained_agent_path $MODEL_PATH \
  --pretrained_world_model_path $WORLD_PATH \
  --run_name eval_t2_train
```

---

## 配置参数

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--nocrash_strict` | False | 启用严格 NoCrash 基准模式 |
| `--town` | Town01 | CARLA 城镇 (Town01/Town02) |
| `--nocrash_weather_group` | training | 天气组: training (4个) 或 new (2个) |
| `--nocrash_scenarios` | empty,regular,dense | 交通密度级别 |
| `--nocrash_max_steps` | 1000 | 最大步数 (严格模式下动态计算) |
| `--nocrash_goal_threshold` | 2.0 | 成功判定距离 (米) |
| `--nocrash_weathers` | ClearNoon,... | 自定义天气 (仅简化模式) |

### 场景数量计算

**严格模式:**
```
总场景数 = 交通级别数 × 天气数 × 路线数
         = 3 × 4 × 25 = 300 (训练天气)
         = 3 × 2 × 25 = 150 (新天气)
```

**简化模式:**
```
总场景数 = 交通级别数 × 天气数
```

---

## 输出指标

### TensorBoard 记录

```
Episode/
├── success_flag        # 是否成功到达
├── collision_flag      # 是否碰撞
├── dist_to_goal        # 到目标距离
├── route_completion    # 路线完成度 (0-1)
└── timeout_flag        # 是否超时

NoCrash/
├── success_rate        # 累计成功率
├── collision_rate      # 累计碰撞率
└── total_episodes      # 总 episode 数
```

### 评估结果 JSON

`runs/<run_name>/nocrash_eval.json`:
```json
{
  "success_rate": {
    "empty_ClearNoon": 0.85,
    "empty_WetNoon": 0.80,
    "regular_ClearNoon": 0.65,
    "regular_WetNoon": 0.60,
    "dense_ClearNoon": 0.40,
    "dense_WetNoon": 0.35
  },
  "collision_rate": {
    "empty_ClearNoon": 0.10,
    ...
  },
  "episodes_per_scenario": 25,
  "total_scenarios": 300
}
```

---

## 论文引用

如果使用此 NoCrash 实现，请引用原始论文：

```bibtex
@inproceedings{codevilla2019exploring,
  title={Exploring the limitations of behavior cloning for autonomous driving},
  author={Codevilla, Felipe and Santana, Eder and L{\'o}pez, Antonio M and Gaidon, Adrien},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9329--9338},
  year={2019}
}
```

---

## 参考资料

- [原始论文 (arXiv)](https://arxiv.org/abs/1904.08980)
- [CARLA Driving Benchmarks](https://github.com/carla-simulator/driving-benchmarks)
- [DI-drive NoCrash 文档](https://opendilab.github.io/DI-drive/features/carla_benchmark.html)
- [Learning by Cheating](https://github.com/dotchen/LearningByCheating) - 另一个 NoCrash 实现参考
