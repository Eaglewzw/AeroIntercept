# 实验 C：预训练视觉主干、Gazebo 行为克隆与 PPO 微调

## 实验目的

实验 C 用于验证以下假设：在保持纯 RGB Actor、640×640 全画面和现有动作协议不变的
条件下，ImageNet 预训练视觉表征与同域 Gazebo 专家示范能够缩短随机初始化 PPO 的
探索阶段，并提高小目标注意力与早期训练稳定性。

实验定义固定为：

```text
ImageNet ResNet-18
        ↓
stride-8/stride-16 多尺度融合 + 空间注意力
        ↓
Gazebo/PX4 专家行为克隆（BC）
        ↓
Gazebo/PX4 CUDA PPO 微调
```

本实验不改变 ros2_ws 中的 C++ 代码，不改变 PX4、Gazebo、Python、PyTorch 或 CUDA
版本，也不使用轻量二维渲染器生成训练图像。

## 固定协议

| 项目 | 实验 C 设置 |
|---|---|
| Actor 输入 | `[N,2,3,640,640] uint8`，RGB，全画面 letterbox |
| 归一化 | 模型内 `float32 / 255`，ImageNet mean/std |
| 视觉主干 | `ResNet-18 IMAGENET1K_V1`，截取至 layer3 |
| 空间特征 | stride-8 与 stride-16 融合，输出 80×80 attention |
| 时序模型 | 2 个帧 token，2 层 Transformer，embedding 128 |
| Actor 输出 | `[forward,right,down,yaw_rate] ∈ [-1,1]` |
| Critic | 独立 15 维仿真真值 MLP，仅训练使用 |
| BC 专家 | Gazebo 真值前置追踪控制器，仅生成动作标签 |
| 初始距离 | 10±1 m |

专家真值只能通过 `GazeboInterceptEnv.expert_state()` 进入数据采集器。Actor 的
`forward` 仍只有 `frames` 一个参数；部署导出只包含 Actor。数据分片只保存每个控制步
的一张 640×640 RGB 当前帧，训练时在 episode 内构造连续双帧，避免重复存储。

## 超参数

`configs/gazebo_e2e.yaml` 是实验 C 的唯一默认配置。关键设置如下：

- BC：200 episodes、sequence 16、batch 2、20 epochs；
- 主干冻结：前 3 个 BC epoch；
- BC 学习率：任务层 `3e-4`，ImageNet 主干 `1e-5`；
- PPO 学习率：任务层 `5e-5`，ImageNet 主干 `5e-6`；
- PPO 冒烟：1 环境、rollout 16、总步数 512、encoder chunk 4。

先以 5 至 20 个 episode 运行数据和显存试验，确认链路后再采集 200 个 episode。真实
公园图像的压缩率取决于纹理与运动，采集前应检查 `data/` 可用空间。

## 执行命令

以下命令均在现有 `AeroIntercept` Conda 环境与工程根目录执行。

### 1. 小规模数据链路验证

```bash
python -m aerointercept.gazebo.scripts.collect_bc_data \
  --launch --headless --mode circle --seed 31 --episodes 2 \
  --out data/gazebo_experiment_c_pilot --overwrite

python -m aerointercept.training.train_e2e_bc \
  --backend gazebo --config configs/gazebo_e2e.yaml \
  --data data/gazebo_experiment_c_pilot \
  --out checkpoints/experiment_c_pilot_bc.pt --device cuda:0 \
  --epochs 1 --batch-size 1 --sequence-length 4
```

### 2. 正式行为克隆数据与训练

```bash
python -m aerointercept.gazebo.scripts.collect_bc_data \
  --launch --headless --mode mixed --seed 31 --episodes 200 \
  --out data/gazebo_experiment_c --overwrite

python -m aerointercept.training.train_e2e_bc \
  --backend gazebo --config configs/gazebo_e2e.yaml \
  --data data/gazebo_experiment_c \
  --out checkpoints/experiment_c_bc.pt --device cuda:0
```

### 3. BC 初始化的 512-step PPO 验收

```bash
python -m aerointercept.gazebo.scripts.train_e2e_ppo \
  --launch --headless --config configs/gazebo_e2e.yaml \
  --device cuda:0 --num-envs 1 --seed 31 \
  --total-steps 512 --rollout-steps 16 --encoder-chunk-size 4 \
  --bc-init checkpoints/experiment_c_bc.pt \
  --checkpoint-interval 256 --mode mixed \
  --logdir runs/experiment_c_ppo_512
```

验收输出必须满足 `parameter_delta > 0`，并生成 best、last 和周期 checkpoint。

### 4. 恢复、评估与曲线

```bash
python -m aerointercept.gazebo.scripts.train_e2e_ppo \
  --launch --headless --device cuda:0 --num-envs 1 --seed 31 \
  --total-steps 512 --rollout-steps 16 \
  --checkpoint runs/experiment_c_ppo_512/checkpoints/last.pt --resume \
  --checkpoint-interval 256 --mode mixed \
  --logdir runs/experiment_c_ppo_512

python -m aerointercept.gazebo.scripts.evaluate \
  --launch --headless --device cuda:0 --episodes 10 --mode mixed \
  --checkpoint runs/experiment_c_ppo_512/checkpoints/best.pt \
  --output results/experiment_c_eval.json

tensorboard --logdir runs/experiment_c_ppo_512 --port 6006
```

## 验收记录

实验记录只填写实际执行结果，不使用估计值代替测量值。

| 日期 | 验收项 | 结果 | 证据/备注 |
|---|---|---|---|
| 2026-08-26 | ResNet-18 多尺度模型离线构建与 TorchScript | 通过 | 64×64 测试输出 attention 8×8，参数量 3,134,094 |
| 2026-08-26 | 专家控制器初次物理 pilot | 未通过并已定位 | circle 最小距离 2.36 m，600 step 超时；前置量与目标速度前馈重复，形成跟随轨道，未纳入有效 BC 数据 |
| 2026-08-26 | 修正后 Gazebo 专家采集 2 episodes | 通过 | 2/2 命中；每集 56 step；112 帧；31 MB；连续帧平均像素差约 3.5 |
| 2026-08-26 | ImageNet 权重下载与 640×640 CUDA 前向 | 通过 | 官方 `IMAGENET1K_V1`；attention 80×80；PyTorch 峰值约 397.72 MB |
| 2026-08-26 | CUDA BC 1 epoch | 通过 | val total 0.3630；action MSE 0.0589；1.12 s；峰值 364.94 MB |
| 2026-08-26 | BC 初始化 512-step CUDA PPO | 通过 | 32 updates；parameter delta 0.00438285；完整指标见下表 |
| 2026-08-26 | checkpoint 恢复 | 通过 | 模型、两组优化器与 RNG 从 step 512 恢复，继续至 step 528 |
| 2026-08-26 | 纯 Actor 部署导出 | 通过 | 13 MB TorchScript；640×640 RGB；`contains_critic=false` |
| 2026-08-26 | 1 episode 部署路径评估 | 链路通过、策略未收敛 | fov_lost；minimum distance 3.52 m；仅 2 个 BC episode，不作为性能结论 |
| 2026-08-26 | 全量回归测试 | 通过 | 56 passed；仅有 TorchScript deprecation warnings |

### 512-step 实测性能

RTX 5070 Ti 16GB、单环境、circle、seed 31、rollout 16、encoder chunk 4：

| 指标 | 旧轻量编码器基线 | 实验 C ResNet-18 | 变化 |
|---|---:|---:|---:|
| PPO updates | 32 | 32 | 相同 |
| Actor parameter max delta | 0.00672755 | 0.00438285 | 均完成实际更新 |
| 整机 GPU 峰值 | 3284 MB | 3253 MB | -31 MB |
| PyTorch allocated 峰值 | 1918 MB | 1570 MB | -348 MB |
| 相机 FPS | 7.49 | 6.70 | -10.5% |
| 训练 FPS | 7.26 | 6.55 | -9.8% |
| PPO 更新累计耗时 | 17.46 s | 18.60 s | +6.5% |
| 总耗时 | 70.53 s | 78.18 s | +10.8% |

实验 C 的低 PyTorch 峰值来自截断至 ResNet layer3 并保持 chunk 4；整机显存仍包含
Gazebo 渲染开销。上述 pilot 数据只有两个同 seed circle episode，验证用途是打通链路，
不能据此比较收敛速度或命中率。正式结论需要按本页命令采集 mixed 200 episodes，并与
相同 seed 和训练步数的随机初始化对照。

## 对照原则

实验 C 与随机初始化基线必须使用相同 seed、目标模式、初始距离、rollout 长度、奖励与
评估 episode 数。至少比较 BC action validation loss、PPO policy/value/auxiliary loss、
approximate KL、hit rate、minimum distance、相机 FPS、训练 FPS 与 GPU 峰值显存。
旧轻量编码器在 2026-08-24 的性能数据只能作为基线，不能标注为实验 C 结果。
