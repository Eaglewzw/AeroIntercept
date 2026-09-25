# Gazebo / PX4 训练指南

## 当前非接触会合任务

当前验收使用初始中心距 10 m、成功中心距 ≤0.5 m，并要求低相对速度持续保持且无接触。
坐标与接触监测修正前的历史数据和 checkpoint 不适用于当前任务。
最新实测、任务定义和验收门槛见 [工程核查记录](noncontact_rendezvous_status.md)。

基础数据 `data/noncontact_center_v2` 已完成；当前采集调整专家轨迹后的纠偏数据
`data/noncontact_geometry_corrective`，同样包含 PX4 自身状态、正确机体中心及相机投影标签。
`data/noncontact_rgb_state` 与 `data/noncontact_formal` 均属于旧模型原点协议，不能通过当前任务检查。
自身状态协议见 [协议说明](own_state_protocol.md)。以下是当前入口；命令列出不代表
对应训练或验收已经完成，实际结果以日志和评估 JSON 为准。

```bash
python -u -m aerointercept.training.train_e2e_bc \
  --backend gazebo --config configs/gazebo_feedback.yaml --data data/noncontact_geometry_corrective \
  --init-checkpoint checkpoints/noncontact_center_v2_feedback_init.pt \
  --action-loss-weight 20 --selection-metric action --learning-rate 0.0001 \
  --out checkpoints/noncontact_geometry_bc.pt --device cuda:0 --epochs 20

python -m aerointercept.gazebo.scripts.train_e2e_ppo \
  --launch --headless --device cuda:0 --mode mixed --seed 100 \
  --config configs/gazebo_feedback.yaml --bc-init checkpoints/noncontact_geometry_bc.pt \
  --total-steps 512 --rollout-steps 16 --logdir runs/noncontact_geometry_ppo

# 开发评估：先评估 BC，再以相同场景计划比较 PPO；不可用此小样本宣称验收。
python -m aerointercept.gazebo.scripts.evaluate \
  --launch --headless --device cuda:0 --suite --episodes 2 --seed 10000 \
  --config configs/gazebo_feedback.yaml --checkpoint checkpoints/noncontact_geometry_bc.pt \
  --output results/noncontact/geometry_bc_development.json
```

最终冻结模型的验收使用 `--suite --episodes 20 --seed 20000`，输出需包含
`complete: true` 和 `acceptance.passed: true`；同时保留逐回合记录与各场景置信区间。
不要以训练期间 `best.pt` 的训练成功率代替独立评估，也不要以 BC 验证损失代替闭环结果。

## 前置检查

固定环境：Gazebo Harmonic、ROS 2 Humble、PX4 SITL、Micro XRCE Agent，以及现有
`AeroIntercept` Conda 环境。不要创建新 Conda 环境，也不要修改 `ros2_ws` C++。

```bash
gz sim --versions
test -x /home/verser/PX4-Autopilot/build/px4_sitl_default/bin/px4
test -x /home/verser/ros2_ws/build/uav_target_sim/uav_target_sim
```

launcher 若发现已有 PX4、Gazebo 或 Micro XRCE 进程会直接拒绝启动，不会静默
`killall`。

## GUI 观察

终端一：

```bash
bash /home/verser/Python/AeroIntercept/aerointercept/gazebo/scripts/launch_gazebo.sh \
  --mode circle --seed 31
```

终端二：

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate AeroIntercept
cd /home/verser/Python/AeroIntercept
python -m aerointercept.gazebo.scripts.view_camera --display
```

Gazebo GUI 显示外部视角；`view_camera` 显示 Actor 收到的精确 640×640 RGB 输入。

## 冒烟

```bash
python -m aerointercept.gazebo.scripts.smoke_camera \
  --launch --headless --frames 8 --mode circle \
  --output results/gazebo_camera.png

python -m aerointercept.gazebo.scripts.smoke_env \
  --launch --headless --steps 32 --mode sinusoidal --seed 31
```

camera smoke 检查形状、dtype、范围、常量帧、连续序号和独立缓冲区；env smoke 检查
物理 reset、PX4 action、15 维 Critic、奖励和终止信息。

## 历史纯视觉实验 C 命令

本节保留旧工作流记录；旧纯视觉 checkpoint 不能直接恢复到当前六维自身状态配置。
当前训练应使用本文开头的新协议命令。

当前 Gazebo 默认配置属于实验 C。正式 PPO 前先按
[实验 C 记录](experiment_c.md)采集 Gazebo 示范并完成行为克隆，然后使用
`--bc-init` 初始化：

```bash
python -m aerointercept.gazebo.scripts.collect_bc_data \
  --launch --headless --mode mixed --seed 31 --episodes 200 \
  --out data/gazebo_experiment_c

python -m aerointercept.training.train_e2e_bc \
  --backend gazebo --config configs/gazebo_e2e.yaml \
  --data data/gazebo_experiment_c \
  --out checkpoints/experiment_c_bc.pt --device cuda:0
```

```bash
python -m aerointercept.gazebo.scripts.train_e2e_ppo \
  --launch --headless --device cuda:0 \
  --config configs/gazebo_e2e.yaml --num-envs 1 --seed 31 \
  --total-steps 512 --rollout-steps 16 --encoder-chunk-size 4 \
  --bc-init checkpoints/experiment_c_bc.pt \
  --checkpoint-interval 256 --logdir runs/gazebo_e2e_smoke --mode mixed
```

输出必须显示 `parameter_delta > 0` 才算实际完成 CUDA 参数更新。checkpoint 位于：

```text
runs/gazebo_e2e_smoke/checkpoints/best.pt
runs/gazebo_e2e_smoke/checkpoints/last.pt
runs/gazebo_e2e_smoke/checkpoints/step_XXXXXXXXX.pt
```

BC 训练还会在 checkpoint 旁保存 `*.pt.metrics.json`，用于记录损失、耗时和 PyTorch
峰值显存。PPO checkpoint 的 `lineage` 字段记录 ImageNet、BC checkpoint 与验证损失。

## 恢复、正式训练和评估

```bash
python -m aerointercept.gazebo.scripts.train_e2e_ppo \
  --launch --headless --device cuda:0 --num-envs 1 --seed 31 \
  --total-steps 512 --rollout-steps 16 \
  --checkpoint runs/gazebo_e2e_smoke/checkpoints/last.pt --resume \
  --checkpoint-interval 256 --logdir runs/gazebo_e2e_smoke --mode mixed

python -m aerointercept.gazebo.scripts.train_e2e_ppo \
  --launch --headless --device cuda:0 --num-envs 1 --seed 31 \
  --total-steps 1000000 --rollout-steps 32 \
  --checkpoint-interval 10000 --logdir runs/gazebo_e2e_formal --mode mixed

python -m aerointercept.gazebo.scripts.evaluate \
  --launch --headless --device cuda:0 --episodes 10 --mode circle \
  --checkpoint runs/gazebo_e2e_smoke/checkpoints/best.pt \
  --output results/gazebo_e2e_eval.json

tensorboard --logdir runs/gazebo_e2e_smoke --port 6006
```

正式百万步训练前应先完成 512-step 实测并确认 16GB 显存安全。Gazebo/PX4 按真实
时间异步运行，吞吐会明显低于内存内向量化二维环境。

## 并行限制

`GazeboVectorEnv` 接受多个 `--socket`，但每个 socket 必须对应独立的 Gazebo
partition、ROS domain、Micro XRCE 端口和 PX4 进程。当前自动 launcher 只支持一套；
在隔离 launcher 完成并经过相机 buffer 测试前，不允许用重复 socket 冒充并行环境。

## 已测性能

下表为 2026-08-24 旧轻量视觉编码器的基线，不是实验 C 的结果。实验 C 的测量结果
单独记录在 [实验 C 记录](experiment_c.md)，不得混用。

RTX 5070 Ti 16GB、seed 31、单环境、rollout 16、总步数 512、
`encoder_chunk_size=4` 的旧基线：

| 指标 | 实测 |
|---|---:|
| PPO updates | 32 |
| Actor parameter max delta | 0.00672755 |
| 整机 GPU 峰值 | 3284 MB |
| PyTorch allocated 峰值 | 1918 MB |
| 相机 FPS | 7.49 |
| 端到端训练 FPS | 7.26 step/s |
| PPO 更新累计耗时 | 17.46 s |
| 总耗时 | 70.53 s |

恢复测试从 global step 512 正确继续到 528，并重新保存 optimizer、RNG、last 和周期
checkpoint。随机初始化仅训练 512 步的 hit rate 为 0，不应作为收敛结果引用。

## 回归测试

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate AeroIntercept
cd /home/verser/Python/AeroIntercept
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

普通回归测试覆盖模型协议、旧环境和无需启动 Gazebo 的任务逻辑。真实物理链路仍需
使用本文前面的 camera smoke、environment smoke 和 CUDA PPO 冒烟命令验收。

## 保留的轻量工作流

轻量二维环境继续用于快速数据采集、行为克隆和算法回归，不代替 Gazebo 高保真验收：

```bash
python -m aerointercept.training.collect_e2e_data \
  --episodes 1000 --out data/e2e_bc

python -m aerointercept.training.train_e2e_bc \
  --data data/e2e_bc --out checkpoints/e2e_bc.pt

python -m aerointercept.training.train_e2e_ppo \
  --bc-init checkpoints/e2e_bc.pt \
  --out checkpoints/e2e_rl.pt --logdir runs/e2e_ppo

python -m aerointercept.evaluation.eval_e2e \
  --policy checkpoints/e2e_rl.pt --episodes 200 \
  --out results/e2e_rl.csv
```
