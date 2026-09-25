# 使用指南

从工程根目录执行命令，先激活已安装依赖的 `AeroIntercept` 环境。
主配置为 `configs/gazebo_feedback.yaml`；完整物理仿真依赖 Gazebo Harmonic、PX4 SITL、ROS 2 Humble。

## 本地仿真与评估

启动仿真；另开终端运行相机查看器：

```bash
bash aerointercept/gazebo/scripts/launch_gazebo.sh --mode circle --seed 31
python -m aerointercept.gazebo.scripts.view_camera --display
```

结束手动仿真后，使用保留基线进行五模式、各 2 回合开发评估：

```bash
python -m aerointercept.gazebo.scripts.evaluate \
  --launch --headless --device cuda:0 --suite --episodes 2 --seed 10000 \
  --config configs/gazebo_feedback.yaml \
  --checkpoint artifacts/runs/remote_20260924/best.pt \
  --output artifacts/runs/remote_20260924/recheck/evaluation.json --trace
```

## 新建 BC 训练

以下为本地新路径；从基线权重初始化后再训练 500 轮，优化器重新建立。
已有服务器任务无需重复启动。

```bash
RUN_DIR="artifacts/runs/bc_500_$(date +%Y%m%d_%H%M%S)"
mkdir -p artifacts/runs
mkdir "$RUN_DIR" || exit 1
nohup python -u -m aerointercept.training.train_e2e_bc \
  --backend gazebo --config configs/gazebo_feedback.yaml \
  --data artifacts/data/noncontact_geometry_corrective \
  --init-checkpoint artifacts/runs/remote_20260924/best.pt \
  --epochs 500 --batch-size 2 --sequence-length 16 \
  --learning-rate 0.0001 --action-loss-weight 20 \
  --selection-metric action --seed 0 --device cuda:0 \
  --out "$RUN_DIR/best.pt" > "$RUN_DIR/train.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/train.pid"
echo "$RUN_DIR"
```

日志为 `train.log`，最佳验证动作模型为 `best.pt`。同次实验的权重、日志、评估与导出
放在一个运行目录，数据集放 `artifacts/data/`；数据与模型不随 Git 分发。

## 查看服务器现有任务

服务器保持部署时的旧路径，本地整理未改动远端任务：

```bash
ssh root@36.150.116.206 -p 33747
cd /root/AeroIntercept_remote_20260924
cat runs/bc_500_20260925_053949_UTC/status.json
tail -f runs/bc_500_20260925_053949_UTC/train.log
```

该目录的 `best.pt` 为本轮最佳权重。服务器环境为 `.venv/bin/python`，已验证 ROCm BC；
Gazebo/PX4 闭环评估在本地完成。同步工程时保留正在写入的远端运行目录。

## 模型输入与协议

- 图像：两张完整 RGB，letterbox 为 640×640，模型内做 ImageNet 归一化。
- 自身状态：机体 FRD `[vx, vy, vz, ωx, ωy, ωz]`，单位 m/s、rad/s；模型内除以 `[8,8,8,2,2,2]`。
- 状态来源：PX4 `vehicle_odometry`。用 DDS `timesync_status` 将测量时间转换至仿真时钟，
  选择不晚于图像的最近状态；年龄不得超过 0.2 s。缺失、非有限、未来或过期状态均拒绝。
- 坐标基准：`gazebo_base_link_center_enu_to_ned_v2`；旧模型原点协议数据不可直接混用。
- Actor 不接收目标真值；独立 Critic 仅在 PPO 使用。部署仅导出 Actor。
- BC 的 `sequence_length=16` 是样本分组，实际视觉历史只有两帧。

实现见 [policy.py](../../aerointercept/end_to_end/policy.py)，
[SVG 架构图](figures/current_model_architecture.svg)。重新绘图：
`python assets/docs/figures/draw_current_model.py`（需要 Matplotlib 和 Noto Sans CJK 字体）。

## 其他入口

`python -m <模块> --help` 查看参数：

| 用途 | 模块 |
|---|---|
| Gazebo 数据采集 | `aerointercept.gazebo.scripts.collect_bc_data` |
| Gazebo PPO | `aerointercept.gazebo.scripts.train_e2e_ppo` |
| BC 吞吐测试 | `aerointercept.gazebo.scripts.benchmark_bc` |
| Actor 导出 | `aerointercept.export_e2e` |

回归测试：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`。
实验结果及验收条件见 [EXPERIMENTS.md](EXPERIMENTS.md)。
