# 使用指南

操作手册：**环境 → 仿真评估 → 数据采集 → 训练微调**。
背景见 [README](../../README.md)，实验数据与验收基准见 [RESULTS.md](RESULTS.md)。
命令均在工程根目录执行；主配置 `configs/gazebo_feedback.yaml`，先核对本机路径。

## 1. 环境

```bash
conda create -n AeroIntercept python=3.10 -y && conda activate AeroIntercept
pip install -r requirements.txt && pip install -e .
```

仿真依赖：Gazebo Harmonic + PX4 v1.16 SITL + ROS 2 Humble + MicroXRCEAgent v2.4.x
（安装说明见[参考工程](https://github.com/Eaglewzw/Autonomous-Intercept-Drone)）。
评估与 BC 训练 4 GB 显存可跑（训练加 `--batch-size 1 --sequence-length 8`）。

## 2. 仿真与评估

启动方式二选一：`--launch` 由评估器拉起整套仿真（要求无残留进程，结束自动关闭）；
省略 `--launch` 则接入已由 `launch_gazebo.sh` 启动、保持运行的世界。

```bash
# 仿真 + 机载相机第一视角
bash aerointercept/gazebo/scripts/launch_gazebo.sh --mode circle --seed 31
python -m aerointercept.gazebo.scripts.view_camera --display

# 闭环评估（--suite 五场景；--mode 单场景；--trace 记录逐步轨迹）
python -m aerointercept.gazebo.scripts.evaluate \
  --launch --headless --device cuda:0 --suite --episodes 2 --seed 10000 \
  --config configs/gazebo_feedback.yaml --checkpoint assets/models/best.pt \
  --output artifacts/runs/experiments/visual_recheck/evaluation.json --trace
```

评估种子与训练种子必须互不重叠：开发评估固定 seed 10000，正式验收保留 seeds 20000+。

## 3. 数据采集

```bash
# 专家数据：专家全程控制，标签即专家动作
python -m aerointercept.gazebo.scripts.collect_bc_data \
  --launch --headless --device cuda:0 --config configs/gazebo_feedback.yaml \
  --mode circle --episodes 30 --seed 31001 --out artifacts/data/<dataset>/circle

# 纠偏数据：模型参与控制、专家 0.5 混合监督，专治分布偏移
python -m aerointercept.gazebo.scripts.collect_bc_data \
  --launch --headless --device cuda:0 --config configs/gazebo_feedback.yaml \
  --mode circle --episodes 30 --seed 32001 \
  --behavior-checkpoint artifacts/runs/training/<run>/best.pt --expert-weight 0.5 \
  --out artifacts/data/<dataset>/circle
```

- `mixed` 只轮换三种训练场景；八字、启停须按 `--mode` 单独采集。
- 各场景目录用 `aerointercept.gazebo.dataset_integrity.merge_collections` 合并（自动校验合同、查重）。

## 4. 训练与微调

```bash
RUN_DIR="artifacts/runs/training/<run>_$(date +%Y%m%d_%H%M%S)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u -m aerointercept.training.train_e2e_bc \
  --backend gazebo --config configs/gazebo_feedback.yaml \
  --data artifacts/data/<dataset>_merged --init-checkpoint assets/models/best.pt \
  --epochs 8 --patience 3 --batch-size 1 --sequence-length 8 \
  --learning-rate 0.0001 --action-loss-weight 20 \
  --selection-metric action --split mode_visibility --visible-action-only --seed 0 --device cuda:0 \
  --out "$RUN_DIR/best.pt" 2>&1 | tee "$RUN_DIR/train.log"
```

- `--init-checkpoint` 微调会重建优化器；第 0 轮先评估初始化模型，`best_epoch` 为 0 即未超越。
- **离线验证损失改善 ≠ 闭环改善**，候选权重必须通过第 2 节的闭环复测。
- 新验证集不能包含初始化权重已训练的场景；回放旧样本用 `--split-file` 固定划分。
- 纠偏规则：失败回合保留接触前的专家纠正标签（契约项 `behavior.failure_tail`），旧纠偏目录不续采。
- 可选：`end_to_end.bc.keep_all_batch_norm_eval: true` 冻结 BN 统计；
  `end_to_end.auxiliary.near_action_weight` 强调末段控制（默认 1 / 2 m）。

## 5. 模型输入与协议

| 项 | 规格 |
|---|---|
| 图像 | 双帧 RGB，letterbox 640×640，模型内 ImageNet 归一化 |
| 自身状态 | 机体 FRD `[vx,vy,vz,ωx,ωy,ωz]`，模型内除以 `[8,8,8,2,2,2]` |
| 状态来源 | PX4 `vehicle_odometry`，DDS 时钟对齐；取不晚于图像的最近状态，年龄 ≤0.2 s，缺失/过期拒绝 |
| 坐标基准 | `gazebo_base_link_center_enu_to_ned_v2`，不可与旧协议混用 |
| 监督 | Actor 不接收目标真值；独立 Critic 仅 PPO 使用；部署只导出 Actor |
| 时序 | 默认双帧架构；可选 `temporal_memory_steps=16` 加 GRU（PPO 不支持时序模型） |

实现见 [policy.py](../../aerointercept/end_to_end/policy.py)。

## 6. 其他入口与回归测试

| 用途 | 模块 |
|---|---|
| Gazebo 数据采集 | `aerointercept.gazebo.scripts.collect_bc_data` |
| Gazebo PPO | `aerointercept.gazebo.scripts.train_e2e_ppo` |
| BC 吞吐测试 | `aerointercept.gazebo.scripts.benchmark_bc` |
| Actor 导出 | `aerointercept.export_e2e` |

回归测试：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`

## 7. 成功标准（contact_intercept_v1）

- 成功：两机接触，或一步内最小中心距 ≤0.5 m（`gazebo_base_link_center_enu_to_ned_v2` 基准）。
- 失败：撞地面、撞目标之外的物体、无效状态、出画；速度指令上限 8 m/s。
- 接触按机体归属：只有拦截机的碰撞计入回合，目标撞树不影响判定。
- 发生接触的回合在下一回合前重建世界；成功不要求保持时间。
- 历史报告的旧 noncontact_rendezvous_v1 结果不能按本判据比较。
