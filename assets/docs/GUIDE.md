# 使用指南

本文是 AeroIntercept 的**操作手册**，回答"怎么跑起来、怎么迭代模型"：
环境准备、闭环仿真与评估、数据采集、BC 训练微调，以及模型输入协议与任务判定标准。
项目背景与最新成绩见 [README](../../README.md)；历次实验数据与验收基准见 [RESULTS.md](RESULTS.md)。

所有命令在工程根目录执行，先激活已安装依赖的 `AeroIntercept` conda 环境。
主配置为 `configs/gazebo_feedback.yaml`，首次使用请核对其中的本机路径；
完整物理仿真依赖 Gazebo Harmonic、PX4 SITL 与 ROS 2 Humble（版本与安装说明见
[参考工程](https://github.com/Eaglewzw/Autonomous-Intercept-Drone)）。

## 1. 环境准备

```bash
conda create -n AeroIntercept python=3.10 -y
conda activate AeroIntercept
pip install -r requirements.txt && pip install -e .
```

评估推理与 BC 训练在 4 GB 显存的 GPU 上即可运行
（BC 训练需 `--batch-size 1 --sequence-length 8` 以控制激活显存）。

## 2. 本地仿真与评估

闭环评估有两种启动方式，二者只能占其一：

- `--launch`：由评估器拉起整套仿真（Gazebo、双机 PX4、MicroXRCEAgent、相机桥），
  要求当前无残留仿真进程，评估结束后自动关闭全部子进程；
- 省略 `--launch`：接入一个已由 `launch_gazebo.sh` 启动并保持运行的世界，
  适合配合机载相机查看器边看边评。

启动仿真；另开终端运行相机查看器：

```bash
bash aerointercept/gazebo/scripts/launch_gazebo.sh --mode circle --seed 31
python -m aerointercept.gazebo.scripts.view_camera --display
```

结束手动仿真后，使用仓库保留权重进行五场景、各 2 回合开发评估：

```bash
python -m aerointercept.gazebo.scripts.evaluate \
  --launch --headless --device cuda:0 --suite --episodes 2 --seed 10000 \
  --config configs/gazebo_feedback.yaml \
  --checkpoint assets/models/best.pt \
  --output artifacts/runs/experiments/visual_recheck/evaluation.json --trace
```

`--suite` 覆盖全部五种目标运动（三种训练场景 + 两种留出场景），`--mode` 可单跑一种。
留出场景检验模型对**未见运动形状**的泛化；独立评估种子检验同形状**新实例**的泛化。
若报"已有 PX4/Gazebo 进程运行"，先清理残留进程再试（`--launch` 拒绝双开）。

## 3. 数据采集

BC 数据由特权专家（true state）标注，执行动作可以是专家或模型：

```bash
# 专家数据：专家全程控制，标签即专家动作
python -m aerointercept.gazebo.scripts.collect_bc_data \
  --launch --headless --device cuda:0 \
  --config configs/gazebo_feedback.yaml \
  --mode circle --episodes 30 --seed 31001 \
  --out artifacts/data/<dataset>/circle

# 纠偏数据：模型参与控制、专家混合监督，专治分布偏移
python -m aerointercept.gazebo.scripts.collect_bc_data \
  --launch --headless --device cuda:0 \
  --config configs/gazebo_feedback.yaml \
  --mode circle --episodes 30 --seed 32001 \
  --behavior-checkpoint artifacts/runs/training/<your_run>/best.pt \
  --expert-weight 0.5 \
  --out artifacts/data/<dataset>/circle
```

要点：

- `mixed` 模式只轮换三种训练场景；八字与启停必须按 `--mode` 单独采集。
- 各场景写入独立目录后，用 `aerointercept.gazebo.dataset_integrity.merge_collections`
  合并为一个训练数据集（自动校验采集合同一致性并查重）。
- 采集种子须与评估种子互不重叠：开发评估固定 seed 10000，正式验收保留 seeds 20000+。
- 训练权重与日志放在 `artifacts/runs/training/`，闭环报告与轨迹放在
  `artifacts/runs/experiments/`，数据集放 `artifacts/data/`。这些目录不随 Git 分发，
  随仓库提供的保留权重只有 `assets/models/best.pt`。

## 4. 训练与微调

创建独立运行目录后启动训练；`--init-checkpoint` 微调会重新建立优化器，
第 0 轮先在当前验证集上评估初始化模型，只有改善指定验证指标的后续轮次才会替换它，
`best_epoch` 为 0 表示微调未优于初始化模型。

```bash
RUN_DIR="artifacts/runs/training/<your_run>_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u -m aerointercept.training.train_e2e_bc \
  --backend gazebo --config configs/gazebo_feedback.yaml \
  --data artifacts/data/<dataset>_merged \
  --init-checkpoint assets/models/best.pt \
  --epochs 8 --patience 3 --batch-size 1 --sequence-length 8 \
  --learning-rate 0.0001 --action-loss-weight 20 \
  --selection-metric action --split mode_visibility --visible-action-only --seed 0 --device cuda:0 \
  --out "$RUN_DIR/best.pt" 2>&1 | tee "$RUN_DIR/train.log"
```

日志为 `train.log`，最佳验证动作模型为 `best.pt`，逐轮诊断为 `best.history.jsonl`。
更换数据集时，新验证集不能包含初始化权重已训练的场景。

离线验证损失的改善不代表闭环改善：每个候选权重都应通过第 2 节的闭环复测再决定是否保留。

训练细则：

- 少量纠偏数据可在运行配置中设置 `end_to_end.bc.keep_all_batch_norm_eval: true`，
  冻结整个 Actor 的 BatchNorm 运行统计，仿射参数仍参与训练。默认关闭，保留从头训练的行为。
- 纠偏采集的 `--behavior-checkpoint <原权重> --expert-weight 0.5`：执行动作由模型与专家混合，
  监督标签仍为专家动作。模型参与控制的失败回合会保留接触前、有效观测上的专家纠正标签；
  失败的是执行动作，不应据此删除不同的专家标签。纯专家控制失败时仍排除最后一秒。
  此规则记入采集契约的 `behavior.failure_tail`，旧纠偏目录不能按新规则续采，应新建目录。
- 设置 `--expert-weight 0` 可记录纯模型执行时的专家纠正标签。
- 合并旧训练样本做回放时，使用 `--split-file split.json` 固定划分，文件包含 `train` 和
  `validation` 两个文件名列表，须不重复且覆盖整个数据集；初始化模型见过的旧样本只能放入训练集。
- 需要强调末段控制时，可配置 `end_to_end.auxiliary.near_action_weight`（默认 1）与
  `near_action_radius_m`（默认 2 米）。该权重只用于已记录距离对应的训练/验证动作损失，
  仍遵循有效性掩码和 `--visible-action-only` 设置；部署动作不增加距离约束。

## 5. 模型输入与协议

- 图像：两张完整 RGB，letterbox 为 640×640，模型内做 ImageNet 归一化。
- 自身状态：机体 FRD `[vx, vy, vz, ωx, ωy, ωz]`，单位 m/s、rad/s；模型内除以 `[8,8,8,2,2,2]`。
- 状态来源：PX4 `vehicle_odometry`。用 DDS `timesync_status` 将测量时间转换至仿真时钟，
  选择不晚于图像的最近状态；年龄不得超过 0.2 s。缺失、非有限、未来或过期状态均拒绝。
- 坐标基准：`gazebo_base_link_center_enu_to_ned_v2`；旧模型原点协议数据不可直接混用。
- Actor 不接收目标真值；独立 Critic 仅在 PPO 使用。部署仅导出 Actor。
- BC 的 `sequence_length=16` 是样本分组，实际视觉历史只有两帧。
- 可选 `model.temporal_memory_steps=16` 在每个时刻融合图像与自身状态后增加 GRU，
  用 `--init-temporal-checkpoint` 迁移，并令训练序列长度一致。评估与运行时在回合开始清空记忆；
  当前 PPO 的打散帧更新不支持此时序模型。默认配置仍为原两帧架构。

实现见 [policy.py](../../aerointercept/end_to_end/policy.py)，
[交互式架构图](figures/current_model_architecture.html) · [SVG 架构图](figures/current_model_architecture.svg)。论文风格静态图（PNG / SVG / PDF）：
`python assets/docs/figures/draw_current_model.py`（需要 Matplotlib）。
交互图编辑同目录 JSON 后运行 `python assets/docs/figures/draw_interactive_model.py`（需要 archify、Node、Playwright 和 Chromium；会同时导出交互风格 PNG / SVG，如需论文风格请再运行静态绘图脚本）。

## 6. 其他入口与回归测试

`python -m <模块> --help` 查看参数：

| 用途 | 模块 |
|---|---|
| Gazebo 数据采集 | `aerointercept.gazebo.scripts.collect_bc_data` |
| Gazebo PPO | `aerointercept.gazebo.scripts.train_e2e_ppo` |
| BC 吞吐测试 | `aerointercept.gazebo.scripts.benchmark_bc` |
| Actor 导出 | `aerointercept.export_e2e` |

回归测试：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`。

## 7. 接触式拦截成功标准（contact_intercept_v1）

Gazebo 任务只有一种语义：两机中心在一步内的最小距离 ≤0.5 m，或检测到两机之间的接触，即成功。
距离在 `gazebo_base_link_center_enu_to_ned_v2` 基准上测量，配置加载时会校验这一基准与接触监视器。
拦截机撞地面、撞到目标之外的物体、无效状态和越界都是失败，命中不会覆盖它们；速度指令上限为 8 m/s。
接触按机体归属：只有拦截机参与的碰撞才计入本回合；目标自己撞到树木或地面时记入
`target_scenery_contact_count` 供审计，不会让拦截回合以碰撞结束。
接触次数与目标接触继续如实记录：撞到目标本身就是成功，撞到别处才是失败。
任何发生过接触的回合（成功拦截通常就是接触）都会在下一回合前重建世界，避免从坠毁状态复位。
成功不再要求相对速度或保持时间，成功之后也不再执行稳定性验证。
历史报告中的旧 noncontact_rendezvous_v1 结果按当时的判据统计，不能直接当作本标准的成功率。
