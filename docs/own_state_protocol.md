# 图像与 PX4 自身状态协议

2026-09-11 用户批准在 Actor 中增加自身速度和角速度。任务仍是非接触会合，
初始中心距 10 m、成功中心距 ≤0.5 m；目标真值不能作为 Actor 输入。

## 输入与时钟

Actor 输入为 `frames [N,2,3,640,640] uint8 RGB` 和 `self_state [N,6] float32`。
自身状态顺序为 `[vx, vy, vz, roll_rate, pitch_rate, yaw_rate]`，采用机体 FRD，
前三项单位 m/s，后三项 rad/s。模型内部除以 `[8,8,8,2,2,2]`，调用者传物理量。
图像继续采用完整视场 letterbox；部署预处理已与 Gazebo 采集对齐。

状态来自 `/px4_1/fmu/out/vehicle_odometry` 的 `velocity`、`angular_velocity` 和 `q`。
使用 PX4 自身姿态将参考坐标系速度转换至机体系，拒绝未知或不一致的 frame 标识。
不读取目标信息，也不以 Gazebo 模型位姿差分代替自身速度或角速度。

本地 PX4 的 DDS 序列化会把 `timestamp_sample` 转换到主机时钟，相机则使用仿真时钟。
桥接订阅 `/px4_1/fmu/out/timesync_status` 的 DDS `estimated_offset`，将样本转换回
PX4 仿真时钟后，选取不晚于图像的最近测量；最大允许年龄 0.2 s。转换对应本地
PX4 `uxrce_dds_client.cpp` 的 `session->time_offset = -timesync->offset()*1000`。
实测 smoke 的最后样本年龄为 4 ms；新数据首批 5 回合、784 条样本为 0–12 ms。

## 数据与模型

新 NPZ 分片增加 `self_state [T,6]`、`self_state_timestamp_ns [T]`、
`image_timestamp_ns [T]`。训练读取时检查形状、有限数值及因果时序，缺失即拒绝。
清单包含 `self_state_source: px4_vehicle_odometry_body_frd_v1`。
旧纯视觉数据缺失实测角速度，不能通过填零或从真值重建用于新模型训练。

新模型通过单独的自身状态编码层与视觉特征融合，Critic 保持独立的 15 维输入。
普通 checkpoint 恢复严格核对架构。迁移旧视觉权重必须显式指定
`--init-visual-checkpoint`，仅允许新增状态编码与融合层缺失；其余视觉权重必须匹配。
融合层初始化保留视觉路径，之后通过训练学习使用自身状态。

```bash
python -u -m aerointercept.training.train_e2e_bc \
  --backend gazebo --config configs/gazebo_spatial.yaml --data data/noncontact_center_v2 \
  --action-loss-weight 20 --selection-metric total \
  --out checkpoints/noncontact_center_v2_bc.pt --device cuda:0 --epochs 20
```

旧两帧纯视觉实验结果只作历史基线，不作为新观测协议的验收结果。
任务已修正为 `base_link` 机体中心基准；旧模型原点协议（包括 `noncontact_rgb_state`）
的数据也不能通过当前任务检查。当前候选与结果见 [核查记录](noncontact_rendezvous_status.md)。

## 导出与运行

导出格式标识为 `rgb_px4_self_state_v1`；metadata 的 `self_state_input` 明确单位、
坐标系、归一化和时效要求。TorchScript 仅包含 Actor。运行时调用：

```python
runtime.step(rgb_image, yaw, self_state=measured_state,
             self_state_age_seconds=measured_age)
```

其中 `measured_age` 必须在图像与状态使用同一时钟后计算，不能用未转换的时间戳相减。
缺失、非有限、未来或过期的自身状态均会报错，不生成替代飞行指令。
旧纯视觉导出仍支持单图像输入。新协议的单元测试覆盖坐标变换、时效、数据验证、
权重迁移、PPO 输入传递及导出运行时；最终性能仍以真实 Gazebo 独立测试为准。

当前机体中心协议的 BC 候选已导出至 `export/noncontact_geometry_bc/policy.pt`。
两条真实验证轨迹各抽取近、中、远时间点，共 6 组图像与实测自身状态，
导出模型五项输出与原始模型的最大绝对误差均为 0，且没有 Critic 模块。
证据在 `results/noncontact/export_geometry_bc_verification.json`。
这仅验证导出一致性；该 BC 模型的闭环开发评估仍为 0/10，不是已验收模型。
