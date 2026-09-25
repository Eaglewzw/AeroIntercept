# 非接触视觉会合：工程核查记录

最近更新：2026-09-12。

**当前坐标基准已升级为 `gazebo_base_link_center_enu_to_ned_v2`。**
SDF 语义解析确认：两架 x500 的 `base_link` 惯性中心在顶层模型原点上方 0.24 m，
相机相对机体中心的 FLU 位移为 `[0.13233, 0, 0.02078]`。原先把顶层模型原点称为
“中心”不准确；姿态不同时两机偏移不能抵消。证据见 `results/noncontact/model_frame_audit.json`。
启动器现在检查解析后的 SDF 坐标，模型变更后不允许静默使用旧外参。

下文旧模型原点协议的数据与评估全部为历史开发记录，不能用于当前中心距验收。
`data/noncontact_spatial` 在投影图像核查发现问题后中止，不能用于新协议训练。
修正中心后的首轮专家 pilot 为 2/3，另 1 回合在 0.529 m 处因“中心出画”终止。
图像中仍有机体可见，因此可见性改为 x500 包围球与相机视锥相交测试
（半径 0.4 m；只判断几何可见性，不声称处理遮挡）。成功半径、相对速度和接触条件不变。
中心处于画面外或 letterbox 黑边内时不施加中心热图监督。
`data/noncontact_center_v2` 已完成：24/24 专家回合无接触成功，共 2,864 帧，
自身状态年龄 0–12 ms；2,461 帧的机体中心位于实际图像内容内，可用于中心热图监督。
数据审计见 `results/noncontact/center_v2_dataset_audit.json`，投影图像核查见
`results/noncontact/center_v2_label_qa.png`。候选 BC 在 22 个训练回合与 2 个验证回合上训练，
日志 `results/noncontact/bc_center_v2.log`，输出 `checkpoints/noncontact_center_v2_bc.pt`。
20 轮训练已完成，耗时 518.54 秒；最佳验证总损失 0.25322（含图像注意力监督，不能与旧总损失直接比较）。
`results/noncontact/center_v2_bc_development.json` 的五模式、每模式 2 回合开发评估已完成：
**0/10 成功、9 fov_lost、1 contact**，另外发生一次复位失败并重启仿真栈。该模型不达标。
验证图像定位 RMSE 为 x=1.63 px、y=1.12 px，但近距离横向速度 RMSE 约 0.44 m/s，
偏航 RMSE 约 0.141 rad/s（`results/noncontact/center_v2_action_audit.json`）。

正在测试 `configs/gazebo_feedback.yaml`：将 RGB 注意力预测的水平坐标经相机内参转换为
偏航反馈，学习头保留残差，BC 和 PPO 均使用同一路径。其输入仍只有 RGB 与自身状态。
`checkpoints/noncontact_center_v2_feedback_init.pt` 仅为显式架构初始化，保留父模型权重，
并未继承父模型的性能指标；不能称为训练或验收完成。五模式 pilot 的输出为
`results/noncontact/center_v2_feedback_pilot.json`。
该 pilot 已完成：0/5，全部 fov_lost、无接触。目标仍在图像内时预测位置与几何投影基本一致，
但持续的微小下降指令会累积垂直误差，使目标从画面上方离开。
`data/noncontact_center_v2_corrective` 采集到 19 回合后中止（17 成功、2 接触），行为策略为上述反馈初始化模型；
远距离专家混合权重 0.3，2 m 内渐变为完全专家控制，标签始终为专家修正。
后续 BC 使用动作轴权重 `[1, 1, 16, 4]`（前、右、下、偏航），特别约束垂直漂移。
这是训练损失调整，不改变任务的任何成功阈值。
两次接触分别为本机机身与目标起落架横杆、以及本机旋翼与目标起落架支柱。
专家参考位置已从 `[-0.45,0,0.20]` 改为 `[-0.46,0,0.14]` m，提前制动时间从
0.5 s 改为 0.8 s，近距离接近增益从 0.5 改为 0.3。新参考位置的名义中心距为
0.481 m，平飞且相机朝向目标时，目标中心在相机垂直视场内。
`data/noncontact_geometry_pilot` 已完成该修正的验证：3/3 无接触成功，共 726 帧，
包括此前接触的 circle seed 804。`data/noncontact_geometry_corrective` 已完成：
24/24 无接触成功，circle、sinusoidal、random_walk 各 8 回合，共 5,227 帧；
自身状态年龄 0–16 ms，5,222 帧的中心位于图像内容内。审计见
`results/noncontact/geometry_dataset_audit.json`。当前用此数据微调
`checkpoints/noncontact_geometry_bc.pt`（日志 `results/noncontact/bc_geometry.log`），
学习率 1e-4、20 轮，按含动作轴权重的验证动作误差选择 checkpoint。
BC 20 轮已完成，最佳加权动作验证误差为 0.002620（第 18 轮），耗时 917.27 秒。
独立开发评估 `results/noncontact/geometry_bc_development.json` 为 **0/10，8 次视野丢失、2 次接触**，
无复位恢复事件。近距离垂直位置控制仍不可靠；模仿误差不代表闭环成功。
PPO 已接入相同的训练期图像位置监督，并用训练奖励鼓励在目标下方保留 0.14 m 垂直间隔，
该奖励不改变 Actor 输入或成功条件。首轮真实 Gazebo PPO 完成 89 次优化（2,848 步），
下一批采样到 2,880 步时暂停/恢复服务失败，原定 4,096 步没有完成。
32 个已记录回合全部因视野丢失终止；这不代表整个运行期间没有接触：桥接日志也记录了
回合边界之外的接触，日志保存在 `results/noncontact/ppo_geometry_failure/`。
最后定期 checkpoint 为 `runs/noncontact_geometry_ppo/checkpoints/step_000002560.pt`。
新增暂停状态请求的有界重试：只重试无应答，显式拒绝立即失败；持续失败仍终止运行。
该 checkpoint 的五模式独立开发评估已完成：**0/5，4 次视野丢失、1 次接触**。
证据 `results/noncontact/geometry_ppo_2560_development.json`；其中 random_walk
实际复位距离 9.7884 m，验收门限正确将其判为不满足 10±0.2 m。
原因是复位只检查第一张图像对应状态，随后第二张图像可能越界；现要求连续两张
真实图像对应状态均满足全部复位条件，新增回归测试覆盖第二帧越界。
恢复训练使用独立目录 `runs/noncontact_geometry_ppo_resume`、seed 1200，
从 2,560 步 checkpoint 补训 1,536 步，保留旧失败日志而不覆盖。
该补训已正常完成：48 次优化、550.77 秒，最终累计 4,096 步。
24 个已记录回合为 **19 次视野丢失、5 次接触、0 次成功**；全部实际起始中心距
在 9.8022–10.1890 m 内。运行中一次暂停无应答经重试恢复；接触后按规则重启自有仿真栈。
最终模型 `runs/noncontact_geometry_ppo_resume/checkpoints/step_000004096.pt`
的独立五模式开发评估已完成：**0/5，3 次视野丢失、2 次接触**。
证据 `results/noncontact/geometry_ppo_4096_development.json`；全部实际起始中心距
满足 10±0.2 m，无复位恢复事件，确认连续两帧复位修复有效。
两个接触回合虽然进入 0.5 m，也严格判为失败。模型未通过性能门限，因此未启动
100 回合正式验收，也未使用保留的最终测试 seeds 20000+。
当前全套测试 100 项通过；软件测试通过不等于飞行验收通过。
本轮训练及评估进程均已结束。下一阶段需要解决策略近距离垂直控制和目标出画问题；
当前证据不支持仅延长相同 PPO 配置就能达到验收要求。
这是专家轨迹调整；初始距离、成功半径、速度保持和无接触判据没有改变。
尚无此最终协议的已验收模型。

当前按用户批准的“图像＋飞控自身速度、角速度”方案推进，以下为旧模型原点协议的实验历史。
接口、数据和时钟转换说明见 [自身状态协议](own_state_protocol.md)。以下纯视觉 BC/PPO
结果属于已完成的历史开发实验，不代表新方案已通过验收。

旧模型原点协议采集 `data/noncontact_rgb_state` 已完成 48 回合、6,151 帧：45 hit、3 contact。
全部六维自身状态数值有效，样本相对图像年龄为 0–16 ms；三条接触失败不会进入 BC。
新模型训练日志为 `results/noncontact/bc_rgb_state.log`，输出
`checkpoints/noncontact_rgb_state_bc.pt`，使用显式视觉权重迁移和 20 倍动作损失权重，
按动作验证误差选择 checkpoint。20 轮训练完成，最佳动作验证损失为 0.00070785，
耗时 977.43 秒。五种模式各 2 回合的开发评估为 **0/10，全部 fov_lost，无接触**；
证据在 `results/noncontact/rgb_state_bc_development.json`。模仿误差降低没有转化为闭环成功。

单回合逐步诊断见 `results/noncontact/rgb_state_bc_trace.trace.jsonl`：目标向画面右侧移动时，
策略仍持续向左横移和偏航。随后尝试采集 `data/noncontact_spatial`，在图像标签核查时发现
中心参考系问题并中止。候选配置 `configs/gazebo_spatial.yaml` 保留相同的 Actor 输入，
已同步修正中心基准与部分可见判定。投影标签只用于 BC 注意力监督，不能作为部署输入。
候选尚未完成闭环验证。

## 任务定义

用户确认任务为不接触、不损伤目标的视觉跟踪与会合，初始距离从先前提出的 30 m
修正为 10 m，希望成功距离阈值从 0.8 m 改为 0.5 m。

已确认 **0.5 m 指两机中心距**。当前 Gazebo 配置已采用初始中心距 10 m
（复位容差 ±0.2 m），成功条件为中心距 ≤0.5 m、相对速度 ≤0.5 m/s、连续保持
至少 0.3 s，且没有接触或其他物理失败。`hit` 字段保留为旧接口的成功别名，
不表示碰撞成功。任何接触、无效状态、触地等失败优先于距离成功。

旧桥接的出生坐标 ENU→NED 偏移存在轴向错误。真实仿真核查中原始模型中心距
4.99083 m，而旧桥接报告 11.09203 m。旧 -π/2 相机挂载补偿也不符合本地
OakD SDF。现在使用原始 Gazebo 模型位姿统一转换到 NED，真实姿态加相机偏移
计算视场；EKF 的局部原点只在复位控制时转换。修正后的原始位姿和桥接距离
分别为 9.94299 m 与 9.94303 m（异步采样差异）。旧数据和 checkpoint 不再允许
作为新任务的训练恢复或验收依据。

接触传感器采用工程生成的 x500/OakD 副本，不修改 PX4 原始模型。Contact 插件
必须在 UserCommands 之后加载，才能看到 PX4 动态生成的模型。参见
[Gazebo 接触传感器说明](https://gazebosim.org/docs/harmonic/sensors/) 和
[模型位姿发布说明](https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1PosePublisher.html)。

## 本次已完成

- 新采集分片将真实终止结果与观测原子保存，避免分片已写入但进度文件未写入时丢失结果。
- 恢复旧数据时不再根据终止前图像/标签猜测成功；缺失终止证据的结果标为 `unknown`。
- 新采集记录任务、专家、动作、相机、标签及模式计划。恢复时要求配置一致，避免不同
  距离判据或标签协议混入同一个数据集。旧数据保留，使用新目录开始新任务。
- 相机快照携带原始图像时间戳；环境记录每回合墙钟耗时和传感器时间跨度。
- 评估保留逐回合证据、实际 reset 距离、任务配置及成功率 95% Wilson 区间。
  删除固定 `步数 / 20` 的耗时估算；缺少传感器时间戳时报告 `null`。
- PPO 日志也改为使用已测回合仿真耗时。

## 实测

| 项目 | 结果 |
|---|---|
| 修改前回归测试 | 57 passed |
| 修改后回归测试 | 94 passed；仅现有 TorchScript 弃用警告 |
| 主机 GPU | RTX 5070 Ti，16303 MiB；沙箱外 `nvidia-smi` 检查通过 |
| Gazebo/PX4 相机验证 | 4 帧，uint8，`[4,3,640,640]`；序号 616、623、631、638 |
| 相机连续帧差 | 均值 6.47946；目标在视场内 |
| 相机验证时距离 | 6.28189 m；仅相机链路测试，不能称为 10 m 会合验收 |

相机画面：`results/gazebo_camera_validation.png`。相机测试结束后已关闭本次启动的进程。

只读审计 `data/gazebo_experiment_c_500`：297 回合、18739 帧；circle 167 回合，
sinusoidal 130 回合，没有 random_walk 数据。167 回合的旧恢复成功标签缺少真实
终止证据，应视为未知；其余记录为旧判据下的 122 hit、8 fov_lost。
审计输出在 `results/dataset_integrity_audit.json`；原始数据未改写。

## 新任务物理验证

- 双机环境 smoke：16 个物理交互步，复位中心距 9.91170 m，图像/状态协议通过，
  两机均已解锁且处于 Offboard，11 个接触传感器已加载。
- hover 专家 pilot：2/2 成功；最终中心距 0.45696 / 0.42815 m；相对速度
  0.08666 / 0.08347 m/s；连续保持 0.38 / 0.44 s；均无接触。
- 上述为专家与任务逻辑验证，**不是学习策略的性能**。
- 混合专家采集 v5：6/6 成功，覆盖 circle、sinusoidal、random_walk；触发中心距
  0.425–0.495 m，接触计数均为 0。数据位于 `data/noncontact_mixed_v5`，仅用于
  训练链路 smoke，正式数据仍需扩大样本量。
- 新的动作后快照等待至少一个控制周期的仿真时间，避免返回动作前排队的旧图像；
  PPO 参数更新期间暂停物理仿真，防止更新耗时改变采样任务。

## 仍需完成

1. 扩大三种运动场景的正式专家数据采集。
2. 新数据上的视觉 BC、PPO 训练与恢复验证。
3. 独立随机种子和留出运动形状（figure_eight、stop_go）的真实 Gazebo 评估。
4. 达标前不报告项目训练完成或泛化达标。

## 正式训练与验收进度（2026-09-11）

`data/noncontact_formal/manifest.json` 已记录完整 48 回合、6,681 帧。
circle、sinusoidal 各 15/16 成功，random_walk 为 16/16，共 46/48（95.8%）。
两条失败均为接触；这仍是专家结果。正式 BC 使用 46 条有明确成功证据且接触计数为零的
演示，按完整回合分为 41 条训练、5 条验证。缺失结果的演示也不进入 BC。
训练与验证分片名单写入新 checkpoint 和指标文件。

小样本 BC 的 20 轮训练已完成，最佳验证损失 0.00692；正式 BC 也已完成。
`bc_formal.log` 是初次运行，因重复解压耗时被中止；当前运行日志为
`results/noncontact/bc_formal_cached.log`，输出 `checkpoints/noncontact_bc_formal_cached.pt`。
正式 BC 最佳验证损失为 0.026714，20 轮耗时 1106.3 秒。
缓存调整只改变读取方式，既不增加示范，也不改变训练目标。

工程验收采用五种运动（circle、sinusoidal、random_walk、figure_eight、stop_go）
各至少 20 回合，合计至少 100 回合；总体成功率至少 90%，各模式至少 80%，
接触数为零。reset 中心距必须在 10±0.2 m，成功触发中心距必须 ≤0.5 m。
原有低相对速度、持续保持和终止后无接触验证继续适用。
评估报告自动输出 `acceptance.checks`，小样本 pilot 不会通过验收。
开发评估从 seed 10000 开始，最终验收预留 seed 20000 及以上，最终 seed 不用于调参。
成功率同时报告 Wilson 95% 区间；单个场景测试通过不代表所有未见运动都能成功。

### 开发闭环结果：尚未通过验收

| 模型 | 开发回合 | 成功 | 丢失视野 | 接触 |
|---|---:|---:|---:|---:|
| 正式 BC | 10 | 0 | 8 | 2 |
| PPO global step 512 | 10 | 0 | 10 | 0 |

逐回合证据分别在 `results/noncontact/bc_development.json` 与
`results/noncontact/ppo_development.json`。均使用开发种子 10000 开始，五种运动各 2 回合。
两项结果均不达标，最终验收种子尚未使用。

PPO 首次运行到 320 步时在复位阶段遇到 Gazebo ODE 碰撞断言崩溃，
从已保存的 256 步恢复后完成到 512 步；恢复运行 Actor 参数最大变化 0.00147256，
整机 GPU 峰值约 3112 MiB，训练回合成功率仍为零。复位失败现在可重启自有仿真栈，
不伪造失败时的图像，也不修改已结束回合的结果。

动作加权对照实验正在执行：`results/noncontact/bc_action_v2.log`。
前一份 `bc_action.log` 因 DotDict 的属性访问返回字典副本，权重写入未生效，已中止，
不应作为动作加权实验引用。修正版通过嵌套字典索引写入，日志总损失已体现 20 倍动作权重。

纠偏示范采集使用 `--behavior-checkpoint` 和 `--expert-weight`，仅在采集时混合
视觉策略与专家控制。专家权重从远距离配置值逐渐上升，到中心距 2 m 内完全使用专家；
存储的动作标签始终是专家修正。清单记录行为 checkpoint 的 SHA256 和混合协议。
评估 Actor 不调用这些混合控制入口。纠偏数据保存在单独目录，不能混称为自主策略成功。
