"""Render the current RGB + own-state policy as a compact, editable diagram."""
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as LinePath

OUTPUT = Path(__file__).resolve().parent
font = subprocess.check_output(
    ["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True
).strip()
font_manager.fontManager.addfont(font)
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=font).get_name(),
    "svg.fonttype": "path",
    "axes.unicode_minus": False,
})
fig = plt.figure(figsize=(16, 8.2), dpi=120, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 1600), ylim=(820, 0))
ax.axis("off")
INK, MUTED, BLUE, AMBER = "#172b45", "#64748b", "#2b63a0", "#aa701c"


def text(x, y, value, size=14, color=INK, bold=False, align="center"):
    ax.text(x, y, value, ha=align, va="center", fontsize=size,
            color=color, weight="bold" if bold else "normal", linespacing=1.5)


def box(x, y, w, h, title, detail, color=BLUE, fill="#f5f8fc"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=9",
        linewidth=1.2, edgecolor=color, facecolor=fill,
    ))
    text(x+w/2, y+29, title, 15, color, bold=True)
    text(x+w/2, y+(h+33)/2, detail, 12)


def arrow(points, color=BLUE):
    path = LinePath(points, [LinePath.MOVETO]+[LinePath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>",
                               mutation_scale=13, linewidth=1.5, color=color))


text(40, 40, "AeroIntercept  /  模型架构", 25, bold=True, align="left")
text(40, 79, "当前配置：gazebo_feedback.yaml", 12, MUTED, align="left")
text(40, 190, "ACTOR  ·  运行时控制", 14, BLUE, bold=True, align="left")

# Runtime policy: each frame is pooled before the two-token Transformer.
box(40, 240, 180, 110, "双帧 RGB", "2 × 3 × 640 × 640\n全画面 letterbox")
box(260, 240, 240, 110, "视觉编码", "ResNet-18 多尺度\n空间注意力 + 位置特征")
box(540, 240, 200, 110, "双帧 Transformer", "2 层 · 2 个 token\n输出 128 维")
box(780, 240, 180, 110, "特征融合", "拼接视觉与自身状态\n128 + 64 → 128")
box(1000, 240, 230, 110, "动作头", "MLP 128 → 128 → 4\n偏航叠加反馈，再 tanh")
box(1270, 240, 290, 110, "速度指令 → PX4", "前 / 右 / 下速度 + 偏航角速度\n缩放、限幅、转换为 NED")
for right, left in [(220, 260), (500, 540), (740, 780), (960, 1000), (1230, 1270)]:
    arrow([(right, 295), (left, 295)])

# Feedback uses image attention, never simulator target truth.
box(650, 108, 360, 86, "图像偏航反馈", "当前帧注意力的水平中心 → 偏航反馈量", AMBER, "#fffaf0")
arrow([(380, 240), (380, 151), (650, 151)], AMBER)
arrow([(1010, 151), (1115, 151), (1115, 240)], AMBER)

box(40, 427, 310, 105, "飞控自身状态", "机体 FRD 三轴速度 + 三轴角速度\n当前时刻 6 维测量")
box(410, 427, 310, 105, "状态编码", "按物理量尺度归一化\nMLP 6 → 64 → 64")
arrow([(350, 479), (410, 479)])
arrow([(720, 479), (870, 479), (870, 350)])
text(1000, 435, "辅助输出：未来位置、风险、可见性", 13, MUTED, align="left")
text(1000, 469, "来自融合特征；通过训练标签监督", 12, MUTED, align="left")

# Independent PPO value branch; not part of the deployed control path.
ax.add_patch(FancyBboxPatch(
    (40, 596), 1520, 130, boxstyle="round,pad=0,rounding_size=9",
    facecolor="#f6f6f6", edgecolor="#d6d9dd", linewidth=1,
))
text(65, 623, "仅 PPO 训练使用", 13, MUTED, bold=True, align="left")
text(230, 673, "15 维仿真状态", 15)
text(780, 673, "Critic MLP：15 → 256 → 256 → 1", 15)
text(1360, 673, "状态价值 V(s)", 15)
arrow([(360, 673), (525, 673)], MUTED)
arrow([(1070, 673), (1250, 673)], MUTED)

text(40, 765, "实际视觉历史为 2 帧；BC 的 sequence_length=16 只是训练分组。", 12, MUTED, align="left")
text(40, 796, "目标真值用于训练监督与评估，不进入 Actor；部署仅导出 Actor。", 12, MUTED, align="left")

for extension in ("png", "svg"):
    target = OUTPUT/f"current_model_architecture.{extension}"
    fig.savefig(target, dpi=120, facecolor="white")
    print(target)
plt.close(fig)
