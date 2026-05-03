import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D

# 自定义 legend handle，单独设置粗细
legend_lines = [
    Line2D([0], [0], color="firebrick", lw=80),  # GlobalRAG, lw=4 是 legend 线宽
    Line2D([0], [0], color="blue", lw=80),       # Step-Search
    Line2D([0], [0], color="black", lw=80)       # Search-R1
]



categories = ["2Wiki", "HotpotQA", "MuSiQue", "WikiHop", "Bamboogle"]

Search_R1 = [23.57, 25.27, 25.93, 28.33, 20.40]
Step_Search = [28.40, 29.00, 31.00, 30.70, 23.73]
GlobalRAG = [32.53, 33.57, 36.53, 33.57, 26.57]

GlobalRAG_raw = [47.4, 35.6, 14.6, 17.5, 47.6]

N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# 闭合曲线

Search_R1_vals = Search_R1 + Search_R1[:1]
Step_Search_vals = Step_Search + Step_Search[:1]
GlobalRAG_vals = GlobalRAG + GlobalRAG[:1]

plt.figure(figsize=(9, 9))
ax = plt.subplot(111, polar=True)

ax.legend(legend_lines, ["GlobalRAG", "Step-Search", "Search-R1"],
          loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=14)

# ------------ 绘制曲线 ------------

ax.plot(angles, GlobalRAG_vals, color="firebrick", linewidth=2.5, label="GlobalRAG")
ax.plot(angles, Step_Search_vals, color="blue", linewidth=2.5, label="Step-Search")
ax.plot(angles, Search_R1_vals, color="black", linewidth=2.5, label="Search-R1")

# ------------ 只填充 GlobalRAG 区域 ------------

ax.fill(angles, GlobalRAG_vals, color="firebrick", alpha=0.15)

# ------------ 标签设置（字号 +4，加粗） ------------

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=18, fontweight="bold")
ax.tick_params(axis='x', pad=20)  # 标签距离中心距离

# ------------ 刻度线设置（虚线 + 加粗 + 灰色） ------------

max_val = max(GlobalRAG)
grid_levels = np.linspace(0, max_val, 5)

# 只显示刻度线，不显示数字
ax.set_yticks(grid_levels[1:])  # 保留刻度线
ax.set_yticklabels([])           # 不显示数字
# 设置径向轴范围
ax.set_ylim(0, max_val)  # max_val = max(GlobalRAG)


# y 轴圆形刻度线

for grid in ax.yaxis.get_gridlines():
    grid.set_linestyle("--")
    grid.set_color("gray")
    grid.set_linewidth(2)

# x 轴角度方向刻度线

for grid in ax.xaxis.get_gridlines():
    grid.set_linestyle("--")
    grid.set_color("gray")
    grid.set_linewidth(2)

# ------------ 添加 GlobalRAG 原始值标签 ------------

# 第1个点
ax.text(
    angles[0]+ np.deg2rad(10),
    GlobalRAG_vals[0] + 2,
    f"{GlobalRAG_raw[0]}",
    color="firebrick",
    fontsize=16,
    fontweight="bold",
    ha="center",
    va="center"
)

# 第2个点
ax.text(
    angles[1] + np.deg2rad(5),
    GlobalRAG_vals[1] + 1.5,
    f"{GlobalRAG_raw[1]}",
    color="firebrick",
    fontsize=16,
    fontweight="bold",
    ha="center",
    va="center"
)

# 第3个点
ax.text(
    angles[2] + np.deg2rad(12),
    GlobalRAG_vals[2] + 1,
    f"{GlobalRAG_raw[2]}",
    color="firebrick",
    fontsize=16,
    fontweight="bold",
    ha="center",
    va="center"
)

# 第4个点
ax.text(
    angles[3] + np.deg2rad(15),
    GlobalRAG_vals[3] + 1,
    f"{GlobalRAG_raw[3]}",
    color="firebrick",
    fontsize=16,
    fontweight="bold",
    ha="center",
    va="center"
)

# 第5个点
ax.text(
    angles[4],
    GlobalRAG_vals[4] + 3,
    f"{GlobalRAG_raw[4]}",
    color="firebrick",
    fontsize=16,
    fontweight="bold",
    ha="center",
    va="center"
)


# ------------ 图例 ------------

ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=14)
ax.spines["polar"].set_visible(False)

plt.tight_layout()
plt.savefig("./radar_normalized.png", dpi=300, bbox_inches='tight')
# plt.show()




