import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 原始数据（Qwen2.5-3B-Instruct）
# -----------------------------
data = {
    "HotpotQA": {
        "Search-R1": {"EM": 31.3, "Token": 107.5},
        "StepSearch": {"EM": 28.9, "Token": 245.8},
        "GlobalRAG": {"EM": 32.9, "Token": 391.4},
    },
    "2WikiMultihopQA": {
        "Search-R1": {"EM": 31.3, "Token": 129.2},
        "StepSearch": {"EM": 31.9, "Token": 246.3},
        "GlobalRAG": {"EM": 42.3, "Token": 385.7},
    },
    "Musique": {
        "Search-R1": {"EM": 7.7, "Token": 126.1},
        "StepSearch": {"EM": 9.5, "Token": 285.9},
        "GlobalRAG": {"EM": 10.8, "Token": 430.9},
    },
    "Bamboogle": {
        "Search-R1": {"EM": 28.0, "Token": 106.3},
        "StepSearch": {"EM": 32.0, "Token": 237.0},
        "GlobalRAG": {"EM": 37.6, "Token": 343.6},
    },
    "WikiHop": {
        "Search-R1": {"EM": 8.2, "Token": 125.3},
        "StepSearch": {"EM": 11.2, "Token": 188.9},
        "GlobalRAG": {"EM": 12.9, "Token": 300.0},
    },
}

# -----------------------------
# 颜色（方法）
# -----------------------------
method_colors = {
    "Search-R1": "#1f77b4",
    "StepSearch": "#ff7f0e",
    "GlobalRAG": "#2ca02c",
}

# -----------------------------
# 形状（数据集）
# -----------------------------
dataset_markers = {
    "HotpotQA": "o",
    "2WikiMultihopQA": "s",
    "Musique": "^",
    "Bamboogle": "D",
    "WikiHop": "P",
}

# -----------------------------
# 气泡与显示参数
# -----------------------------
BASE_SIZE = 4800
LEGEND_SCALE = 100
LEGEND_SIZE = BASE_SIZE / LEGEND_SCALE

X_MARGIN_RATIO = 0.15
Y_MARGIN_RATIO = 0.15

# -----------------------------
# 开始绘图（单一 Figure）
# -----------------------------
plt.figure(figsize=(9, 7))

xs_all, ys_all = [], []

for dataset, methods in data.items():
    # Search-R1 作为该数据集内的效率基准
    sr1 = methods["Search-R1"]
    sr1_eff = sr1["EM"] / np.log(sr1["Token"])

    for method, values in methods.items():
        token = values["Token"]
        em = values["EM"]

        x = np.log10(token)
        y = em

        eff = em / np.log(token)
        bubble_size = BASE_SIZE * (eff / sr1_eff)

        xs_all.append(x)
        ys_all.append(y)

        plt.scatter(
            x,
            y,
            s=bubble_size,
            color=method_colors[method],
            marker=dataset_markers[dataset],
            alpha=0.75,
            edgecolors="black",
            linewidth=0.8,
            zorder=3,
        )

# -----------------------------
# 轴范围扩展，避免气泡越界
# -----------------------------
x_min, x_max = min(xs_all), max(xs_all)
y_min, y_max = min(ys_all), max(ys_all)

x_margin = (x_max - x_min) * X_MARGIN_RATIO
y_margin = (y_max - y_min) * Y_MARGIN_RATIO

plt.xlim(x_min - x_margin, x_max + x_margin)
plt.ylim(y_min - y_margin, y_max + y_margin)

# -----------------------------
# 坐标轴与标题
# -----------------------------
plt.xlabel("Inference Cost (log Token)", fontsize=20, fontweight="bold")
plt.ylabel("Reasoning Performance (EM)", fontsize=20, fontweight="bold")

plt.xticks(fontsize=18, fontweight="bold")
plt.yticks(fontsize=18, fontweight="bold")

plt.title("Accuracy–Efficiency Trade-off across Datasets", fontsize=24, fontweight="bold")

# -----------------------------
# Legend 1：方法（颜色）
# -----------------------------
method_handles = [
    plt.scatter(
        [], [],
        s=LEGEND_SIZE,
        color=method_colors[m],
        edgecolors="black",
        linewidth=0.8,
        label=m
    )
    for m in method_colors
]

legend1 = plt.legend(
    handles=method_handles,
    title="Method",
    fontsize=12,
    title_fontsize=13,
    loc="upper left",
    frameon=True
)

plt.gca().add_artist(legend1)

# -----------------------------
# Legend 2：数据集（形状）
# -----------------------------
dataset_handles = [
    plt.scatter(
        [], [],
        s=LEGEND_SIZE,
        color="gray",
        marker=dataset_markers[d],
        edgecolors="black",
        linewidth=0.8,
        label=d
    )
    for d in dataset_markers
]

plt.legend(
    handles=dataset_handles,
    title="Dataset",
    fontsize=12,
    title_fontsize=13,
    loc="lower right",
    frameon=True
)

# -----------------------------
# 网格与保存
# -----------------------------
plt.grid(True, linestyle="--", alpha=0.5, zorder=0)
plt.tight_layout()
plt.savefig("bubble.png", dpi=300)
plt.close()

print("Figure saved as all_datasets_tradeoff.png")
