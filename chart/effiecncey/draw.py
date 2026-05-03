import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 原始数据（Qwen2.5-3B-Instruct）
# -----------------------------
data = {
    "HotpotQA": {
        "Search-R1":  {"EM": 31.3, "F1": 41.5, "Token": 107.5},
        "StepSearch": {"EM": 28.9, "F1": 39.9, "Token": 245.8},
        "GlobalRAG":  {"EM": 32.9, "F1": 44.2, "Token": 391.4},
    },
    "2WikiMultihopQA": {
        "Search-R1":  {"EM": 31.3, "F1": 36.5, "Token": 129.2},
        "StepSearch": {"EM": 31.9, "F1": 38.3, "Token": 246.3},
        "GlobalRAG":  {"EM": 42.3, "F1": 47.8, "Token": 385.7},
    },
    "Musique": {
        "Search-R1":  {"EM": 7.7,  "F1": 13.2, "Token": 126.1},
        "StepSearch": {"EM": 9.5,  "F1": 16.6, "Token": 285.9},
        "GlobalRAG":  {"EM": 10.8, "F1": 18.6, "Token": 430.9},
    },
    "Bamboogle": {
        "Search-R1":  {"EM": 28.0, "F1": 34.7, "Token": 106.3},
        "StepSearch": {"EM": 32.0, "F1": 43.8, "Token": 237.0},
        "GlobalRAG":  {"EM": 37.6, "F1": 49.3, "Token": 343.6},
    },
    "WikiHop": {
        "Search-R1":  {"EM": 8.2,  "F1": 13.8, "Token": 125.3},
        "StepSearch": {"EM": 11.2, "F1": 18.1, "Token": 188.9},
        "GlobalRAG":  {"EM": 12.9, "F1": 20.7, "Token": 300.0},
    },
}

# -----------------------------
# 颜色映射
# -----------------------------
method_colors = {
    "Search-R1": "#1f77b4",
    "StepSearch": "#ff7f0e",
    "GlobalRAG": "#2ca02c",
}

# Search-R1 的基准气泡面积
BASE_SIZE = 4000

# Legend 缩放系数
LEGEND_SCALE = 40
LEGEND_SIZE = BASE_SIZE / LEGEND_SCALE

# 放大倍数
AMPLIFY = 3

# -----------------------------
# 逐数据集绘图
# -----------------------------
for dataset, methods in data.items():
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Search-R1 作为效率基准
    sr1 = methods["Search-R1"]
    sr1_score = (sr1["EM"] + sr1["F1"]) / 2.0
    sr1_eff = sr1_score / np.log(sr1["Token"])

    xs, ys, bubble_sizes = [], [], []

    max_bubble_size = 0

    for method, values in methods.items():
        token = values["Token"]
        score = (values["EM"] + values["F1"]) / 2.0

        x = np.log10(token)
        y = score

        eff = score / np.log(token)
        ratio = eff / sr1_eff
        bubble_size = BASE_SIZE * (1 + AMPLIFY * (ratio - 1))

        # 记录最大气泡面积
        max_bubble_size = max(max_bubble_size, bubble_size)

        xs.append(x)
        ys.append(y)
        bubble_sizes.append(bubble_size)

        plt.scatter(
            x,
            y,
            s=bubble_size,
            color=method_colors[method],
            alpha=0.75,
            edgecolors="black",
            linewidth=0.8,
            zorder=3,
        )

    # -----------------------------
    # 轴范围扩展，保证气泡在边框内
    # -----------------------------
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # 气泡半径转换（points -> 数据坐标）
    bubble_radius_pt = np.sqrt(max_bubble_size)

    fig_width, fig_height = plt.gcf().get_size_inches()
    x_range = x_max - x_min
    y_range = y_max - y_min

    x_padding = (bubble_radius_pt / 72) / fig_width * x_range
    y_padding = (bubble_radius_pt / 72) / fig_height * y_range

    plt.xlim(x_min - x_padding, x_max + x_padding)
    plt.ylim(y_min - y_padding, y_max + y_padding)

    plt.xlabel("Inference Cost", fontsize=20, fontweight="bold")
    plt.ylabel("Reasoning Performance", fontsize=20, fontweight="bold")

    plt.xticks(fontsize=18, fontweight="bold")
    plt.yticks(fontsize=18, fontweight="bold")

    plt.title(dataset, fontsize=24, fontweight="bold")

    # -----------------------------
    # 手动构造 Legend（气泡大小一致）
    # -----------------------------
    legend_handles = [
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

    plt.legend(
        handles=legend_handles,
        fontsize=10,
        loc="upper left",
        frameon=True
    )

    plt.grid(True, linestyle="--", alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{dataset}.png", dpi=300)
    plt.close()

print("All figures have been saved successfully.")



