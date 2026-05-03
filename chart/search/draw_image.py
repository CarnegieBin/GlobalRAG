
import pandas as pd
import matplotlib.pyplot as plt

# 要读取的文件及对应标签
files = {
    "./3b.csv": "Qwen2.5-3B",
    "./7b.csv": "Qwen2.5-7B",
}

plt.figure(figsize=(8, 6))

for filename, label in files.items():
    df = pd.read_csv(filename)

    # 每 3 步取一个点 .iloc[::3]
    steps = df["step"].iloc[:180]
    rewards = df["search_step"].iloc[:180]

    plt.plot(
        steps,
        rewards,
        marker="o",
        linewidth=2,
        markersize=5,
        label=label,
    )

plt.xlabel(
    "Step",
    fontsize=28,
    fontweight="bold"
)
plt.ylabel(
    "Search Num",
    fontsize=28,
    fontweight="bold"
)

plt.xticks(fontsize=16, fontweight="bold")
plt.yticks(fontsize=16, fontweight="bold")

plt.legend(
    fontsize=32,      # 图例文字大小
    loc="upper right" # 明确右上角（可选）
)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "search.png",
    format="png",
    bbox_inches="tight"
)

plt.close()  # 可选：防止在循环或脚本中重复占用内存



