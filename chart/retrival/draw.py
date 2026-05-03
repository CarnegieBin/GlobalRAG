import pandas as pd
import matplotlib.pyplot as plt

# 要读取的文件及对应标签
files = {
    "1.csv": "Top-k 1",
    "3.csv": "Top-k 3",
    "5.csv": "Top-k 5",
}

plt.figure(figsize=(8, 6))

for filename, label in files.items():
    df = pd.read_csv(filename)

    # 每 3 步取一个点 .iloc[::3]
    steps = df["step"]
    rewards = df["rewards"]

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
    fontsize=20,
    fontweight="bold"
)
plt.ylabel(
    "Train Rewards",
    fontsize=20,
    fontweight="bold"
)

plt.xticks(fontsize=16, fontweight="bold")
plt.yticks(fontsize=16, fontweight="bold")

plt.legend(
    fontsize=24,      # 图例文字大小
    loc="upper right" # 明确右上角（可选）
)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "appendix_retrieval.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.close()  # 可选：防止在循环或脚本中重复占用内存