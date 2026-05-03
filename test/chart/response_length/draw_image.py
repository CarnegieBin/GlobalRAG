import pandas as pd
import matplotlib.pyplot as plt


def plot_four_csv(path1, path2, output_path="plot.png"):
    # 读取 CSV
    df_list = [
        pd.read_csv(path1),
        pd.read_csv(path2),
    ]

    labels = ["qwen2.5-3b", "qwen2.5-7b"]

    # 创建图
    plt.figure(figsize=(10, 6))

    for df, label in zip(df_list, labels):
        # 限制最大 step
        df = df[df["step"] <= 175]

        # ====== 新增：每隔 5 个 step 采样一次 ======
        # df = df.iloc[::4, :]

        plt.plot(
            df["step"],
            df["response_length"],
            label=label,
            linewidth=1.5,
            marker="s",         # 可选：加点
            markersize=4        # 可选：点大小
        )

    plt.xlabel("Step", fontweight="bold")
    plt.ylabel("Response Length", fontweight="bold")
    plt.legend(prop={"weight": "bold"})
    plt.grid(True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"图像已保存到: {output_path}")

# 用法示例：
path1 = "./3b.csv"
path2 = "./7b.csv"
plot_four_csv(path1, path2, "result.png")

