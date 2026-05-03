import pandas as pd
import matplotlib.pyplot as plt


def plot_four_csv(path1, path2, path3, path4, output_path="plot.png"):
    # 读取 CSV
    df_list = [
        # pd.read_csv(path1),
        pd.read_csv(path2),
        # pd.read_csv(path3),
        pd.read_csv(path4)
    ]

    labels = ["qwen2.5-3b", "qwen2.5-7b"]

    # 创建图
    plt.figure(figsize=(10, 6))

    for df, label in zip(df_list, labels):
        # ------- 新增：限制 step 最大为 175 -------
        df = df[df["step"] <= 175]

        if not {"step", "rewards_scaled"}.issubset(df.columns):
            raise ValueError(f"CSV 缺少必要列: {label}")

        plt.plot(
            df["step"],
            df["rewards"],
            label=label,
            linewidth=1.5,
            marker = "s",  # 可选：加点
            markersize = 4  # 可选：点大小
        )

    # Annealing Weight 也要限制 step
    df0 = df_list[0]
    df0 = df0[df0["step"] <= 175]  # 新增
    plt.plot(
        df0["step"],
        (df0["m"] - 1) / 1.6,
        label="Annealing Weight",
        linewidth=1.5
    )

    plt.xlabel("Step", fontweight="bold")
    plt.ylabel("Train Reward", fontweight="bold")
    # plt.title("Rewards Scaled vs Step", fontweight="bold")

    plt.legend(prop={"weight": "bold"})
    plt.grid(True)

    # 保存文件
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"图像已保存到: {output_path}")

# 用法示例：
path1 = "./globalrag_qwen2.5-3b-base.csv"
path2 = "./globalrag_qwen2.5-3b.csv"
path3 = "./globalrag_qwen2.5-7b-base.csv"
path4 = "./globalrag_qwen2.5-7b.csv"
plot_four_csv(path1, path2, path3, path4, "result.png")
