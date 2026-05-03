import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df1 = pd.read_csv("./0_5.csv")
df2 = pd.read_csv("./1.csv")
df3 = pd.read_csv("./2_5.csv")

# 创建画布
plt.figure(figsize=(8, 6))

# 绘制折线 + 圆点

plt.plot(
    df3["step"], df3["bamboogle_step"],
    marker="o",
    label=r"$\lambda = \gamma = 0.25$"
)
plt.plot(
    df1["step"], df1["bamboogle_step"],
    marker="o",
    label=r"$\lambda = \gamma = 0.5$"
)
plt.plot(
    df2["step"], df2["bamboogle_step"],
    marker="o",
    label=r"$\lambda = \gamma = 1.0$"
)


# 坐标轴与标题
plt.xlabel("Step", fontweight='bold', fontsize=16)
plt.ylabel("Bamboogle EM", fontweight='bold', fontsize=16)

plt.xticks(fontsize=16, fontweight="bold")
plt.yticks(fontsize=16, fontweight="bold")

# 图例放在左下角
plt.legend(loc="lower left", fontsize=16)

# 网格（如论文不需要，可删除）
plt.grid(True)

# 布局与显示
plt.tight_layout()
plt.savefig("./reward_sensitivity.png")


