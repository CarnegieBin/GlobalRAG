import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# 数据准备
datasets = ['HotpotQA', '2WikiMultihopQA', 'Musique', 'Bamboogle', 'WikiHop']
search_r1_em = [30.1, 36.5, 8.3, 32.0, 10.6]
global_rag_em = [32.9, 42.3, 10.8, 37.6, 12.9]

# 设置绘图风格
plt.rcParams['font.family'] = 'sans-serif'
fig, axes = plt.subplots(1, 5, figsize=(14, 6), sharey=True)
plt.subplots_adjust(wspace=0.1)

# 颜色设置
color_baseline = '#f0f0f0'  # 浅灰
color_ours = '#3b5998'  # 深蓝
edge_color = 'black'

# --- 关键参数调整 ---
bar_width = 0.3  # 柱子宽度（越小越瘦）
# x_pos 控制两个柱子的中心位置。
# 如果想让它们靠得更近，就把这两个值的绝对值调小（例如 -0.12 和 0.12）
x_pos = [-0.20, 0.20]

# 提升百分比标注样式
improve_color = '#d4af37'   # 金黄色（论文友好）
improve_fontsize = 20     # 字号（可再调大）
improve_offset = 1.0       # 相对虚线的垂直偏移


for i, ax in enumerate(axes):
    val_base = search_r1_em[i]
    val_ours = global_rag_em[i]
    improvement = ((val_ours - val_base) / val_base) * 100

    # 绘制两个柱子，分别位于中心线左侧和右侧
    ax.bar(x_pos[0], val_base, color=color_baseline, edgecolor=edge_color, width=bar_width)
    ax.bar(x_pos[1], val_ours, color=color_ours, edgecolor=edge_color, width=bar_width)

    # 数值标注 (放在柱子上方)
    # ax.text(x_pos[0], val_base + 1, f'{val_base:.1f}', ha='center', va='bottom', fontsize=9)
    # ax.text(x_pos[1], val_ours + 1, f'{val_ours:.1f}', ha='center', va='bottom', fontsize=9)

    # 绘制辅助虚线（从左柱顶端延伸到右柱上方）
    ax.hlines(y=val_ours, xmin=x_pos[0]-0.175, xmax=0.20-0.175, color='black', linestyle='--', linewidth=1.2)
    # 绘制提升箭头
    ax.annotate('', xy=(x_pos[0], val_ours), xytext=(x_pos[0], val_base),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    # 标注提升百分比（严格放在虚线上方）
    ax.text(
        np.mean(x_pos),  # 水平居中于两根柱子
        val_ours + improve_offset,  # 虚线上方
        f'+{improvement:.1f}%',
        ha='center',
        va='bottom',
        color=improve_color,
        fontweight='bold',
        fontsize=improve_fontsize
    )


    # 格式设置
    ax.set_title(datasets[i], fontweight='bold', fontsize=16, pad=10)
    ax.set_xticks([])  # 隐藏 X 轴刻度线
    ax.set_xlim(-0.5, 0.5)  # 限制 X 轴范围，让柱子在子图内居中且不显得太空旷
    ax.set_ylim(0, 50)  # 纵轴上限 50

# 设置纵轴标签
axes[0].set_ylabel('EM', fontsize=18)

# 添加底部的 Legend
patch1 = mpatches.Rectangle(
    (0, 0), 3, 3,
    facecolor=color_baseline,
    edgecolor=edge_color,
    label='Search-R1'
)
patch2 = mpatches.Rectangle(
    (0, 0), 3, 3,
    facecolor=color_ours,
    edgecolor=edge_color,
    label='GlobalRAG'
)


fig.legend(handles=[patch1, patch2], loc='lower center', ncol=2,
           frameon=False, fontsize=30, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.1, 1, 0.95])

plt.savefig(
    "algorithm.pdf",
    format="pdf",
    bbox_inches="tight"
)
plt.close()