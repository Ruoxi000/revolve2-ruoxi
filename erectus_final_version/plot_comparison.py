"""
tools/plot_comparison.py
功能：读取两个实验的 CSV 历史数据，绘制 Height 和 Dxy 的对比图。
用于生成论文 Fig 3 (Evolutionary Progress) 和 Fig 4 (Curriculum Comparison)。
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config

# --- 配置区域 ---
# 请替换为您实际生成的 CSV 文件名
FILE_CURRICULUM = "history_database_500_v4_h1_pure.csv"  # 主实验 (有课程)
FILE_BASELINE = "history_database_500_v4_h1_no_transition_new.csv"  # 对照实验 (无课程)

# 定义课程阶段边界 (根据 config.py)
PHASE_1_END = 50  # e.g., 50
PHASE_2_END = PHASE_1_END + 10  # e.g., 60

# 绘图风格设置
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'lines.linewidth': 2.5
})


# ----------------

def load_and_label(filename, label):
    """读取 CSV 并添加实验标签列"""
    try:
        df = pd.read_csv(filename)
        df['Experiment'] = label
        return df
    except FileNotFoundError:
        print(f"Error: File {filename} not found. Please run sample_generations.py first.")
        return None


def plot_metric_comparison(df_all, metric_col, ylabel, title, filename, y_limit=None):
    """绘制单个指标的对比图，并标注课程阶段区域"""
    plt.figure(figsize=(8, 5))

    # 使用 seaborn 绘制带置信区间（如果有多次运行数据）的折线图
    # 如果只有单次运行数据，它只会画一条线
    sns.lineplot(
        data=df_all,
        x="generation",
        y=metric_col,
        hue="Experiment",
        palette=["#1f77b4", "#d62728"],  # 蓝/红配色
        marker="o", markersize=6, markevery=5  # 每5个点标一个记号，防止太密
    )

    # 添加阶段背景色
    plt.axvspan(0, PHASE_1_END, color='gray', alpha=0.1, label='Stand Phase')
    plt.axvspan(PHASE_1_END, PHASE_2_END, color='yellow', alpha=0.1, label='Transition Phase')
    # 添加阶段分割线
    plt.axvline(x=PHASE_1_END, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=PHASE_2_END, color='k', linestyle='--', alpha=0.3)

    # 设置坐标轴
    plt.xlim(0, config.NUM_GENERATIONS)
    if y_limit:
        plt.ylim(y_limit)

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel(ylabel)

    # 优化图例
    handles, labels = plt.gca().get_legend_handles_labels()
    # 重新排序图例：先把实验放前面，阶段说明放后面
    order = [0, 1, 2, 3] if len(labels) > 2 else [0, 1]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='best')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")


def main():
    # 1. 加载数据
    df_curr = load_and_label(FILE_CURRICULUM, "With Curriculum (Ours)")
    df_base = load_and_label(FILE_BASELINE, "No Curriculum (Baseline)")

    if df_curr is None or df_base is None:
        return

    # 合并数据
    df_all = pd.concat([df_curr, df_base], ignore_index=True)

    # 2. 绘制 Height 对比图 (Fig 4 in plan)
    plot_metric_comparison(
        df_all,
        metric_col="h_mean",
        ylabel="Max Normalized Height (h_norm)",
        title="Evolution of Body Height",
        filename="comparison_height.png",
        y_limit=(0, 1.0)  # Height 是归一化的，0-1
    )

    # 3. 绘制 Dxy 对比图 (Fig 3b in plan)
    #    对于 Dxy，我们可能更关心主实验本身的进展，但也对比一下baseline
    plot_metric_comparison(
        df_all,
        metric_col="dxy",
        ylabel="Max Displacement (m)",
        title="Evolution of Locomotion Distance",
        filename="comparison_dxy.png"
        # y_limit 不设，自动调整
    )

    print("\nDone! Generated comparison_height.png and comparison_dxy.png.")


if __name__ == "__main__":
    main()