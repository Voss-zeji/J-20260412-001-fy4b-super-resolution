#!/usr/bin/env python3
"""
lv2_micro 实验结果分析工具
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def load_results():
    """加载实验结果"""
    if not Path('results.tsv').exists():
        print("Error: results.tsv not found")
        return None

    df = pd.read_csv('results.tsv', sep='\t', comment='#')
    return df


def print_ranking(df):
    """打印排名"""
    # 按 val_psnr 排序
    df_sorted = df.sort_values('val_psnr', ascending=False).reset_index(drop=True)
    df_sorted.index = df_sorted.index + 1  # 排名从1开始

    print("\n" + "="*80)
    print("实验结果排名 (按 val_psnr 降序)")
    print("="*80)

    # 格式化输出
    print(f"{'Rank':<6}{'Experiment':<25}{'PSNR':<10}{'Memory':<10}{'Status':<10}{'Description'}")
    print("-"*80)

    for idx, row in df_sorted.iterrows():
        status_color = "✓" if row['status'] == 'keep' else "✗"
        print(f"{idx:<6}{row['experiment']:<25}{row['val_psnr']:<10.2f}{row['memory_gb']:<10.1f}{status_color:<10}{row['description']}")

    print("-"*80)
    print(f"\n最佳实验: {df_sorted.iloc[0]['experiment']}")
    print(f"最佳 PSNR: {df_sorted.iloc[0]['val_psnr']:.2f} dB")

    # 统计信息
    keep_count = len(df[df['status'] == 'keep'])
    discard_count = len(df[df['status'] == 'discard'])
    print(f"\n统计: {keep_count} keep, {discard_count} discard, {len(df)} total")

    return df_sorted


def plot_progress(df):
    """绘制优化进度"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1：实验时间线
    df_sorted = df.sort_values('val_psnr', ascending=False)
    best_psnr = df_sorted.iloc[0]['val_psnr']

    axes[0].plot(range(len(df)), df['val_psnr'].values, 'o-', color='steelblue', label='All experiments')
    axes[0].axhline(y=best_psnr, color='r', linestyle='--', label=f'Best: {best_psnr:.2f} dB')
    axes[0].fill_between(range(len(df)), df['val_psnr'].values, alpha=0.3, color='steelblue')
    axes[0].set_xlabel('Experiment Order')
    axes[0].set_ylabel('val_psnr (dB)')
    axes[0].set_title('Optimization Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 图2：keep vs discard 分布
    status_counts = df['status'].value_counts()
    colors = ['#2ecc71' if s == 'keep' else '#e74c3c' for s in status_counts.index]
    axes[1].bar(status_counts.index, status_counts.values, color=colors)
    axes[1].set_ylabel('Count')
    axes[1].set_title('Experiment Status Distribution')

    for i, v in enumerate(status_counts.values):
        axes[1].text(i, v + 0.5, str(v), ha='center')

    plt.tight_layout()
    plt.savefig('experiments/optimization_progress.png', dpi=150)
    print(f"\n优化进度图已保存: experiments/optimization_progress.png")


def plot_parameter_analysis(df):
    """分析参数影响（如果有明确的参数信息）"""
    # 从 description 中提取可能的参数
    descriptions = df['description'].astype(str)

    # 检查是否有学习率实验
    lr_exps = df[descriptions.str.contains('lr=', na=False)]
    if len(lr_exps) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))

        # 提取学习率值
        lr_values = []
        for desc in lr_exps['description']:
            try:
                lr = float(desc.split('lr=')[1].split(',')[0])
                lr_values.append(lr)
            except:
                lr_values.append(None)

        lr_exps = lr_exps.copy()
        lr_exps['lr'] = lr_values
        lr_exps = lr_exps.dropna(subset=['lr'])

        if len(lr_exps) > 0:
            ax.scatter(lr_exps['lr'], lr_exps['val_psnr'], s=100, alpha=0.6)
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel('val_psnr (dB)')
            ax.set_xscale('log')
            ax.set_title('Learning Rate vs Performance')
            ax.grid(True, alpha=0.3)

            # 标注最佳点
            best_idx = lr_exps['val_psnr'].idxmax()
            best_row = lr_exps.loc[best_idx]
            ax.annotate(f"{best_row['val_psnr']:.2f}",
                       (best_row['lr'], best_row['val_psnr']),
                       xytext=(10, 10), textcoords='offset points')

            plt.tight_layout()
            plt.savefig('experiments/lr_analysis.png', dpi=150)
            print(f"学习率分析图已保存: experiments/lr_analysis.png")


def suggest_next_experiment(df):
    """基于历史结果建议下一个实验方向"""
    print("\n" + "="*80)
    print("下一步实验建议")
    print("="*80)

    keep_exps = df[df['status'] == 'keep']

    if len(keep_exps) == 0:
        print("建议：先运行基线实验")
        return

    # 找出最近的最佳实验
    best_exp = keep_exps.loc[keep_exps['val_psnr'].idxmax()]
    print(f"当前最佳: {best_exp['experiment']} (PSNR: {best_exp['val_psnr']:.2f} dB)")
    print(f"描述: {best_exp['description']}")

    print("\n建议的实验方向:")
    print("1. 学习率微调: 在当前最佳 lr 附近 ±20% 尝试")
    print("2. 模型深度: 尝试增加/减少 1-2 个 block")
    print("3. 损失权重: 调整 L1 和 SSIM 的权重比例")
    print("4. 批次大小: 尝试更大 batch size + 更多 epochs")
    print("5. 消融实验: 移除某个组件验证其必要性")

    # 检查是否需要早停
    recent = df.tail(10) if len(df) >= 10 else df
    recent_improvements = len(recent[recent['status'] == 'keep'])

    if len(recent) >= 5 and recent_improvements == 0:
        print("\n⚠️  警告: 最近 5 次实验无改善，建议:")
        print("   - 换一个参数方向")
        print("   - 尝试组合之前有效的改动")
        print("   - 考虑进入 lv3_fusion 层")


def main():
    parser = argparse.ArgumentParser(description='分析 lv2_micro 实验结果')
    parser.add_argument('--plot', action='store_true', help='生成可视化图表')
    parser.add_argument('--suggest', action='store_true', help='显示实验建议')
    args = parser.parse_args()

    # 加载数据
    df = load_results()
    if df is None:
        return

    # 打印排名
    df_sorted = print_ranking(df)

    # 生成图表
    if args.plot:
        plot_progress(df)
        plot_parameter_analysis(df)

    # 显示建议
    if args.suggest or len(df) < 3:
        suggest_next_experiment(df)

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
