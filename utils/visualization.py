"""
可视化工具模块
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional
import seaborn as sns


class TradingVisualizer:
    """交易分析可视化工具"""
    
    def __init__(self):
        # 设置中文字体和样式
        plt.style.use('seaborn-v0_8')
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_predictions_vs_truth(self, preds: pd.Series, y_true: pd.Series,
                                 title: str = "预测值 vs 真实值",
                                 save_path: Optional[str] = None) -> None:
        """绘制预测值与真实值对比图"""
        if preds is None or len(preds) == 0:
            return
        
        # 对齐索引
        idx = preds.dropna().index.intersection(y_true.dropna().index)
        if len(idx) == 0:
            return
        
        y_plot = y_true.loc[idx].astype(float)
        p_plot = preds.loc[idx].astype(float)

        # 基于预测生成持仓与换手点
        pos = np.sign(p_plot).fillna(0.0)
        change = pos.diff().fillna(pos.iloc[0])
        turnover = change.abs()
        trade_times = turnover[turnover > 0].index
        long_entries = change[change > 0].index
        short_entries = change[change < 0].index

        fig, ax = plt.subplots(figsize=(15, 8))
        
        # 绘制时间序列
        ax.plot(y_plot.index, y_plot.values, label='真实值', 
               color='#1f77b4', alpha=0.8, linewidth=1.5)
        ax.plot(p_plot.index, p_plot.values, label='预测值', 
               color='#ff7f0e', alpha=0.8, linewidth=1.5)

        # 标注交易变更时间点
        for t in trade_times:
            ax.axvline(t, color='gray', alpha=0.15, linewidth=1)
        
        # 标注买入卖出点
        if len(long_entries) > 0:
            ax.scatter(long_entries, np.zeros(len(long_entries)), 
                      marker='^', color='green', label='做多', zorder=3, s=50)
        if len(short_entries) > 0:
            ax.scatter(short_entries, np.zeros(len(short_entries)), 
                      marker='v', color='red', label='做空', zorder=3, s=50)

        ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.3)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('收益率', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        corr = p_plot.corr(y_plot)
        ic_text = f'IC: {corr:.4f}' if pd.notna(corr) else 'IC: N/A'
        ax.text(0.02, 0.98, ic_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"图片已保存至: {save_path}")
        else:
            plt.show()
    
    def plot_feature_importance(self, importance: Dict[str, float],
                               title: str = "特征重要性",
                               top_n: int = 20,
                               save_path: Optional[str] = None) -> None:
        """绘制特征重要性图"""
        if not importance:
            print("没有特征重要性数据")
            return
        
        # 排序并取前N个
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        if top_n:
            sorted_features = sorted_features[:top_n]
        
        features, values = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(features) * 0.4)))
        
        # 创建水平条形图
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, values, color='skyblue', alpha=0.8)
        
        # 设置标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('重要性', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.4f}', ha='left', va='center', fontsize=9)
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"特征重要性图已保存至: {save_path}")
        else:
            plt.show()
    
    def plot_cross_validation_results(self, cv_results: Dict,
                                     save_path: Optional[str] = None) -> None:
        """绘制交叉验证结果"""
        if 'by_fold' not in cv_results:
            print("没有交叉验证结果数据")
            return
        
        df_folds = pd.DataFrame(cv_results['by_fold'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('交叉验证结果', fontsize=16, fontweight='bold')
        
        # Pearson IC
        axes[0, 0].bar(df_folds['fold'], df_folds['pearson_ic'], 
                      color='lightblue', alpha=0.7)
        axes[0, 0].axhline(df_folds['pearson_ic'].mean(), color='red', 
                          linestyle='--', label=f'均值: {df_folds["pearson_ic"].mean():.4f}')
        axes[0, 0].set_title('Pearson IC')
        axes[0, 0].set_xlabel('折数')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Spearman IC
        axes[0, 1].bar(df_folds['fold'], df_folds['spearman_ic'], 
                      color='lightgreen', alpha=0.7)
        axes[0, 1].axhline(df_folds['spearman_ic'].mean(), color='red', 
                          linestyle='--', label=f'均值: {df_folds["spearman_ic"].mean():.4f}')
        axes[0, 1].set_title('Spearman IC')
        axes[0, 1].set_xlabel('折数')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # RMSE
        axes[1, 0].bar(df_folds['fold'], df_folds['rmse'], 
                      color='salmon', alpha=0.7)
        axes[1, 0].axhline(df_folds['rmse'].mean(), color='red', 
                          linestyle='--', label=f'均值: {df_folds["rmse"].mean():.4f}')
        axes[1, 0].set_title('RMSE')
        axes[1, 0].set_xlabel('折数')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # 方向准确率
        axes[1, 1].bar(df_folds['fold'], df_folds['dir_acc'], 
                      color='gold', alpha=0.7)
        axes[1, 1].axhline(df_folds['dir_acc'].mean(), color='red', 
                          linestyle='--', label=f'均值: {df_folds["dir_acc"].mean():.4f}')
        axes[1, 1].set_title('方向准确率')
        axes[1, 1].set_xlabel('折数')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"交叉验证结果图已保存至: {save_path}")
        else:
            plt.show()
    
    def plot_return_distribution(self, returns: pd.Series,
                               title: str = "收益率分布",
                               save_path: Optional[str] = None) -> None:
        """绘制收益率分布图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 直方图
        ax1.hist(returns.dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(returns.mean(), color='red', linestyle='--', 
                   label=f'均值: {returns.mean():.6f}')
        ax1.axvline(returns.median(), color='green', linestyle='--', 
                   label=f'中位数: {returns.median():.6f}')
        ax1.set_xlabel('收益率')
        ax1.set_ylabel('频数')
        ax1.set_title('收益率直方图')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Q-Q图
        from scipy import stats
        stats.probplot(returns.dropna(), dist="norm", plot=ax2)
        ax2.set_title('Q-Q图 (正态分布)')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"收益率分布图已保存至: {save_path}")
        else:
            plt.show()
