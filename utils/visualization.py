"""
可视化工具模块
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional
import seaborn as sns
from utils.statistics import StatisticsAnalyzer


class TradingVisualizer:
    """交易分析可视化工具"""
    
    def __init__(self):
        # 设置中文字体和样式
        plt.style.use('seaborn-v0_8')
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        self.stats_analyzer = StatisticsAnalyzer()
    
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

    def plot_strategy_comprehensive(self,
                                   price: pd.Series,
                                   nav: pd.Series,
                                   benchmark: Optional[pd.Series] = None,
                                   buy_points: Optional[pd.DataFrame] = None,
                                   sell_points: Optional[pd.DataFrame] = None,
                                   title: str = "策略综合分析",
                                   save_path: Optional[str] = None) -> None:
        """
        绘制策略收盘价走势、买卖点位、策略与市场累计收益对比（参考 MA_strategy_v2 风格）

        参数说明：
        - price: 收盘价时间序列（索引为时间）
        - nav: 策略净值（初始=1.0）
        - benchmark: 基准净值（初始=1.0），可选
        - buy_points: 可选，包含列 ['time', 'price'] 的 DataFrame
        - sell_points: 可选，包含列 ['time', 'price'] 的 DataFrame
        - title: 图标题
        - save_path: 图片保存路径
        """
        if price is None or len(price) == 0 or nav is None or len(nav) == 0:
            print("缺少必要的数据用于绘图（price/nav）")
            return
        
        # 统一索引
        if not isinstance(price.index, pd.DatetimeIndex):
            price.index = pd.to_datetime(price.index)
        if not isinstance(nav.index, pd.DatetimeIndex):
            nav.index = pd.to_datetime(nav.index)
        if benchmark is not None and not isinstance(benchmark.index, pd.DatetimeIndex):
            benchmark.index = pd.to_datetime(benchmark.index)

        common_idx = price.index.intersection(nav.index)
        if benchmark is not None:
            common_idx = common_idx.intersection(benchmark.index)
        price = price.loc[common_idx]
        nav = nav.loc[common_idx]
        if benchmark is not None:
            benchmark = benchmark.loc[common_idx]

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 1, hspace=0.28, height_ratios=[2, 1.6, 1.4])

        # 1) 价格与买卖点
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(price.index, price.values, color="#2E86C1", lw=1.6, label="收盘价")
        
        # 买卖点标注
        def _scatter_points(points: Optional[pd.DataFrame], marker: str, color: str, label: str):
            if points is not None and len(points) > 0:
                pts = points.copy()
                if 'time' in pts.columns:
                    pts_time = pd.to_datetime(pts['time'])
                else:
                    pts_time = pts.index
                pts_price = pts['price'] if 'price' in pts.columns else price.reindex(pts_time).values
                ax1.scatter(pts_time, pts_price, marker=marker, s=90, zorder=5,
                            color=color, edgecolors='black', linewidths=0.8, label=label)

        _scatter_points(buy_points, marker='^', color='red', label='买入')
        _scatter_points(sell_points, marker='v', color='green', label='卖出')

        ax1.set_title('价格走势与买卖点', fontsize=14, fontweight='bold')
        ax1.set_ylabel('价格', fontsize=11)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(alpha=0.3, linestyle='--')

        # 2) 策略净值 vs 基准净值
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(nav.index, nav.values, color="#E74C3C", lw=1.8, label='策略净值')
        if benchmark is not None:
            ax2.plot(benchmark.index, benchmark.values, color="#95A5A6", lw=1.6, ls='--', label='基准净值')
        ax2.axhline(1.0, color='black', ls=':', lw=1, alpha=0.5)
        ax2.set_title('策略净值 vs 基准净值', fontsize=14, fontweight='bold')
        ax2.set_ylabel('净值', fontsize=11)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(alpha=0.3, linestyle='--')

        # 3) 累计收益对比（含超额收益）
        ax3 = fig.add_subplot(gs[2, 0])
        cum_ret_strategy = (nav - 1.0) * 100.0
        ax3.plot(cum_ret_strategy.index, cum_ret_strategy.values, color="#E74C3C", lw=1.8, label='策略累计收益')
        if benchmark is not None:
            cum_ret_bench = (benchmark - 1.0) * 100.0
            ax3.plot(cum_ret_bench.index, cum_ret_bench.values, color="#95A5A6", lw=1.6, ls='--', label='基准累计收益')
            excess = cum_ret_strategy - cum_ret_bench
            ax3_2 = ax3.twinx()
            ax3_2.fill_between(excess.index, 0, excess.values, color="#3498DB", alpha=0.22, label='超额收益')
            ax3_2.set_ylabel('超额收益(%)', color="#3498DB")
            ax3_2.tick_params(axis='y', labelcolor="#3498DB")
            ax3_2.legend(loc='upper right', fontsize=9)
        ax3.axhline(0, color='black', ls=':', lw=1, alpha=0.5)
        ax3.set_title('累计收益对比', fontsize=13, fontweight='bold')
        ax3.set_xlabel('时间', fontsize=11)
        ax3.set_ylabel('累计收益(%)', fontsize=11)
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(alpha=0.3, linestyle='--')

        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"策略综合分析图已保存至: {save_path}")
        else:
            plt.show()
