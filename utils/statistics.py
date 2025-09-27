"""
统计分析工具模块
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Union
from scipy import stats


class StatisticsAnalyzer:
    """统计分析器"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def calculate_moments(self, data: pd.Series, name: str = "数据") -> Dict[str, float]:
        """计算数据的各阶矩统计"""
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            return {"error": "数据为空"}
        
        stats_dict = {
            "样本数": len(clean_data),
            "均值": float(clean_data.mean()),
            "标准差": float(clean_data.std()),
            "方差": float(clean_data.var()),
            "偏度": float(clean_data.skew()),  # 三阶矩
            "峰度": float(clean_data.kurtosis()),  # 四阶矩
            "最小值": float(clean_data.min()),
            "25分位数": float(clean_data.quantile(0.25)),
            "中位数": float(clean_data.median()),
            "75分位数": float(clean_data.quantile(0.75)),
            "最大值": float(clean_data.max()),
            "极差": float(clean_data.max() - clean_data.min())
        }
        
        # 使用scipy计算更精确的偏度和峰度
        stats_dict["偏度_scipy"] = float(stats.skew(clean_data))
        stats_dict["峰度_scipy"] = float(stats.kurtosis(clean_data))  # 默认是excess kurtosis
        stats_dict["峰度_fisher"] = float(stats.kurtosis(clean_data, fisher=True))  # excess kurtosis
        stats_dict["峰度_pearson"] = float(stats.kurtosis(clean_data, fisher=False))  # raw kurtosis
        
        return stats_dict
    
    def print_moments_summary(self, data: pd.Series, name: str = "数据"):
        """打印数据的统计摘要"""
        stats_dict = self.calculate_moments(data, name)
        
        if "error" in stats_dict:
            print(f"错误: {stats_dict['error']}")
            return
        
        print(f"\n{'='*50}")
        print(f"{name} 统计摘要")
        print(f"{'='*50}")
        
        print(f"基础统计:")
        print(f"  样本数: {stats_dict['样本数']:,}")
        print(f"  均值: {stats_dict['均值']:.6f}")
        print(f"  标准差: {stats_dict['标准差']:.6f}")
        print(f"  方差: {stats_dict['方差']:.6f}")
        
        print(f"\n分布形状:")
        print(f"  偏度 (pandas): {stats_dict['偏度']:.4f}")
        print(f"  偏度 (scipy): {stats_dict['偏度_scipy']:.4f}")
        print(f"  峰度 (pandas): {stats_dict['峰度']:.4f}")
        print(f"  峰度 (scipy excess): {stats_dict['峰度_scipy']:.4f}")
        print(f"  峰度 (pearson): {stats_dict['峰度_pearson']:.4f}")
        
        print(f"\n分位数:")
        print(f"  最小值: {stats_dict['最小值']:.6f}")
        print(f"  25%: {stats_dict['25分位数']:.6f}")
        print(f"  50% (中位数): {stats_dict['中位数']:.6f}")
        print(f"  75%: {stats_dict['75分位数']:.6f}")
        print(f"  最大值: {stats_dict['最大值']:.6f}")
        print(f"  极差: {stats_dict['极差']:.6f}")
        
        # 解释偏度和峰度
        self._interpret_skewness_kurtosis(stats_dict['偏度'], stats_dict['峰度'])
    
    def _interpret_skewness_kurtosis(self, skewness: float, kurtosis: float):
        """解释偏度和峰度的含义"""
        print(f"\n分布特征解释:")
        
        # 偏度解释
        if abs(skewness) < 0.5:
            skew_desc = "近似对称"
        elif skewness > 0.5:
            skew_desc = "右偏 (正偏，尾部向右延伸)"
        else:
            skew_desc = "左偏 (负偏，尾部向左延伸)"
        
        print(f"  偏度 ({skewness:.4f}): {skew_desc}")
        
        # 峰度解释 (pandas默认使用Fisher定义，即excess kurtosis)
        if abs(kurtosis) < 0.5:
            kurt_desc = "接近正态分布的峰度"
        elif kurtosis > 0.5:
            kurt_desc = "尖峰分布 (比正态分布更尖)"
        else:
            kurt_desc = "平峰分布 (比正态分布更平)"
        
        print(f"  峰度 ({kurtosis:.4f}): {kurt_desc}")
    
    def plot_distribution_analysis(self, data: pd.Series, name: str = "数据", 
                                 save_path: Optional[str] = None):
        """绘制分布分析图"""
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            print("数据为空，无法绘图")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{name} 分布分析', fontsize=16, fontweight='bold')
        
        # 1. 直方图
        axes[0, 0].hist(clean_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(clean_data.mean(), color='red', linestyle='--', 
                          label=f'均值: {clean_data.mean():.4f}')
        axes[0, 0].axvline(clean_data.median(), color='green', linestyle='--', 
                          label=f'中位数: {clean_data.median():.4f}')
        axes[0, 0].set_title('直方图')
        axes[0, 0].set_xlabel('数值')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Q-Q图
        stats.probplot(clean_data, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q图 (与正态分布比较)')
        axes[0, 1].grid(alpha=0.3)
        
        # 3. 箱线图
        box_plot = axes[1, 0].boxplot(clean_data, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        axes[1, 0].set_title('箱线图')
        axes[1, 0].set_ylabel('数值')
        axes[1, 0].grid(alpha=0.3)
        
        # 4. 密度图 + 正态分布对比
        axes[1, 1].hist(clean_data, bins=50, density=True, alpha=0.7, 
                       color='skyblue', label='实际分布')
        
        # 叠加正态分布
        mu, sigma = clean_data.mean(), clean_data.std()
        x = np.linspace(clean_data.min(), clean_data.max(), 100)
        normal_curve = stats.norm.pdf(x, mu, sigma)
        axes[1, 1].plot(x, normal_curve, 'r-', linewidth=2, label='正态分布')
        
        axes[1, 1].set_title('密度分布对比')
        axes[1, 1].set_xlabel('数值')
        axes[1, 1].set_ylabel('密度')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        # 添加统计信息文本
        stats_text = f"""统计信息:
样本数: {len(clean_data):,}
均值: {clean_data.mean():.4f}
标准差: {clean_data.std():.4f}
偏度: {clean_data.skew():.4f}
峰度: {clean_data.kurtosis():.4f}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"分布分析图已保存至: {save_path}")
        else:
            plt.show()
    
    def compare_distributions(self, data_dict: Dict[str, pd.Series], 
                            save_path: Optional[str] = None):
        """比较多个数据的分布"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('多数据分布比较', fontsize=16, fontweight='bold')
        
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral']
        
        # 1. 直方图对比
        for i, (name, data) in enumerate(data_dict.items()):
            clean_data = data.dropna()
            if len(clean_data) > 0:
                axes[0, 0].hist(clean_data, bins=30, alpha=0.6, 
                              color=colors[i % len(colors)], label=name)
        axes[0, 0].set_title('直方图对比')
        axes[0, 0].set_xlabel('数值')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. 箱线图对比
        clean_data_list = []
        labels = []
        for name, data in data_dict.items():
            clean_data = data.dropna()
            if len(clean_data) > 0:
                clean_data_list.append(clean_data)
                labels.append(name)
        
        if clean_data_list:
            axes[0, 1].boxplot(clean_data_list, labels=labels, patch_artist=True)
            for i, box in enumerate(axes[0, 1].artists):
                box.set_facecolor(colors[i % len(colors)])
        axes[0, 1].set_title('箱线图对比')
        axes[0, 1].set_ylabel('数值')
        axes[0, 1].grid(alpha=0.3)
        
        # 3. 偏度对比
        names = []
        skewnesses = []
        for name, data in data_dict.items():
            clean_data = data.dropna()
            if len(clean_data) > 0:
                names.append(name)
                skewnesses.append(clean_data.skew())
        
        if names:
            bars = axes[1, 0].bar(names, skewnesses, color=colors[:len(names)])
            axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 0].set_title('偏度对比')
            axes[1, 0].set_ylabel('偏度')
            axes[1, 0].grid(alpha=0.3)
            
            # 添加数值标签
            for bar, skew in zip(bars, skewnesses):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{skew:.3f}', ha='center', va='bottom')
        
        # 4. 峰度对比
        kurtoses = []
        for name, data in data_dict.items():
            clean_data = data.dropna()
            if len(clean_data) > 0:
                kurtoses.append(clean_data.kurtosis())
        
        if names:
            bars = axes[1, 1].bar(names, kurtoses, color=colors[:len(names)])
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].set_title('峰度对比')
            axes[1, 1].set_ylabel('峰度')
            axes[1, 1].grid(alpha=0.3)
            
            # 添加数值标签
            for bar, kurt in zip(bars, kurtoses):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{kurt:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"分布比较图已保存至: {save_path}")
        else:
            plt.show()
    
    def analyze_feature_distributions(self, X: pd.DataFrame, save_dir: Optional[str] = None):
        """分析特征数据的分布"""
        print(f"\n{'='*60}")
        print("特征分布分析")
        print(f"{'='*60}")
        
        # 计算所有特征的偏度和峰度
        skew_kurt_data = []
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                clean_data = X[col].dropna()
                if len(clean_data) > 0:
                    skew_kurt_data.append({
                        '特征名': col,
                        '偏度': clean_data.skew(),
                        '峰度': clean_data.kurtosis(),
                        '样本数': len(clean_data),
                        '缺失值': len(X[col]) - len(clean_data)
                    })
        
        if not skew_kurt_data:
            print("没有找到数值型特征")
            return
        
        df_stats = pd.DataFrame(skew_kurt_data)
        
        # 按偏度绝对值排序
        df_stats['偏度_abs'] = df_stats['偏度'].abs()
        df_stats = df_stats.sort_values('偏度_abs', ascending=False)
        
        print(f"特征分布统计 (按偏度绝对值排序):")
        print(df_stats[['特征名', '偏度', '峰度', '样本数', '缺失值']].to_string(index=False))
        
        # 找出异常分布的特征
        high_skew = df_stats[df_stats['偏度_abs'] > 2.0]
        high_kurt = df_stats[df_stats['峰度'].abs() > 3.0]
        
        if not high_skew.empty:
            print(f"\n高偏度特征 (|偏度| > 2.0):")
            print(high_skew[['特征名', '偏度']].to_string(index=False))
        
        if not high_kurt.empty:
            print(f"\n高峰度特征 (|峰度| > 3.0):")
            print(high_kurt[['特征名', '峰度']].to_string(index=False))
        
        # 可视化前10个最偏的特征
        if len(df_stats) > 0:
            top_features = df_stats.head(min(10, len(df_stats)))['特征名'].tolist()
            
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                
                for feature in top_features[:5]:  # 只画前5个避免太多图
                    self.plot_distribution_analysis(
                        X[feature], 
                        name=f"特征: {feature}",
                        save_path=os.path.join(save_dir, f"distribution_{feature}.png")
                    )
        
        return df_stats
