"""
绘制K线图和交易信号
"""

import pandas as pd
import mplfinance as mpf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def plot_trading(df: pd.DataFrame, 
                title: str = "BTCUSDT Trading Chart",
                save_path: str = None,
                show_volume: bool = True,
                show_signals: bool = True):
    """
    绘制K线图和交易信号
    
    Parameters:
    -----------
    df : pd.DataFrame
        包含以下列的数据框：
        - open, high, low, close: 开盘价、最高价、最低价、收盘价
        - volume: 成交量
        - position: 持仓信号（1: 做多, -1: 做空, 0: 空仓）
        - nav: 净值曲线
    title : str
        图表标题
    save_path : str
        保存图表的路径，如果为None则显示图表
    show_volume : bool
        是否显示成交量
    show_signals : bool
        是否显示交易信号
    """
    # 确保数据框包含必要的列
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'position', 'nav']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"数据框必须包含以下列: {required_columns}")
    
    # 设置图表样式
    mc = mpf.make_marketcolors(
        up='red',           # 上涨为红色
        down='green',       # 下跌为绿色
        edge='inherit',     # 边框颜色继承
        wick='inherit',     # 上下影线颜色继承
        volume='in',        # 成交量颜色继承
        ohlc='inherit'      # K线颜色继承
    )
    
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='--',     # 网格线样式
        y_on_right=True,    # y轴在右边
        rc={'font.family': 'SimHei'}  # 使用中文字体
    )
    
    # 准备绘图数据
    df_plot = df.copy()
    df_plot.index = pd.to_datetime(df_plot.index)
    
    # 准备交易信号
    if show_signals:
        # 找出做多和做空的点
        long_signals = df_plot[df_plot['position'] == 1]
        short_signals = df_plot[df_plot['position'] == -1]
        
        # 创建做多和做空的标记
        long_markers = mpf.make_addplot(
            long_signals['close'],
            type='scatter',
            marker='^',
            markersize=100,
            color='red'
            # ax=0
        )
        
        short_markers = mpf.make_addplot(
            short_signals['close'],
            type='scatter',
            marker='v',
            markersize=100,
            color='green'
            # ax=0
        )
        
        # 创建净值曲线
        nav_plot = mpf.make_addplot(
            df_plot['nav'],
            panel=2,
            color='blue',
            title='净值曲线'
        )
        
        add_plots = [long_markers, short_markers, nav_plot]
    else:
        add_plots = []
    
    # 设置图表布局
    figsize = (15, 10)
    if show_volume:
        figsize = (15, 12)
    
    # 绘制图表
    mpf.plot(
        df_plot,
        type='candle',
        style=s,
        title=title,
        volume=show_volume,
        addplot=add_plots if show_signals else None,
        figsize=figsize,
        panel_ratios=(3, 1, 1) if show_signals else (3, 1),
        savefig=save_path if save_path else None,
        show_nontrading=False
    )
    
    if not save_path:
        plt.show()

def plot_strategy_performance(df: pd.DataFrame, 
                            title: str = "Strategy Performance",
                            save_path: str = None):
    """
    绘制策略表现图表
    
    Parameters:
    -----------
    df : pd.DataFrame
        包含以下列的数据框：
        - nav: 净值曲线
        - ret: 收益率
        - position: 持仓信号
    title : str
        图表标题
    save_path : str
        保存图表的路径，如果为None则显示图表
    """
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
    fig.suptitle(title, fontsize=16)
    
    # 绘制净值曲线
    ax1.plot(df.index, df['nav'], label='净值曲线', color='blue')
    ax1.set_title('净值曲线')
    ax1.grid(True)
    ax1.legend()
    
    # 绘制持仓信号
    ax2.plot(df.index, df['position'], label='持仓信号', color='red')
    ax2.set_title('持仓信号')
    ax2.grid(True)
    ax2.legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示图表
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

if __name__ == "__main__":
    # 示例使用
    from crypto_process import load_data, resample_data
    
    # 加载数据
    df = load_data("2024-01", "2024-02")
    df = resample_data(df, "1h")
    
    # 添加一些示例信号（这里只是示例，实际应该使用你的策略信号）
    df['position'] = 0
    df.loc[df['close'] > df['close'].shift(1), 'position'] = 1
    df.loc[df['close'] < df['close'].shift(1), 'position'] = -1
    
    # 计算收益率和净值
    df['ret'] = df['close'].pct_change()
    df['nav'] = (1 + df['ret'] * df['position']).cumprod()
    
    # 绘制图表
    plot_trading(df, title="BTCUSDT 1小时K线图", show_volume=True, show_signals=True)
    plot_strategy_performance(df, title="策略表现分析")