"""
示例：使用改进的可视化工具绘制策略分析图
"""
import pandas as pd
import numpy as np
from utils.visualization import TradingVisualizer

# 示例1：使用MA_strategy_v2.py的结果
def visualize_ma_strategy():
    """可视化MA策略的结果"""
    from util.MA_strategy_v2 import MA_Strategy, load_daily_data
    
    # 加载数据
    start_date = "2025-05-01"
    end_date = "2025-06-30"
    df_price = load_daily_data(start_date, end_date, "15m", crypto="BNBUSDT")
    
    # 运行策略
    transactions, strategy_data = MA_Strategy(df_price, window_short=5, window_median=12, 
                                              window_long=30, loss_ratio=0.05)
    
    # 使用新的可视化工具
    visualizer = TradingVisualizer()
    visualizer.plot_strategy_comprehensive(
        strategy_data=strategy_data,
        transactions=transactions,
        title=f"MA策略综合分析 - BNBUSDT 15分钟 ({start_date} to {end_date})",
        save_path="/Users/aming/project/python/crypto-trade/output/evaluate_metric/ma_strategy_comprehensive.png"
    )
    
    print("MA策略可视化完成！")


# 示例2：模拟策略数据
def visualize_mock_strategy():
    """使用模拟数据演示可视化功能"""
    
    # 生成模拟数据
    dates = pd.date_range('2025-01-01', periods=1000, freq='1H')
    np.random.seed(42)
    
    # 模拟价格数据
    close = 100 + np.cumsum(np.random.randn(1000) * 0.5)
    
    # 模拟策略净值（使用简单移动平均策略）
    sma_short = pd.Series(close).rolling(window=20).mean()
    sma_long = pd.Series(close).rolling(window=50).mean()
    
    # 生成交易信号
    position = np.where(sma_short > sma_long, 1, 0)
    position = pd.Series(position, index=dates)
    
    # 计算收益率
    ret = pd.Series(close, index=dates).pct_change().fillna(0)
    strategy_ret = ret * position.shift(1).fillna(0)
    
    # 计算净值
    nav = (1 + strategy_ret).cumprod()
    benchmark = (1 + ret).cumprod()
    
    # 创建策略数据DataFrame
    strategy_data = pd.DataFrame({
        'close': close,
        'position': position,
        'ret': ret,
        'nav': nav,
        'benchmark': benchmark,
        'flag': position.diff().fillna(0)
    }, index=dates)
    
    # 生成交易记录
    buy_signals = strategy_data[strategy_data['flag'] > 0]
    sell_signals = strategy_data[strategy_data['flag'] < 0]
    
    transactions = pd.DataFrame({
        '买入日期': buy_signals.index,
        '买入价格': buy_signals['close'].values,
        '卖出日期': sell_signals.index[:len(buy_signals)],
        '卖出价格': sell_signals['close'].values[:len(buy_signals)]
    })
    
    # 使用可视化工具
    visualizer = TradingVisualizer()
    visualizer.plot_strategy_comprehensive(
        strategy_data=strategy_data,
        transactions=transactions,
        title="模拟策略综合分析",
        save_path="/Users/aming/project/python/crypto-trade/output/evaluate_metric/mock_strategy_comprehensive.png"
    )
    
    print("模拟策略可视化完成！")
    print(f"\n策略统计：")
    print(f"总收益率: {(nav.iloc[-1] - 1) * 100:.2f}%")
    print(f"基准收益率: {(benchmark.iloc[-1] - 1) * 100:.2f}%")
    print(f"超额收益: {((nav.iloc[-1] - benchmark.iloc[-1])) * 100:.2f}%")
    print(f"交易次数: {len(buy_signals)}")


# 示例3：仅展示策略数据（不需要transactions）
def visualize_strategy_only():
    """仅使用策略数据，不提供交易记录"""
    
    # 生成模拟数据
    dates = pd.date_range('2025-01-01', periods=500, freq='4H')
    np.random.seed(123)
    
    close = 50000 + np.cumsum(np.random.randn(500) * 100)
    ret = pd.Series(close).pct_change().fillna(0)
    
    # 简单的持仓逻辑
    position = np.where(ret > 0, 1, -1)  # 支持做空
    position = pd.Series(position, index=dates)
    
    strategy_ret = ret * position.shift(1).fillna(0)
    nav = (1 + strategy_ret).cumprod()
    benchmark = (1 + ret).cumprod()
    
    strategy_data = pd.DataFrame({
        'close': close,
        'position': position,
        'nav': nav,
        'benchmark': benchmark
    }, index=dates)
    
    # 不提供transactions，从position推断买卖点
    visualizer = TradingVisualizer()
    visualizer.plot_strategy_comprehensive(
        strategy_data=strategy_data,
        transactions=None,  # 不提供交易记录
        title="从持仓推断的策略分析",
        save_path="/Users/aming/project/python/crypto-trade/output/evaluate_metric/inferred_strategy_comprehensive.png"
    )
    
    print("策略可视化完成（从持仓推断）！")


if __name__ == "__main__":
    print("=" * 60)
    print("策略可视化示例")
    print("=" * 60)
    
    # 选择要运行的示例
    choice = input("\n请选择示例 (1=MA策略, 2=模拟策略, 3=持仓推断, 0=全部运行): ")
    
    if choice == "1":
        visualize_ma_strategy()
    elif choice == "2":
        visualize_mock_strategy()
    elif choice == "3":
        visualize_strategy_only()
    elif choice == "0":
        print("\n>>> 运行示例2: 模拟策略")
        visualize_mock_strategy()
        print("\n>>> 运行示例3: 持仓推断")
        visualize_strategy_only()
        print("\n>>> 运行示例1: MA策略（需要数据文件）")
        try:
            visualize_ma_strategy()
        except Exception as e:
            print(f"MA策略示例失败: {e}")
    else:
        print("无效选择，运行模拟策略示例...")
        visualize_mock_strategy()
    
    print("\n" + "=" * 60)
    print("可视化完成！请查看 output/evaluate_metric/ 目录")
    print("=" * 60)

