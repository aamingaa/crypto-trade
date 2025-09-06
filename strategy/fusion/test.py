import pandas as pd
import numpy as np

def calculate_high_freq_downside_ratio_by_bars(
    df_minute_returns: pd.DataFrame,
    lookback_bars: int = 480  # 回溯的bar数量（如2个交易日=480分钟）
) -> pd.DataFrame:
    """
    基于bar数量计算“高频下行波动占比”因子
    
    参数：
    df_minute_returns: 分钟级收益DataFrame，index为时间（含时分），columns为股票代码，值为分钟收益
    lookback_bars: 计算因子时的回溯bar数量（如480=2个交易日×240分钟）
    
    返回：
    factor_df: 因子值DataFrame，index为分钟级时间戳，columns为股票代码
               每个时间点的值表示“过去lookback_bars个bar的下行波动占比均值”
    """
    # ---------- 步骤1：分离“下行收益” ----------
    # 仅保留收益<0的部分（收益≥0时置为0）
    downside_returns = df_minute_returns.where(df_minute_returns < 0, 0)
    
    # ---------- 步骤2：计算“下行收益平方”与“总收益平方” ----------
    downside_returns_squared = downside_returns **2  # 下行收益的平方
    total_returns_squared = df_minute_returns** 2    # 所有收益的平方
    
    # ---------- 步骤3：滚动窗口计算“下行波动占比” ----------
    # 滚动窗口内的下行平方和
    rolling_downside_sum = downside_returns_squared.rolling(window=lookback_bars, min_periods=lookback_bars).sum()
    # 滚动窗口内的总平方和
    rolling_total_sum = total_returns_squared.rolling(window=lookback_bars, min_periods=lookback_bars).sum()
    
    # 下行波动占比 = 下行平方和 / 总平方和
    factor_df = rolling_downside_sum / rolling_total_sum
    
    # 处理除零情况（总平方和为0时置为NaN）
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    
    return factor_df


# 测试代码
if __name__ == "__main__":
    # 构造测试数据：2只股票，10个交易日（2400分钟）的分钟收益
    np.random.seed(42)
    total_minutes = 240 * 10  # 10个交易日×240分钟
    dates = pd.date_range(start='2025-01-01', periods=total_minutes, freq='T')  # 分钟级时间索引
    stocks = ['stock1', 'stock2']
    
    # 构造收益数据：stock2下行波动更大
    returns_stock1 = np.random.normal(0, 0.001, total_minutes)
    returns_stock2 = np.random.normal(0, 0.001, total_minutes)
    returns_stock2[returns_stock2 > 0] *= 0.5  # 上行收益缩小
    returns_stock2[returns_stock2 < 0] *= 2    # 下行收益放大
    
    df_minute_returns = pd.DataFrame(
        data={'stock1': returns_stock1, 'stock2': returns_stock2},
        index=dates
    )
    
    # 计算因子（回溯2个交易日=480个bar）
    factor_df = calculate_high_freq_downside_ratio_by_bars(df_minute_returns, lookback_bars=480)
    
    # 查看结果（最后10条数据）
    print("因子值（最后10条）：")
    print(factor_df.tail(10))
    
    # 验证逻辑：stock2的下行占比应显著高于stock1
    print("\n股票平均下行波动占比：")
    print(factor_df.mean())
