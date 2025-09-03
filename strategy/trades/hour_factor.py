import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --------------------------
# 1. 数据预处理
# --------------------------
def preprocess_trades(trades_df):
    """预处理逐笔成交数据"""
    # 转换时间格式（毫秒级时间戳）
    trades_df['trade_time'] = pd.to_datetime(trades_df['trade_time'], unit='ms')
    # 按时间排序
    trades_df = trades_df.sort_values('trade_time').reset_index(drop=True)
    # 计算单笔成交金额（假设volume单位为股，price为每股价格）
    trades_df['dollar_amount'] = trades_df['price'] * trades_df['volume']
    # 推断主动买卖方向（无订单簿数据时的近似方法）
    # 价格上涨且高于前一笔→主动买；价格下跌且低于前一笔→主动卖
    trades_df['price_change'] = trades_df['price'].diff()
    trades_df['direction'] = np.where(
        trades_df['price_change'] > 0, 1,  # 主动买
        np.where(trades_df['price_change'] < 0, -1, 0)  # 主动卖，平盘为0
    )
    return trades_df

# --------------------------
# 2. 生成Dollar Bar
# --------------------------
def generate_dollar_bars(trades_df, target_hour=1):
    """生成近似目标小时级的Dollar Bar"""
    # 计算小时级平均成交额作为阈值
    hourly_volume = trades_df.resample(f'{target_hour}H', on='trade_time')['dollar_amount'].sum()
    dollar_threshold = hourly_volume.mean()  # 阈值=目标小时级平均成交额
    
    # 生成Bar
    trades_df['cum_dollar'] = trades_df['dollar_amount'].cumsum()
    trades_df['bar_id'] = (trades_df['cum_dollar'] // dollar_threshold).astype(int)
    
    # 过滤不完整的最后一个Bar
    last_valid_id = trades_df['bar_id'].max() - 1
    trades_df = trades_df[trades_df['bar_id'] <= last_valid_id]
    
    # 提取每个Bar的时间区间和基础统计量
    bar_info = trades_df.groupby('bar_id').agg(
        start_time=('trade_time', 'min'),
        end_time=('trade_time', 'max'),
        total_dollar=('dollar_amount', 'sum'),
        price_open=('price', 'first'),
        price_close=('price', 'last'),
        price_high=('price', 'max'),
        price_low=('price', 'min'),
        trade_count=('price', 'count')
    ).reset_index()
    
    # 计算Bar的未来收益（预测目标：下一个Bar的涨跌幅）
    bar_info['future_return'] = (bar_info['price_close'].shift(-1) - bar_info['price_close']) / bar_info['price_close'] * 100
    
    return trades_df, bar_info, dollar_threshold

# --------------------------
# 3. 因子计算
# --------------------------
def calculate_factors(bar_id, group, bar_info):
    """计算单个Dollar Bar的因子"""
    factor = {
        'bar_id': bar_id,
        'start_time': group['trade_time'].min(),
        'end_time': group['trade_time'].max()
    }
    
    # 3.1 主动买成交占比
    total_dollar = group['dollar_amount'].sum()
    active_buy = group[group['direction'] == 1]['dollar_amount'].sum()
    factor['active_buy_ratio'] = active_buy / (total_dollar + 1e-6)
    
    # 3.2 大额订单偏向度
    if len(group) >= 10:  # 避免样本量不足
        large_threshold = group['dollar_amount'].quantile(0.9)
        large_buy = group[(group['direction'] == 1) & (group['dollar_amount'] >= large_threshold)]['dollar_amount'].sum()
        large_sell = group[(group['direction'] == -1) & (group['dollar_amount'] >= large_threshold)]['dollar_amount'].sum()
        factor['large_order_bias'] = (large_buy - large_sell) / (large_buy + large_sell + 1e-6)
    else:
        factor['large_order_bias'] = 0
    
    # 3.3 成交间隔波动率
    group = group.sort_values('trade_time')
    group['interval'] = group['trade_time'].diff().dt.total_seconds() * 1000  # 毫秒间隔
    interval_mean = group['interval'].mean()
    factor['interval_volatility'] = group['interval'].std() / (interval_mean + 1e-6) if interval_mean != 0 else 0
    
    # 3.4 量价协同度
    price_change = (group['price'].iloc[-1] - group['price'].iloc[0]) / group['price'].iloc[0] * 100
    factor['price_volume_synergy'] = price_change * (2 * factor['active_buy_ratio'] - 1)  # 标准化协同度
    
    # 3.5 流动性消耗速率
    price_range = group['price'].max() - group['price'].min()
    mid_price = (group['price'].max() + group['price'].min()) / 2
    slipped_fee = price_range / mid_price if mid_price != 0 else 0
    factor['liquidity_depletion'] = slipped_fee / (total_dollar / 1e6)  # 每百万成交的滑点率
    
    return pd.Series(factor)

def compute_all_factors(trades_df, bar_info):
    """批量计算所有Bar的因子"""
    # 按Bar分组计算基础因子
    factors = []
    for bar_id, group in trades_df.groupby('bar_id'):
        if len(group) < 50:  # 过滤成交笔数太少的Bar
            continue
        factors.append(calculate_factors(bar_id, group, bar_info))
    
    factors_df = pd.DataFrame(factors)
    # 合并Bar的未来收益（预测目标）
    factors_df = factors_df.merge(
        bar_info[['bar_id', 'future_return']],
        on='bar_id',
        how='left'
    )
    
    # 计算滑动窗口因子：成交金额斜率（过去3个Bar）
    factors_df = factors_df.sort_values('bar_id')
    factors_df['dollar_slope'] = factors_df['total_dollar'].rolling(3, min_periods=2).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
    )
    
    return factors_df

# --------------------------
# 4. 因子优化
# --------------------------
def optimize_factors(factors_df):
    """因子优化：去极端值、标准化、平滑"""
    optimized = factors_df.copy()
    factor_cols = [col for col in optimized.columns if col not in ['bar_id', 'start_time', 'end_time', 'future_return']]
    
    # 4.1 去极端值（分位数截断）
    for col in factor_cols:
        optimized[col] = optimized[col].clip(
            optimized[col].quantile(0.01),
            optimized[col].quantile(0.99)
        )
    
    # 4.2 标准化（Z-score）
    for col in factor_cols:
        mean = optimized[col].mean()
        std = optimized[col].std()
        optimized[col] = (optimized[col] - mean) / (std + 1e-6)
    
    # 4.3 平滑处理（指数移动平均）
    for col in factor_cols:
        optimized[f'smoothed_{col}'] = optimized[col].ewm(span=3).mean()
    
    # 4.4 因子组合（IC加权）
    ic_values = {col: optimized[col].corr(optimized['future_return']) for col in factor_cols}
    valid_cols = [col for col, ic in ic_values.items() if not np.isnan(ic)]
    total_ic = sum(abs(ic_values[col]) for col in valid_cols)
    
    if total_ic > 0:
        optimized['combined_factor'] = sum(
            optimized[col] * (abs(ic_values[col]) / total_ic) 
            for col in valid_cols
        )
    else:
        optimized['combined_factor'] = 0
    
    return optimized, ic_values

# --------------------------
# 5. 因子验证
# --------------------------
def validate_factors(optimized_df):
    """验证因子有效性"""
    results = {}
    factor_cols = [col for col in optimized_df.columns if 'smoothed_' in col or col == 'combined_factor']
    
    # 5.1 计算所有因子的IC值
    ic_results = {col: optimized_df[col].corr(optimized_df['future_return']) for col in factor_cols}
    results['ic'] = ic_results
    
    # 5.2 样本内外测试（6:4分割）
    train, test = train_test_split(optimized_df.dropna(), test_size=0.4, shuffle=False)
    train_ic = {col: train[col].corr(train['future_return']) for col in factor_cols}
    test_ic = {col: test[col].corr(test['future_return']) for col in factor_cols}
    results['train_ic'] = train_ic
    results['test_ic'] = test_ic
    
    # 5.3 因子重要性（随机森林）
    feature_cols = [col for col in factor_cols if col != 'combined_factor']
    if len(feature_cols) > 0 and 'future_return' in optimized_df.columns:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train[feature_cols].dropna(), train['future_return'].dropna())
        results['feature_importance'] = dict(zip(feature_cols, model.feature_importances_))
    
    # 5.4 绘制IC曲线
    plt.figure(figsize=(12, 6))
    rolling_ic = optimized_df[factor_cols].rolling(20).corr(optimized_df['future_return'])
    for col in factor_cols:
        plt.plot(rolling_ic.index, rolling_ic[col], label=col)
    plt.title('Rolling IC (20-period window)')
    plt.legend()
    plt.savefig('factor_rolling_ic.png')
    plt.close()
    
    return results

# --------------------------
# 主流程
# --------------------------
def main():
    # 示例数据生成（实际使用时替换为真实数据）
    # 生成10天的逐笔成交数据（每100ms一笔）
    n_samples = 10 * 24 * 3600 * 10  # 10天×24小时×3600秒×10笔/秒
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='100ms')
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)  # 模拟价格序列
    volumes = np.random.randint(10, 1000, size=n_samples)  # 模拟成交量
    
    trades_df = pd.DataFrame({
        'trade_time': timestamps.astype(np.int64) // 10**6,  # 毫秒级时间戳
        'price': prices,
        'volume': volumes
    })
    
    # 1. 数据预处理
    trades_df = preprocess_trades(trades_df)
    
    # 2. 生成Dollar Bar（目标1小时级）
    trades_with_bar, bar_info, threshold = generate_dollar_bars(trades_df, target_hour=1)
    print(f"Dollar Bar阈值: {threshold:.2f} 元，平均Bar时长: {(bar_info['end_time'] - bar_info['start_time']).mean()}")
    
    # 3. 计算因子
    factors_df = compute_all_factors(trades_with_bar, bar_info)
    print(f"计算完成 {len(factors_df)} 个Bar的因子")
    
    # 4. 因子优化
    optimized_df, ic_values = optimize_factors(factors_df)
    print("因子IC值:")
    for col, ic in ic_values.items():
        print(f"  {col}: {ic:.4f}")
    
    # 5. 因子验证
    validation_results = validate_factors(optimized_df)
    print("\n样本外IC值:")
    for col, ic in validation_results['test_ic'].items():
        print(f"  {col}: {ic:.4f}")
    
    # 保存结果
    optimized_df.to_csv('optimized_factors.csv', index=False)
    print("\n优化后的因子已保存至 optimized_factors.csv")

if __name__ == "__main__":
    main()
