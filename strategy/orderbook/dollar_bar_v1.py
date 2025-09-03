import pandas as pd
import numpy as np
from scipy.stats import linregress

# 假设df为原始数据，包含列：['time','ap0', 'av0', 'bp0', 'bv0', ..., 'ap4', 'av4', 'bp4', 'bv4']
# 其中time为毫秒级时间戳（如13位整数），ap/bp为卖/买价，av/bv为卖/买量

def preprocess_data(df):
    """数据预处理：转换时间格式并计算近似成交额"""
    df['time'] = pd.to_datetime(df['timestamp'], unit='us')  # 转换时间戳
    df = df.sort_values('time').set_index('time', drop=True)
    selected_columns = ['symbol','asks[0].price', 'asks[0].amount', 'bids[0].price', 'bids[0].amount', 'asks[1].price', 'asks[1].amount', 'bids[1].price', 'bids[1].amount', 'asks[2].price', 'asks[2].amount', 'bids[2].price', 'bids[2].amount', 'asks[3].price', 'asks[3].amount', 'bids[3].price', 'bids[3].amount', 'asks[4].price', 'asks[4].amount', 'bids[4].price', 'bids[4].amount']
    df = df.loc[:, selected_columns]
    
    df.columns = ['symbol','ap0', 'av0', 'bp0', 'bv0', 'ap1', 'av1', 'bp1', 'bv1', 'ap2', 'av2', 'bp2', 'bv2', 'ap3', 'av3', 'bp3', 'bv3', 'ap4', 'av4', 'bp4', 'bv4']
    
    # 计算中间价（用于近似成交价格）
    df['mid_price'] = (df['ap0'] + df['bp0']) / 2
    
    # 估算每笔快照的成交额变化（近似，实际需逐笔成交数据）
    # 逻辑：盘口量变化 * 中间价 = 该时段内的近似成交额
    df['delta_bv0'] = df['bv0'].diff().abs()  # 买一量变化绝对值
    df['delta_av0'] = df['av0'].diff().abs()  # 卖一量变化绝对值
    df['approx_volume'] = (df['delta_bv0'] + df['delta_av0']) * df['mid_price']  # 近似成交额（美元）
    df['approx_volume'] = df['approx_volume'].fillna(0)  # 首行填充0
    
    return df

def generate_dollar_bars(df, dollar_threshold=1_000_000):
    """生成Dollar Bar：累计成交额达到阈值时形成一个bar"""
    # 累计成交额
    df['cum_dollar'] = df['approx_volume'].cumsum()
    
    # 为每个数据点分配bar编号
    df['bar_id'] = (df['cum_dollar'] // dollar_threshold).astype(int)
    
    # 过滤最后一个不完整的bar（累计额未达阈值）
    last_bar_id = df['bar_id'].max()
    df = df[df['bar_id'] < last_bar_id].copy()
    
    return df

def calculate_order_flow_factors(bar_data):
    """计算订单流不平衡类因子（复用逻辑，适配bar数据）"""
    factors = {}
    
    # 1. Bar内主动买订单量占比
    bar_data['delta_ap0'] = bar_data['ap0'].diff()
    bar_data['delta_bp0'] = bar_data['bp0'].diff()
    
    bar_data['active_sell'] = np.where(
        (bar_data['delta_ap0'] < 0) & (bar_data['av0'].diff() < 0),
        -bar_data['av0'].diff(),
        0
    )
    bar_data['active_buy'] = np.where(
        (bar_data['delta_bp0'] > 0) & (bar_data['bv0'].diff() < 0),
        -bar_data['bv0'].diff(),
        0
    )
    
    total_active = bar_data['active_buy'].sum() + bar_data['active_sell'].sum()
    factors['active_buy_ratio'] = bar_data['active_buy'].sum() / total_active if total_active != 0 else 0
    
    # 2. Bar内订单流不平衡波动率
    order_imbalance = bar_data['active_buy'] - bar_data['active_sell']
    factors['order_imbalance_vol'] = order_imbalance.std()
    
    # 3. Bar内大额订单流偏向度
    avg_order_size = (bar_data['active_buy'].mean() + bar_data['active_sell'].mean()) / 2
    large_order_threshold = avg_order_size * 5 if avg_order_size != 0 else 100
    
    large_buy = bar_data[bar_data['active_buy'] > large_order_threshold]['active_buy'].sum()
    large_sell = bar_data[bar_data['active_sell'] > large_order_threshold]['active_sell'].sum()
    factors['large_order_bias'] = (large_buy - large_sell) / (large_buy + large_sell + 1e-6)
    
    return factors

def calculate_liquidity_factors(bar_data):
    """计算流动性结构类因子（适配Dollar Bar）"""
    factors = {}
    
    # 1. Bar内平均买卖价差率
    mid_price = (bar_data['ap0'] + bar_data['bp0']) / 2
    spread = bar_data['ap0'] - bar_data['bp0']
    spread_ratio = spread / mid_price
    factors['avg_spread_ratio'] = spread_ratio.mean()
    
    # 2. Bar内深度覆盖度（买五档总深度/卖五档总深度）
    buy_depth = bar_data[['bv0', 'bv1', 'bv2', 'bv3', 'bv4']].sum(axis=1)
    sell_depth = bar_data[['av0', 'av1', 'av2', 'av3', 'av4']].sum(axis=1)
    factors['depth_coverage'] = (buy_depth / (sell_depth + 1e-6)).mean()
    
    # 3. Bar内流动性突变频率（基于bar内5分钟子窗口）
    # 按5分钟分割当前bar（若bar时长超过5分钟）
    bar_start = bar_data.index.min()
    bar_end = bar_data.index.max()
    sub_windows = pd.date_range(start=bar_start, end=bar_end, freq='5T')
    
    if len(sub_windows) < 2:
        factors['liquidity_jump_freq'] = 0
    else:
        jump_count = 0
        total_sub = 0
        for i in range(len(sub_windows)-1):
            sub_mask = (bar_data['time'] >= sub_windows[i]) & (bar_data['time'] < sub_windows[i+1])
            sub_data = bar_data[sub_mask]
            if len(sub_data) < 10:  # 子窗口数据不足
                continue
            total_sub += 1
            
            # 计算价差和深度变化
            sub_spread = sub_data['ap0'] - sub_data['bp0']
            sub_buy_depth = sub_data[['bv0', 'bv1', 'bv2', 'bv3', 'bv4']].sum(axis=1)
            sub_sell_depth = sub_data[['av0', 'av1', 'av2', 'av3', 'av4']].sum(axis=1)
            
            spread_jump = sub_spread.pct_change().abs().max() > 0.2
            depth_jump = (sub_buy_depth.pct_change().abs().max() > 0.2) | \
                         (sub_sell_depth.pct_change().abs().max() > 0.2)
            if spread_jump or depth_jump:
                jump_count += 1
        
        factors['liquidity_jump_freq'] = jump_count / total_sub if total_sub > 0 else 0
    
    return factors

def calculate_price_book_factors(bar_data):
    """计算价格-订单簿联动类因子（适配Dollar Bar）"""
    factors = {}
    
    # 1. Bar内深度-价格弹性
    # 按10%成交额分割bar为子窗口（而非固定时间）
    bar_cum_dollar = bar_data['cum_dollar'].iloc[-1] - bar_data['cum_dollar'].iloc[0]
    sub_thresholds = np.linspace(0, bar_cum_dollar, 11)[1:-1]  # 10个等分点
    sub_windows = []
    
    current_cum = 0
    start_idx = 0
    for i, row in bar_data.iterrows():
        current_cum += row['approx_volume']
        if current_cum >= sub_thresholds[len(sub_windows)]:
            sub_windows.append(bar_data.iloc[start_idx:i+1])
            start_idx = i+1
            if len(sub_windows) == len(sub_thresholds):
                break
    
    if len(sub_windows) < 3:
        factors['depth_price_elasticity'] = 0
    else:
        # 提取子窗口的价格和买深度
        sub_prices = [sw['mid_price'].mean() for sw in sub_windows]
        sub_buy_depths = [sw[['bv0', 'bv1', 'bv2', 'bv3', 'bv4']].sum(axis=1).mean() for sw in sub_windows]
        
        price_pct_change = np.diff(sub_prices) / sub_prices[:-1] * 100
        depth_pct_change = np.diff(sub_buy_depths) / sub_buy_depths[:-1] * 100
        
        up_sample = price_pct_change > 0
        if up_sample.sum() >= 2:
            slope, _, _, _, _ = linregress(price_pct_change[up_sample], depth_pct_change[up_sample])
            factors['depth_price_elasticity'] = slope
        else:
            factors['depth_price_elasticity'] = 0
    
    # 2. Bar内价差收敛速度
    first_spread = (bar_data.iloc[0]['ap0'] - bar_data.iloc[0]['bp0'])
    last_spread = (bar_data.iloc[-1]['ap0'] - bar_data.iloc[-1]['bp0'])
    factors['spread_convergence_speed'] = (first_spread - last_spread) / (first_spread + 1e-6)
    
    # 3. Bar内摆盘价格偏离度
    bar_avg_price = bar_data['mid_price'].mean()
    bp_deviation = (bar_data['bp0'] - bar_avg_price).abs() / bar_avg_price
    factors['bp_deviation'] = bp_deviation.mean()
    
    return factors

def calculate_dynamic_factors(bar_data):
    """计算订单簿动态变化类因子（适配Dollar Bar）"""
    factors = {}
    
    # 1. Bar内深度递增斜率（按成交额比例分割子窗口）
    bar_cum_dollar = bar_data['cum_dollar'].iloc[-1] - bar_data['cum_dollar'].iloc[0]
    sub_thresholds = np.linspace(0, bar_cum_dollar, 6)[1:-1]  # 5个等分点
    sub_windows = []
    
    current_cum = 0
    start_idx = 0
    for i, row in bar_data.iterrows():
        current_cum += row['approx_volume']
        if current_cum >= sub_thresholds[len(sub_windows)]:
            sub_windows.append(bar_data.iloc[start_idx:i+1])
            start_idx = i+1
            if len(sub_windows) == len(sub_thresholds):
                break
    
    if len(sub_windows) >= 3:
        buy_depth_series = [sw[['bv0', 'bv1', 'bv2', 'bv3', 'bv4']].sum(axis=1).mean() for sw in sub_windows]
        x = np.arange(len(buy_depth_series))
        slope, _, _, _, _ = linregress(x, buy_depth_series)
        factors['buy_depth_slope'] = slope
    else:
        factors['buy_depth_slope'] = 0
    
    # 2. Bar内撤单率趋势（前半段 vs 后半段）
    bar_mid_cum = bar_cum_dollar / 2
    current_cum = 0
    mid_idx = 0
    for i, row in bar_data.iterrows():
        current_cum += row['approx_volume']
        if current_cum >= bar_mid_cum:
            mid_idx = i
            break
    
    first_half = bar_data.iloc[:mid_idx+1]
    second_half = bar_data.iloc[mid_idx+1:]
    
    # 计算撤单率
    def get_cancel_rate(data):
        data['total_bid'] = data[['bv0', 'bv1', 'bv2', 'bv3', 'bv4']].sum(axis=1)
        data['total_ask'] = data[['av0', 'av1', 'av2', 'av3', 'av4']].sum(axis=1)
        cancel_bid = data[data['total_bid'].diff() < 0]['total_bid'].diff().abs().sum()
        cancel_ask = data[data['total_ask'].diff() < 0]['total_ask'].diff().abs().sum()
        total_depth = data[['total_bid', 'total_ask']].sum().sum()
        return (cancel_bid + cancel_ask) / (total_depth + 1e-6)
    
    first_cancel = get_cancel_rate(first_half.copy())
    second_cancel = get_cancel_rate(second_half.copy())
    factors['cancel_rate_trend'] = second_cancel - first_cancel
    
    # 3. Bar内跨档订单占比趋势
    if len(sub_windows) >= 2:
        cross_ratios = []
        for sw in sub_windows:
            cross_buy = (sw['ap1'] / sw['ap0'] < 0.99).mean()  # 跨档买比例
            cross_sell = (sw['bp1'] / sw['bp0'] > 1.01).mean()  # 跨档卖比例
            cross_ratios.append((cross_buy + cross_sell) / 2)
        factors['cross_level_trend'] = cross_ratios[-1] - cross_ratios[0]
    else:
        factors['cross_level_trend'] = 0
    
    return factors

def compute_all_factors(raw_df, dollar_threshold=1_000_000):
    """主函数：生成Dollar Bar并计算因子"""
    # 预处理数据
    df = preprocess_data(raw_df)
    
    # 生成Dollar Bar
    df_with_bars = generate_dollar_bars(df, dollar_threshold)
    if len(df_with_bars) == 0:
        return pd.DataFrame()
    
    # 按Bar分组计算因子
    bar_factors = []
    for bar_id, group in df_with_bars.groupby('bar_id'):
        if len(group) < 50:  # 过滤数据量太少的bar
            continue
        
        # 计算各类因子
        order_flow = calculate_order_flow_factors(group.copy())
        liquidity = calculate_liquidity_factors(group.copy())
        price_book = calculate_price_book_factors(group.copy())
        dynamic = calculate_dynamic_factors(group.copy())
        
        # 合并因子并添加bar信息
        all_factors = {
            'bar_id': bar_id,
            'start_time': group['time'].min(),
            'end_time': group['time'].max(),
            'duration_min': (group['time'].max() - group['time'].min()).total_seconds() / 60,
            **order_flow,** liquidity,
            **price_book,** dynamic
        }
        bar_factors.append(all_factors)
    
    return pd.DataFrame(bar_factors).set_index('bar_id')

from datetime import datetime, timedelta
def generate_date_range(start_date, end_date):    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    return date_list

# 使用示例（假设raw_df为原始数据）：
# '2019-12-01'
date_list = generate_date_range('2025-01-01', '2025-01-02')
# print(date_list)
raw_df = []
for date in date_list:
    raw_df.append(pd.read_csv(f'/Volumes/Ext-Disk/data/futures/um/tardis/orderbook/ETHUSDT/binance_book_snapshot_5_{date}_ETHUSDT.csv.gz'))

raw_df = pd.concat(raw_df)
# print(raw_df.head())
# print(raw_df.tail())

factors_df = compute_all_factors(raw_df)
print(factors_df.head())  # 查看前5小时的因子结果



# raw_df = pd.read_csv('/Volumes/Ext-Disk/data/futures/um/tardis/orderbook/ETHUSDT/binance_book_snapshot_5_2019-12-01_ETHUSDT.csv.gz')
# # raw_df = pd.read_csv('order_book_data.csv')  # 读取你的毫秒级订单簿数据
# factors_df = compute_all_factors(raw_df)
# print(factors_df.head())  # 查看前5小时的因子结果
