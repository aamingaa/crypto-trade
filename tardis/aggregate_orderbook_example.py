import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from util.orderbook_aggregation import OrderBookAggregator
import time

def load_and_aggregate_orderbook_data():
    """
    加载并聚合实际的订单簿数据
    """
    # 加载数据
    print("正在加载订单簿数据...")
    df = pd.read_csv('/Users/aming/project/python/crypto-trade/tardis/dataset/binance-futures_book_snapshot_5_2019-12-01_ETHUSDT.csv.gz')
    
    print(f"原始数据形状: {df.shape}")
    print(f"原始数据列: {df.columns.tolist()}")
    print("\n原始数据前5行:")
    print(df.head())
    
    # 转换时间戳
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
    df.set_index('timestamp', inplace=True)
    
    # 重置索引以便处理
    df = df.reset_index()
    
    # 创建聚合器
    aggregator = OrderBookAggregator(time_window_ms=1)
    
    # 方法1: 基本聚合（使用最后一个值）
    print("\n" + "="*50)
    print("方法1: 基本聚合（使用最后一个值）")
    print("="*50)
    
    start_time = time.time()
    aggregated_basic = aggregator.aggregate_orderbook_data(df)
    end_time = time.time()
    
    print(f"聚合耗时: {end_time - start_time:.2f}秒")
    print(f"聚合后数据形状: {aggregated_basic.shape}")
    print("聚合后数据前5行:")
    print(aggregated_basic.head())
    
    # 方法2: 成交量加权平均价格聚合
    print("\n" + "="*50)
    print("方法2: 成交量加权平均价格聚合")
    print("="*50)
    
    start_time = time.time()
    aggregated_vwap = aggregator.aggregate_with_volume_weighted_price(df)
    end_time = time.time()
    
    print(f"聚合耗时: {end_time - start_time:.2f}秒")
    print(f"聚合后数据形状: {aggregated_vwap.shape}")
    print("聚合后数据前5行:")
    print(aggregated_vwap.head())
    
    # 方法3: OHLC聚合
    print("\n" + "="*50)
    print("方法3: OHLC聚合")
    print("="*50)
    
    start_time = time.time()
    aggregated_ohlc = aggregator.aggregate_with_ohlc(df)
    end_time = time.time()
    
    print(f"聚合耗时: {end_time - start_time:.2f}秒")
    print(f"聚合后数据形状: {aggregated_ohlc.shape}")
    print("聚合后数据前5行:")
    print(aggregated_ohlc.head())
    
    # 获取订单簿快照
    print("\n" + "="*50)
    print("订单簿快照（5档）")
    print("="*50)
    
    start_time = time.time()
    snapshot = aggregator.get_orderbook_snapshot(df, method='last')
    end_time = time.time()
    
    print(f"快照生成耗时: {end_time - start_time:.2f}秒")
    print(f"快照数据形状: {snapshot.shape}")
    print("快照数据前5行:")
    print(snapshot.head())
    
    # 保存聚合结果
    print("\n" + "="*50)
    print("保存聚合结果")
    print("="*50)
    
    # 保存基本聚合结果
    aggregated_basic.to_csv('/Users/aming/project/python/crypto-trade/tardis/dataset/aggregated_basic_1ms.csv', index=False)
    print("基本聚合结果已保存到: aggregated_basic_1ms.csv")
    
    # 保存VWAP聚合结果
    aggregated_vwap.to_csv('/Users/aming/project/python/crypto-trade/tardis/dataset/aggregated_vwap_1ms.csv', index=False)
    print("VWAP聚合结果已保存到: aggregated_vwap_1ms.csv")
    
    # 保存OHLC聚合结果
    aggregated_ohlc.to_csv('/Users/aming/project/python/crypto-trade/tardis/dataset/aggregated_ohlc_1ms.csv', index=False)
    print("OHLC聚合结果已保存到: aggregated_ohlc_1ms.csv")
    
    # 保存订单簿快照
    snapshot.to_csv('/Users/aming/project/python/crypto-trade/tardis/dataset/orderbook_snapshot_1ms.csv', index=False)
    print("订单簿快照已保存到: orderbook_snapshot_1ms.csv")
    
    return aggregated_basic, aggregated_vwap, aggregated_ohlc, snapshot

def analyze_aggregation_results(aggregated_basic, aggregated_vwap, aggregated_ohlc, snapshot):
    """
    分析聚合结果
    """
    print("\n" + "="*50)
    print("聚合结果分析")
    print("="*50)
    
    # 分析时间分布
    print("\n1. 时间分布分析:")
    print(f"原始数据时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"聚合后数据时间范围: {aggregated_basic['timestamp_ms'].min()} 到 {aggregated_basic['timestamp_ms'].max()}")
    
    # 计算时间窗口数量
    unique_windows = aggregated_basic['timestamp_ms'].nunique()
    print(f"唯一时间窗口数量: {unique_windows}")
    
    # 分析价格分布
    print("\n2. 价格分布分析:")
    print(f"原始数据价格范围: {df['price'].min():.4f} - {df['price'].max():.4f}")
    print(f"基本聚合价格范围: {aggregated_basic['price'].min():.4f} - {aggregated_basic['price'].max():.4f}")
    print(f"VWAP聚合价格范围: {aggregated_vwap['price'].min():.4f} - {aggregated_vwap['price'].max():.4f}")
    
    # 分析成交量分布
    print("\n3. 成交量分布分析:")
    print(f"原始数据总成交量: {df['amount'].sum():.4f}")
    print(f"基本聚合总成交量: {aggregated_basic['amount'].sum():.4f}")
    print(f"VWAP聚合总成交量: {aggregated_vwap['amount'].sum():.4f}")
    
    # 验证数据一致性
    print("\n4. 数据一致性验证:")
    price_diff = abs(aggregated_basic['price'].sum() - aggregated_vwap['price'].sum())
    print(f"基本聚合与VWAP聚合价格差异: {price_diff:.6f}")
    
    amount_diff = abs(aggregated_basic['amount'].sum() - aggregated_vwap['amount'].sum())
    print(f"基本聚合与VWAP聚合成交量差异: {amount_diff:.6f}")
    
    # 分析订单簿快照
    print("\n5. 订单簿快照分析:")
    print(f"快照数量: {len(snapshot)}")
    
    # 计算买卖价差
    if 'bid_price_1' in snapshot.columns and 'ask_price_1' in snapshot.columns:
        spread = snapshot['ask_price_1'] - snapshot['bid_price_1']
        print(f"平均买卖价差: {spread.mean():.6f}")
        print(f"最小买卖价差: {spread.min():.6f}")
        print(f"最大买卖价差: {spread.max():.6f}")

def compare_different_time_windows():
    """
    比较不同时间窗口的聚合效果
    """
    print("\n" + "="*50)
    print("不同时间窗口聚合效果比较")
    print("="*50)
    
    # 加载数据
    df = pd.read_csv('/Users/aming/project/python/crypto-trade/tardis/dataset/binance-futures_book_snapshot_5_2019-12-01_ETHUSDT.csv.gz')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
    df = df.reset_index()
    
    time_windows = [1, 5, 10, 50, 100]  # 毫秒
    
    for window_ms in time_windows:
        print(f"\n时间窗口: {window_ms}ms")
        
        aggregator = OrderBookAggregator(time_window_ms=window_ms)
        
        start_time = time.time()
        aggregated = aggregator.aggregate_orderbook_data(df)
        end_time = time.time()
        
        print(f"  聚合耗时: {end_time - start_time:.2f}秒")
        print(f"  聚合后数据形状: {aggregated.shape}")
        print(f"  唯一时间窗口数量: {aggregated['timestamp_ms'].nunique()}")
        print(f"  数据压缩比: {len(df) / len(aggregated):.2f}:1")

if __name__ == "__main__":
    # 执行主要聚合
    aggregated_basic, aggregated_vwap, aggregated_ohlc, snapshot = load_and_aggregate_orderbook_data()
    
    # 分析结果
    analyze_aggregation_results(aggregated_basic, aggregated_vwap, aggregated_ohlc, snapshot)
    
    # 比较不同时间窗口
    compare_different_time_windows()

