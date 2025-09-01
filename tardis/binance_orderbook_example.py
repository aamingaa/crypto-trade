import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from util.binance_orderbook_api import BinanceFuturesOrderBookAPI
from util.orderbook_aggregation import OrderBookAggregator

def demo_orderbook_data_collection():
    """
    演示订单簿数据收集和处理
    """
    print("=== Binance合约订单簿数据收集演示 ===")
    
    # 创建API客户端
    api = BinanceFuturesOrderBookAPI()
    
    # 设置参数
    symbol = "BTCUSDT"
    collection_duration = 60  # 收集60秒数据
    collection_interval = 1  # 每秒收集一次
    
    print(f"开始收集 {symbol} 订单簿数据...")
    print(f"收集时长: {collection_duration} 秒")
    print(f"收集间隔: {collection_interval} 秒")
    
    # 收集数据
    data_list = []
    start_time = time.time()
    
    while time.time() - start_time < collection_duration:
        try:
            # 获取综合订单簿数据
            data = api.get_comprehensive_orderbook_data(symbol)
            
            if data:
                data_list.append(data)
                current_time = datetime.now().strftime("%H:%M:%S")
                mid_price = data['order_book_metrics']['mid_price']
                spread = data['order_book_metrics']['spread']
                print(f"[{current_time}] 中间价: {mid_price:.2f}, 价差: {spread:.2f}")
            
            time.sleep(collection_interval)
            
        except KeyboardInterrupt:
            print("\n数据收集被用户中断")
            break
        except Exception as e:
            print(f"收集数据时出错: {e}")
            time.sleep(collection_interval)
    
    print(f"\n数据收集完成，共收集 {len(data_list)} 条记录")
    
    # 保存原始数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_filename = f"raw_orderbook_{symbol}_{timestamp}.json"
    api.save_orderbook_data(data_list, raw_filename)
    
    return data_list

def process_orderbook_data(data_list):
    """
    处理订单簿数据，转换为适合聚合的格式
    """
    print("\n=== 处理订单簿数据 ===")
    
    processed_data = []
    
    for data in data_list:
        timestamp = datetime.fromisoformat(data['timestamp'])
        
        # 处理买盘数据
        for bid in data['bids_df'][:5]:  # 只取前5档
            processed_data.append({
                'timestamp': timestamp,
                'local_timestamp': timestamp,
                'exchange_timestamp': timestamp,
                'sequence_number': len(processed_data),
                'symbol': data['symbol'],
                'exchange': 'binance-futures',
                'channel': 'book_snapshot_5',
                'side': 'buy',
                'price': bid['price'],
                'amount': bid['quantity'],
                'id': len(processed_data)
            })
        
        # 处理卖盘数据
        for ask in data['asks_df'][:5]:  # 只取前5档
            processed_data.append({
                'timestamp': timestamp,
                'local_timestamp': timestamp,
                'exchange_timestamp': timestamp,
                'sequence_number': len(processed_data),
                'symbol': data['symbol'],
                'exchange': 'binance-futures',
                'channel': 'book_snapshot_5',
                'side': 'sell',
                'price': ask['price'],
                'amount': ask['quantity'],
                'id': len(processed_data)
            })
    
    # 转换为DataFrame
    df = pd.DataFrame(processed_data)
    print(f"处理后的数据形状: {df.shape}")
    print("数据前5行:")
    print(df.head())
    
    return df

def aggregate_orderbook_data(df):
    """
    聚合订单簿数据
    """
    print("\n=== 聚合订单簿数据 ===")
    
    # 创建聚合器
    aggregator = OrderBookAggregator(time_window_ms=1000)  # 1秒聚合
    
    # 使用不同方法聚合
    print("1. 基本聚合:")
    aggregated_basic = aggregator.aggregate_orderbook_data(df)
    print(f"聚合后数据形状: {aggregated_basic.shape}")
    
    print("\n2. VWAP聚合:")
    aggregated_vwap = aggregator.aggregate_with_volume_weighted_price(df)
    print(f"聚合后数据形状: {aggregated_vwap.shape}")
    
    print("\n3. 生成订单簿快照:")
    snapshot = aggregator.get_orderbook_snapshot(df, method='last')
    print(f"快照数据形状: {snapshot.shape}")
    
    return aggregated_basic, aggregated_vwap, snapshot

def analyze_orderbook_metrics(data_list):
    """
    分析订单簿指标
    """
    print("\n=== 订单簿指标分析 ===")
    
    # 提取指标数据
    metrics_data = []
    for data in data_list:
        metrics = data['order_book_metrics']
        metrics['timestamp'] = data['timestamp']
        metrics_data.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
    
    print("指标统计:")
    print(metrics_df.describe())
    
    # 分析价差变化
    print("\n价差分析:")
    print(f"平均价差: {metrics_df['spread'].mean():.6f}")
    print(f"价差标准差: {metrics_df['spread'].std():.6f}")
    print(f"最小价差: {metrics_df['spread'].min():.6f}")
    print(f"最大价差: {metrics_df['spread'].max():.6f}")
    
    # 分析订单不平衡
    print("\n订单不平衡分析:")
    print(f"平均不平衡: {metrics_df['order_imbalance'].mean():.6f}")
    print(f"不平衡标准差: {metrics_df['order_imbalance'].std():.6f}")
    
    return metrics_df

def save_processed_data(aggregated_basic, aggregated_vwap, snapshot, metrics_df):
    """
    保存处理后的数据
    """
    print("\n=== 保存处理后的数据 ===")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存聚合数据
    aggregated_basic.to_csv(f"aggregated_basic_{timestamp}.csv", index=False)
    print("基本聚合数据已保存")
    
    aggregated_vwap.to_csv(f"aggregated_vwap_{timestamp}.csv", index=False)
    print("VWAP聚合数据已保存")
    
    snapshot.to_csv(f"orderbook_snapshot_{timestamp}.csv", index=False)
    print("订单簿快照已保存")
    
    metrics_df.to_csv(f"orderbook_metrics_{timestamp}.csv", index=False)
    print("订单簿指标已保存")

def demo_realtime_monitoring():
    """
    演示实时监控
    """
    print("\n=== 实时监控演示 ===")
    
    api = BinanceFuturesOrderBookAPI()
    symbol = "BTCUSDT"
    
    def orderbook_callback(data):
        """WebSocket回调函数"""
        try:
            # 解析订单簿数据
            if 'data' in data:
                order_book = data['data']
                timestamp = datetime.fromtimestamp(order_book['E'] / 1000)
                
                # 计算基本指标
                best_bid = float(order_book['bids'][0][0])
                best_ask = float(order_book['asks'][0][0])
                mid_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
                
                current_time = timestamp.strftime("%H:%M:%S.%f")[:-3]
                print(f"[{current_time}] 中间价: {mid_price:.2f}, 价差: {spread:.2f}")
                
        except Exception as e:
            print(f"处理WebSocket数据时出错: {e}")
    
    # 启动WebSocket监控
    print("启动WebSocket实时监控...")
    api.start_websocket_orderbook(symbol, orderbook_callback)
    
    # 监控30秒
    time.sleep(30)
    
    # 停止监控
    api.stop_websocket()
    print("实时监控已停止")

def main():
    """
    主函数
    """
    print("Binance合约订单簿数据处理演示")
    print("=" * 50)
    
    # 1. 收集数据
    data_list = demo_orderbook_data_collection()
    
    if not data_list:
        print("没有收集到数据，程序结束")
        return
    
    # 2. 处理数据
    df = process_orderbook_data(data_list)
    
    # 3. 聚合数据
    aggregated_basic, aggregated_vwap, snapshot = aggregate_orderbook_data(df)
    
    # 4. 分析指标
    metrics_df = analyze_orderbook_metrics(data_list)
    
    # 5. 保存数据
    save_processed_data(aggregated_basic, aggregated_vwap, snapshot, metrics_df)
    
    print("\n=== 演示完成 ===")
    print("所有数据已处理并保存完成")
    
    # 询问是否启动实时监控
    try:
        choice = input("\n是否启动实时监控演示？(y/n): ")
        if choice.lower() == 'y':
            demo_realtime_monitoring()
    except KeyboardInterrupt:
        print("\n程序被用户中断")

if __name__ == "__main__":
    main()

