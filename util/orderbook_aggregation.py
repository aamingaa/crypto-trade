import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderBookAggregator:
    """
    五档订单簿数据聚合器
    将ns级别的订单簿数据聚合到ms级别
    """
    
    def __init__(self, time_window_ms: int = 1):
        """
        初始化聚合器
        
        Args:
            time_window_ms: 时间窗口大小（毫秒），默认为1ms
        """
        self.time_window_ms = time_window_ms
        self.logger = logging.getLogger(__name__)
    
    def aggregate_orderbook_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将ns级别的订单簿数据聚合到ms级别
        
        Args:
            df: 包含订单簿数据的DataFrame，必须包含timestamp列
            
        Returns:
            聚合后的DataFrame
        """
        # 确保timestamp列存在且为datetime类型
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame必须包含timestamp列")
        
        # 将timestamp转换为datetime类型（如果还不是）
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
        
        # 创建毫秒级别的时间戳
        df['timestamp_ms'] = df['timestamp'].dt.floor(f'{self.time_window_ms}ms')
        
        # 按毫秒时间戳分组聚合
        aggregated_data = self._group_and_aggregate(df)
        
        return aggregated_data
    
    def _group_and_aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按时间窗口分组并聚合数据
        
        Args:
            df: 原始DataFrame
            
        Returns:
            聚合后的DataFrame
        """
        # 定义聚合函数
        agg_functions = {
            'timestamp': 'first',  # 使用窗口内的第一个时间戳
            'local_timestamp': 'first',
            'exchange_timestamp': 'first',
            'sequence_number': 'last',  # 使用最新的序列号
            'symbol': 'first',
            'exchange': 'first',
            'channel': 'first',
            'side': 'first',
            'price': 'mean',  # 价格取平均值
            'amount': 'sum',   # 数量求和
            'id': 'first'      # ID保持不变
        }
        
        # 按毫秒时间戳和side分组
        grouped = df.groupby(['timestamp_ms', 'side'])
        
        # 应用聚合函数
        aggregated = grouped.agg(agg_functions).reset_index()
        
        # 重新整理列顺序
        column_order = [
            'timestamp_ms', 'timestamp', 'local_timestamp', 'exchange_timestamp',
            'sequence_number', 'symbol', 'exchange', 'channel', 'side', 
            'price', 'amount', 'id'
        ]
        
        # 只保留存在的列
        existing_columns = [col for col in column_order if col in aggregated.columns]
        aggregated = aggregated[existing_columns]
        
        return aggregated
    
    def aggregate_with_volume_weighted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用成交量加权平均价格进行聚合
        
        Args:
            df: 原始DataFrame
            
        Returns:
            聚合后的DataFrame
        """
        # 确保timestamp列存在且为datetime类型
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame必须包含timestamp列")
        
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
        
        # 创建毫秒级别的时间戳
        df['timestamp_ms'] = df['timestamp'].dt.floor(f'{self.time_window_ms}ms')
        
        # 计算成交量加权平均价格
        df['price_amount'] = df['price'] * df['amount']
        
        # 按时间窗口和side分组
        grouped = df.groupby(['timestamp_ms', 'side'])
        
        # 计算成交量加权平均价格
        vwap_data = grouped.agg({
            'timestamp': 'first',
            'local_timestamp': 'first',
            'exchange_timestamp': 'first',
            'sequence_number': 'last',
            'symbol': 'first',
            'exchange': 'first',
            'channel': 'first',
            'price_amount': 'sum',
            'amount': 'sum',
            'id': 'first'
        }).reset_index()
        
        # 计算VWAP
        vwap_data['price'] = vwap_data['price_amount'] / vwap_data['amount']
        vwap_data = vwap_data.drop('price_amount', axis=1)
        
        return vwap_data
    
    def aggregate_with_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用OHLC（开高低收）方法进行聚合
        
        Args:
            df: 原始DataFrame
            
        Returns:
            聚合后的DataFrame，包含OHLC价格信息
        """
        # 确保timestamp列存在且为datetime类型
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame必须包含timestamp列")
        
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
        
        # 创建毫秒级别的时间戳
        df['timestamp_ms'] = df['timestamp'].dt.floor(f'{self.time_window_ms}ms')
        
        # 按时间窗口和side分组
        grouped = df.groupby(['timestamp_ms', 'side'])
        
        # 计算OHLC
        ohlc_data = grouped.agg({
            'timestamp': 'first',
            'local_timestamp': 'first',
            'exchange_timestamp': 'first',
            'sequence_number': 'last',
            'symbol': 'first',
            'exchange': 'first',
            'channel': 'first',
            'price': ['open', 'high', 'low', 'close'],
            'amount': 'sum',
            'id': 'first'
        }).reset_index()
        
        # 重命名列
        ohlc_data.columns = [
            'timestamp_ms', 'side', 'timestamp', 'local_timestamp', 'exchange_timestamp',
            'sequence_number', 'symbol', 'exchange', 'channel',
            'price_open', 'price_high', 'price_low', 'price_close', 'amount', 'id'
        ]
        
        return ohlc_data
    
    def get_orderbook_snapshot(self, df: pd.DataFrame, method: str = 'last') -> pd.DataFrame:
        """
        获取每个毫秒时间窗口的订单簿快照
        
        Args:
            df: 原始DataFrame
            method: 聚合方法 ('last', 'vwap', 'ohlc')
            
        Returns:
            订单簿快照DataFrame
        """
        if method == 'last':
            aggregated = self.aggregate_orderbook_data(df)
        elif method == 'vwap':
            aggregated = self.aggregate_with_volume_weighted_price(df)
        elif method == 'ohlc':
            aggregated = self.aggregate_with_ohlc(df)
        else:
            raise ValueError("method必须是 'last', 'vwap', 或 'ohlc'")
        
        # 创建订单簿快照
        snapshot_data = []
        
        for timestamp_ms in aggregated['timestamp_ms'].unique():
            timestamp_data = aggregated[aggregated['timestamp_ms'] == timestamp_ms]
            
            # 分离买卖盘
            bids = timestamp_data[timestamp_data['side'] == 'buy'].sort_values('price', ascending=False)
            asks = timestamp_data[timestamp_data['side'] == 'sell'].sort_values('price', ascending=True)
            
            # 取前5档
            top_bids = bids.head(5)
            top_asks = asks.head(5)
            
            # 创建快照记录
            snapshot_record = {
                'timestamp_ms': timestamp_ms,
                'timestamp': timestamp_data['timestamp'].iloc[0],
                'symbol': timestamp_data['symbol'].iloc[0],
                'exchange': timestamp_data['exchange'].iloc[0]
            }
            
            # 添加买盘数据
            for i, (_, bid) in enumerate(top_bids.iterrows(), 1):
                snapshot_record[f'bid_price_{i}'] = bid['price']
                snapshot_record[f'bid_amount_{i}'] = bid['amount']
            
            # 添加卖盘数据
            for i, (_, ask) in enumerate(top_asks.iterrows(), 1):
                snapshot_record[f'ask_price_{i}'] = ask['price']
                snapshot_record[f'ask_amount_{i}'] = ask['amount']
            
            snapshot_data.append(snapshot_record)
        
        return pd.DataFrame(snapshot_data)


def demo_aggregation():
    """
    演示如何使用聚合器
    """
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    
    # 生成ns级别的时间戳
    base_time = pd.Timestamp('2019-12-01 00:00:00')
    timestamps = [base_time + pd.Timedelta(nanoseconds=np.random.randint(0, 1000000)) 
                  for _ in range(n_samples)]
    
    # 创建示例订单簿数据
    data = {
        'timestamp': timestamps,
        'local_timestamp': timestamps,
        'exchange_timestamp': timestamps,
        'sequence_number': range(n_samples),
        'symbol': ['ETHUSDT'] * n_samples,
        'exchange': ['binance-futures'] * n_samples,
        'channel': ['book_snapshot_5'] * n_samples,
        'side': np.random.choice(['buy', 'sell'], n_samples),
        'price': np.random.uniform(180, 220, n_samples),
        'amount': np.random.uniform(0.1, 10, n_samples),
        'id': range(n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 创建聚合器
    aggregator = OrderBookAggregator(time_window_ms=1)
    
    # 执行聚合
    print("原始数据形状:", df.shape)
    print("原始数据前5行:")
    print(df.head())
    
    # 使用不同方法聚合
    methods = ['last', 'vwap', 'ohlc']
    
    for method in methods:
        print(f"\n使用 {method} 方法聚合:")
        if method == 'ohlc':
            aggregated = aggregator.aggregate_with_ohlc(df)
        elif method == 'vwap':
            aggregated = aggregator.aggregate_with_volume_weighted_price(df)
        else:
            aggregated = aggregator.aggregate_orderbook_data(df)
        
        print(f"聚合后数据形状: {aggregated.shape}")
        print("聚合后数据前5行:")
        print(aggregated.head())
    
    # 获取订单簿快照
    print("\n订单簿快照:")
    snapshot = aggregator.get_orderbook_snapshot(df, method='last')
    print(f"快照数据形状: {snapshot.shape}")
    print("快照数据前5行:")
    print(snapshot.head())


if __name__ == "__main__":
    demo_aggregation()

