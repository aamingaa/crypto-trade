from binance.client import Client
from datetime import datetime
import pandas as pd
import time

def get_previous_15min_kline(symbol='BTCUSDT'):
    # 初始化Binance客户端
    # 注意：这里使用空字符串作为API key和secret，因为只是获取公开数据
    client = Client("", "")
    
    # 获取上一个15分钟K线数据
    client.get_recent_trades()
    klines = client.get_klines(
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_15MINUTE,
        limit=1  # 只获取最近一根K线
    )
    
    # 将数据转换为DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # 转换数据类型
    # 将UTC时间戳转换为本地时间
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    return df

def OnBar(kline_data):
    # 在这里撰写策略的主逻辑
    # 数据可以实时保存，也可以丢掉
    # 实盘转储——高频数据
    # 计算因子和仓位
    # 输出仓位
    # 调用my trading action做下单
    pass

# 增加一个回调
# websocket回调函数会有一些复杂

if __name__ == "__main__":
    print("开始监控BTC/USDT的15分钟K线数据...")
    while True:
        try:
            # 获取BTC/USDT的15分钟K线数据
            kline_data = get_previous_15min_kline()
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{current_time}] 上一个15分钟K线数据:")
            print(kline_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']])
            OnBar(kline_data)
            # 等待15分钟
            time.sleep(15 * 60)
        except Exception as e:
            print(f"发生错误: {e}")
            # 如果发生错误，等待1分钟后重试
            time.sleep(60)
