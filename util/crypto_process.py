"""
resample crypto数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import talib as ta
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os


interval = '15m'
file_path = f'/Volumes/Ext-Disk/data/futures/um/daily/klines'

raw_data_all = {}
feed_data_all = {}
pnl_all = {}
fee_rate = 0.0001
suffix = "2025-01-01_2025-07-01"

def load_data(start_month:str, end_month:str) -> pd.DataFrame:
    start_month = datetime.strptime(start_month, '%Y-%m')
    end_month = datetime.strptime(end_month, '%Y-%m')

    zs = []
    while start_month <= end_month:
        start_month = start_month.strftime('%Y-%m')
        file_path = f"/Users/aming/project/python/课程代码合集/高频因子和高频策略/Data/BTCUSDT-1m-{start_month}.csv"
        z = pd.read_csv(file_path)
        zs.append(z)
        start_month = datetime.strptime(start_month, '%Y-%m')
        if start_month.month == 12:
            start_month = start_month.replace(year=start_month.year + 1, month=1)
        else:
            start_month = start_month.replace(month=start_month.month + 1)
    
    z = pd.concat(zs, axis=0, ignore_index=True)
    # 处理时间戳,变成年月日-时分秒格式
    z['open_time'] = pd.to_datetime(z['open_time'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
    z['close_time'] = pd.to_datetime(z['close_time'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')

    return z


def load_daily_data(start_date:str, end_date:str, interval:str, crypto:str = "BNBUSDT") -> pd.DataFrame:
    date_list = generate_date_range(start_date, end_date)
    crypto_date_data = []
    for date in date_list:
        crypto_date_data.append(pd.read_csv(f"{file_path}/{crypto}/{interval}/{suffix}/{crypto}-{interval}-{date}.zip"))
    
    z = pd.concat(crypto_date_data, axis=0, ignore_index=True)
    # 处理时间戳,变成年月日-时分秒格式
    z['open_time'] = pd.to_datetime(z['open_time'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
    z['close_time'] = pd.to_datetime(z['close_time'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
    return z

def generate_date_range(start_date, end_date):    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    return date_list



def resample_data(z:pd.DataFrame,freq:str) -> pd.DataFrame:
    z_ = z.copy()
    z_.index = pd.to_datetime(z_.open_time)
    z_rspled = z_.resample(freq).agg({'open_time':'first', 
                              'open':'first', 
                              'high':'max', 
                              'low':'min', 
                              'close':'last', 
                              'volume':'sum', 
                              'close_time':'last',
                              'quote_volume':'sum', 
                              'count':'sum', 
                              'taker_buy_volume':'sum', 
                              'taker_buy_quote_volume':'sum',
                              'ignore':'last'})
    z = z_rspled[['open','high','low','close','volume','quote_volume','count','taker_buy_volume','taker_buy_quote_volume']]
    return z


if __name__ == '__main__':
    # start_month = '2024-10'
    # end_month = '2024-10'
    # freq = '5min'
    # z = load_data(start_month,end_month)
    # z = resample_data(z,freq)
    start_date = "2025-06-01"
    end_date = "2025-06-07"
    daily_data = load_daily_data(start_date, end_date, "15m")
    print(daily_data)