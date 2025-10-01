import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# BTCUSDT SOLUSDT XPLUSDT DOGEUSDT XRPUSDT BNBUSDT PUMPUSDT AVAXUSDT SUIUSDT
# crypto_list = ["BTCUSDT", "BNBUSDT", "PUMPUSDT", "BNBUSDT"]

crypto_list = ["BNBUSDT"]


interval = '15m'
file_path = f'/Volumes/Ext-Disk/data/futures/um/daily/klines'

raw_data_all = {}
feed_data_all = {}
pnl_all = {}
fee_rate = 0.0001

suffix = "2025-01-01_2025-07-01"

# 1-2-3-4-5-6-7-8-9 
# A:1-2-3 10% 
# B:4-5-6 10% 
# C:7-8-9 10% 
# CORR--0 covariance contribution PCA KERNEL FUNCTION 

def generate_date_range(start_date, end_date):    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    return date_list

start_date = "2025-06-01"
end_date = "2025-06-01"

date_list = generate_date_range(start_date, end_date)
crypto_data = {}
for crypto in crypto_list:
    crypto_data[crypto] = {}
    crypto_date_data = []
    for date in date_list:
        print(f"{file_path}/{crypto}/{interval}/{suffix}/{crypto}-{interval}-{date}.zip")
        crypto_date_data.append(pd.read_csv(f"{file_path}/{crypto}/{interval}/{suffix}/{crypto}-{interval}-{date}.zip"))
    
    crypto_data[crypto] = crypto_date_data

print(crypto_data)

# for crypto in crypto_list:
#     # 准备三个币种的数据，分别是BTC, BNB, ETH
#     pnl_all[crypto] = {}
#     raw_data_all[crypto] = pd.DataFrame()
#     raw_data_all[crypto] = pd.read_csv(f"{file_path}/{crypto}/{interval}/{suffix}/{crypto}-{interval}-{}.zip" )
#     raw_data_all[crypto].index = pd.to_datetime(raw_data_all[crypto].index)
#     raw_data_all[crypto] = raw_data_all[crypto][['open', 'high', 'low', 'close', 'volume']]
#     raw_data_all[crypto] = raw_data_all[crypto][raw_data_all[crypto].index >= pd.to_datetime('2017-11-08')]
#     raw_data_all[crypto]['ret_curr'] = (raw_data_all[crypto]['close'].shift(-1) / raw_data_all[crypto]['close'] - 1).fillna(0)
#     final_timestamp = raw_data_all[crypto].index[1000:]  
#     feed_data_all[crypto] = pd.DataFrame(index=final_timestamp)
    
    
#     position = np.zeros_like(fct_value)
#     thd_chosen = 1
#     position[fct_value >= thd_chosen] = 1
#     position[fct_value <= -thd_chosen] = -1
#     pnl_gross_brk_thd = position * ret_curr
#     position_changes_brk_thd = np.abs(np.diff(position, prepend=0))
#     position_changes_brk_thd = np.where(position_changes_brk_thd < 1e-10, 0, position_changes_brk_thd)
#     transaction_cost_brk_thd = position_changes_brk_thd * fee_rate
#     pnl_net_brk_thd = pnl_gross_brk_thd - transaction_cost_brk_thd
#     cumulative_return_brk_thd = 1 + np.cumsum(pnl_net_brk_thd)
#     cumulative_cost_brk_thd = np.cumsum(transaction_cost_brk_thd)
    
#     pnl_all[crypto][fct] = pnl_net_onbar