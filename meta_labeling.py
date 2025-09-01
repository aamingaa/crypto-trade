from util.sampling import dollar_bars, dollar_bars_v2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

import warnings 
warnings.filterwarnings('ignore')


from sklearn.preprocessing import MinMaxScaler, StandardScaler

from label import triple_barrier as tb
import ta
from util import getTA
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# 计算项目根目录（dir_b的上一级目录）
project_root = os.path.abspath(os.path.join(current_dir, ".."))
# 将项目根目录添加到Python搜索路径
sys.path.append(project_root)


# lib
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from scipy.stats import norm, moment

#feature
from sklearn import preprocessing
from sklearn.decomposition import PCA 
#ML
# import autogluon as ag


# deep learning
import keras

# Technical analysis
import ta
from util import getTA #local
from util import tautil #local


def getDailyVol(close, span0=100):
    # daily vol, reindexed to cloes
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0>0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1 # daily returns
    df0 = df0.ewm(span=span0).std()
    return df0


df = pd.read_csv('/Volumes/Ext-Disk/data/futures/um/daily/trades/ETHUSDT/ETHUSDT-trades-2025-06-24.zip')
# df = pd.read_csv('/Volumes/Ext-Disk/data/futures/um/monthly/trades/ETHUSDT/ETHUSDT-trades-2025-05.zip')

df['time'] = pd.to_datetime(df['time'], unit= 'ms')
# print(df.head(100))

group_dollar_bar_df = dollar_bars_v2(df, bar_size=10000 * 9000)

print(group_dollar_bar_df.head(100))


# # 绘制两条线
# plt.figure(figsize=(12,6))
# # plt.plot(df.index, df.close, label='Original Close', alpha=0.7)
# plt.plot(group_dollar_bar_v2.index, group_dollar_bar_v2.close, label='Dollar Bar Close', alpha=0.7)
# # plt.plot(group_dollar_bar_v2.index, group_dollar_bar_v2.taker_qty, label='Dollar buy quote qty', alpha=0.7)
# # plt.plot(group_dollar_bar_v2.index, group_dollar_bar_v2.maker_qty, label='Dollar sell quote qty', alpha=0.7)

# # plt.plot(group_dollar_bar_v2.index, group_dollar_bar_v2.num_tickers, label='Dollar Bar Close', alpha=0.7)


# plt.title('Dollar Bar Close Prices')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.show() # 确保图形显示



windows = np.arange(10,200,10)
rsi_df = pd.DataFrame(index=df.index)
rsi_df = tautil.get_rsi(group_dollar_bar_df.close, windows)
rsi_df.dropna(inplace=True)

for i in rsi_df.columns:
    sc = rsi_df[i].copy()
    plt.figure(figsize=(9,1))
    plt.plot(df.close.loc[sc.index], linewidth=0.5,alpha=0.6)
    plt.scatter(df.close.loc[sc.index].index, df.close.loc[sc.index], c=sc,cmap='gray_r', alpha=1, vmin=0,vmax=1)
    plt.colorbar()
    plt.title('{}'.format(i))
    plt.show()




# for w in windows:
#     rsi_df['rsi_{}'.format(w)] = tautil.my_rsi_2(df.close, w)
# rsi_df.dropna(inplace=True)





