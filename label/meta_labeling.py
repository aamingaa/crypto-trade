from util.sampling import dollar_bars
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
import getTA
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# 计算项目根目录（dir_b的上一级目录）
project_root = os.path.abspath(os.path.join(current_dir, ".."))
# 将项目根目录添加到Python搜索路径
sys.path.append(project_root)



import triple_barrier as tb #local

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
import getTA #local
from util import tautil #local




def getDailyVol(close, span0=100):
    # daily vol, reindexed to cloes
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0>0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1 # daily returns
    df0 = df0.ewm(span=span0).std()
    return df0


# df = pd.read_csv('')



# # volumn_resample_bar = dollar_bars(es_contracts, 100000000)

# # daily_vol = getDailyVol(volumn_resample_bar['Close'])



# daily_vol.plot()

