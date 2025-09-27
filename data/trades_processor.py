"""
交易数据处理模块
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import warnings
from core.base import BaseDataProcessor


class TradesContext:
    """交易数据上下文类，提供高性能的数据访问"""
    
    def __init__(self, t_ns: np.ndarray, price: np.ndarray, qty: np.ndarray, 
                 quote: np.ndarray, sign: np.ndarray):
        self.t_ns = t_ns  # int64 ns 时间戳（已排序）
        self.price = price.astype(np.float64)
        self.qty = qty.astype(np.float64)
        self.quote = quote.astype(np.float64)
        self.sign = sign.astype(np.float64)

        # 衍生量
        self.logp = np.log(self.price)
        self.ret = np.diff(self.logp)
        self.ret2 = np.r_[0.0, self.ret ** 2]
        
        # |r_t||r_{t-1}| 对齐成与 price 同长（首位补0）
        abs_r = np.abs(self.ret)
        self.abs_r = np.r_[0.0, abs_r]
        bp_core = np.r_[0.0, np.r_[0.0, abs_r[1:] * abs_r[:-1]]]

        # 前缀和（与 price 同长）
        self._compute_cumulative_sums()

    def _compute_cumulative_sums(self):
        """计算累计和"""
        self.csum_qty = np.cumsum(self.qty)
        self.csum_quote = np.cumsum(self.quote)
        self.csum_signed_qty = np.cumsum(self.sign * self.qty)
        self.csum_signed_quote = np.cumsum(self.sign * self.quote)
        self.csum_pxqty = np.cumsum(self.price * self.qty)
        self.csum_ret2 = np.cumsum(self.ret2)
        self.csum_abs_r = np.cumsum(self.abs_r)
        
        abs_r = np.abs(self.ret)
        bp_core = np.r_[0.0, np.r_[0.0, abs_r[1:] * abs_r[:-1]]]
        self.csum_bpv = np.cumsum(bp_core)

    def locate(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Tuple[int, int]:
        """定位时间区间在数组中的位置"""
        s = np.searchsorted(self.t_ns, int(np.int64(np.datetime64(start_ts, 'ns'))), side='left')
        e = np.searchsorted(self.t_ns, int(np.int64(np.datetime64(end_ts, 'ns'))), side='right') - 1
        return s, e


class TradesProcessor(BaseDataProcessor):
    """交易数据处理器"""
    
    def __init__(self):
        self.context: Optional[TradesContext] = None
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """验证输入数据格式"""
        required_columns = ['time', 'price', 'qty', 'is_buyer_maker']
        return all(col in data.columns for col in required_columns)
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理交易数据"""
        if not self.validate_input(data):
            raise ValueError("输入数据缺少必要列")
        
        df = data.copy()
        
        # 时间处理
        df['time'] = self._ensure_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        
        # 计算成交额
        if 'quote_qty' not in df.columns or df['quote_qty'].isna().all():
            df['quote_qty'] = df['price'] * df['qty']
        
        # 标记交易方向
        df['trade_sign'] = np.where(df['is_buyer_maker'], -1, 1)
        df['buy_qty'] = df['qty'].where(df['trade_sign'] > 0, 0.0)
        df['sell_qty'] = df['qty'].where(df['trade_sign'] < 0, 0.0)
        
        return df
    
    def build_context(self, trades: pd.DataFrame) -> TradesContext:
        """构建交易上下文"""
        df = self.process(trades)
        
        t = self._ensure_datetime(df['time']).values.astype('datetime64[ns]').astype('int64')
        order = np.argsort(t)
        
        t_ns = t[order]
        price = df['price'].to_numpy(dtype=float)[order]
        qty = df['qty'].to_numpy(dtype=float)[order]
        quote = (df['quote_qty'] if 'quote_qty' in df.columns else df['price'] * df['qty']).to_numpy(dtype=float)[order]
        sign = np.where(df['is_buyer_maker'].to_numpy()[order], -1.0, 1.0)
        
        self.context = TradesContext(t_ns, price, qty, quote, sign)
        return self.context
    
    def _ensure_datetime(self, series: pd.Series) -> pd.Series:
        """时间戳格式转换和标准化"""
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        
        s = pd.to_numeric(series, errors='coerce')
        
        if s.isna().any():
            warnings.warn("序列中包含无法转换为数值的元素，已自动转为NaT")
        
        if s.empty:
            return pd.Series([], dtype='datetime64[ns]')
        
        # 基于时间戳量级判断单位
        ns_threshold = 1e17
        us_threshold = 1e14
        ms_threshold = 1e11
        s_abs = s.abs()
        
        if (s_abs > ns_threshold).any():
            return pd.to_datetime(s, unit='ns', errors='coerce')
        elif (s_abs > us_threshold).any():
            return pd.to_datetime(s, unit='us', errors='coerce')
        elif (s_abs > ms_threshold).any():
            return pd.to_datetime(s, unit='ms', errors='coerce')
        else:
            return pd.to_datetime(s, unit='s', errors='coerce')
