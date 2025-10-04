"""
Time Bar 生成模块
"""
import numpy as np
import pandas as pd
from typing import Optional
from core.base import BaseDataProcessor
from data.trades_processor import TradesProcessor


class TimeBarBuilder(BaseDataProcessor):
    """Time Bar 构建器：按照时间频率重采样逐笔成交数据

    输出字段尽量与 `DollarBarBuilder` 对齐，以便后续特征提取与管道复用。
    """

    def __init__(self, freq: str = '1H'):
        """
        参数
        - freq: pandas 频率字符串，如 '1H', '15min', '5T', '1min' 等。
        """
        self.freq = freq
        self.trades_processor = TradesProcessor()

    def validate_input(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        return self.trades_processor.validate_input(data)

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成 Time Bars"""
        return self.build_time_bars(data, self.freq)

    def build_time_bars(self, trades: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        生成 time bars，bar 边界按给定时间频率对齐
        """
        df = self.trades_processor.process(trades)

        # 记录原始行号以便输出 start/end_trade_idx
        df = df.reset_index(drop=True)
        df['original_index'] = df.index

        # 设置时间索引并按频率分箱（按 end_time 聚合：右闭右标记）
        df = df.set_index(pd.to_datetime(df['time']))
        df = df.sort_index()

        agg = {
            'price': ['first', 'max', 'min', 'last'],
            'qty': 'sum',
            'quote_qty': 'sum',
            'buy_qty': 'sum',
            'sell_qty': 'sum',
            'original_index': ['first', 'last']
        }

        grouped = df.resample(freq, label='right', closed='left').agg(agg)

        if len(grouped) == 0:
            return pd.DataFrame(columns=[
                'bar_id','start_time','end_time','open','high','low','close',
                'volume','dollar_value','buy_volume','sell_volume','trades',
                'start_trade_idx','end_trade_idx'
            ])

        # 展平列名
        grouped.columns = [
            'open', 'high', 'low', 'close',
            'volume', 'dollar_value',
            'buy_volume', 'sell_volume',
            'start_trade_idx', 'end_trade_idx'
        ]

        # 交易笔数（每个时间桶内的逐笔条数）
        trade_counts = df['price'].resample(freq, label='right', closed='left').count().rename('trades')
        bars = grouped.join(trade_counts)

        # 丢弃空桶
        bars = bars[bars['trades'] > 0]

        # 使用桶的右端点作为 end_time，左端点作为 start_time
        offset = pd.tseries.frequencies.to_offset(freq)
        bars['end_time'] = bars.index
        bars['start_time'] = bars['end_time'] - offset

        # 添加逐笔交易级别的累计前缀和数据
        self._add_trade_level_cumulative_data(bars, df)
        
        # 重置 bar_id 为连续整数
        bars = bars.reset_index(drop=True)
        bars['bar_id'] = bars.index
        
        # 添加大单统计
        bars = self._add_large_order_stats(df, bars)

        # 与 DollarBarBuilder 保持一致的技术指标和累计列
        bars = self._add_technical_indicators(bars)

        return bars
    
    def _add_trade_level_cumulative_data(self, bars: pd.DataFrame, df: pd.DataFrame):
        """添加逐笔交易级别的累计前缀和数据"""
        # 提取数组用于向量化计算
        prices = df['price'].to_numpy(dtype=float)
        qtys = df['qty'].to_numpy(dtype=float) 
        quotes = df['quote_qty'].to_numpy(dtype=float)
        signs = np.where(df['is_buyer_maker'].to_numpy(), -1.0, 1.0)
        
        # 计算累计前缀和
        csum_qty = np.cumsum(qtys)
        csum_signed_qty = np.cumsum(signs * qtys)
        csum_quote = np.cumsum(quotes)
        csum_signed_quote = np.cumsum(signs * quotes)
        csum_pxqty = np.cumsum(prices * qtys)
        
        # 波动相关的累计数据
        logp = np.log(prices)
        r = np.diff(logp)
        ret2 = np.r_[0.0, r * r]
        abs_r = np.r_[0.0, np.abs(r)]
        # Bipower variation: |r_t||r_{t-1}|
        bp_core = np.r_[0.0, np.r_[0.0, abs_r[1:] * abs_r[:-1]]]
        
        csum_ret2 = np.cumsum(ret2)
        csum_abs_r = np.cumsum(abs_r)
        csum_bpv = np.cumsum(bp_core)
        
        # 将累计数据映射到每个bar的结束位置
        end_idx = bars['end_trade_idx'].to_numpy(dtype=int)
        bars['cs_qty'] = csum_qty[end_idx]
        bars['cs_quote'] = csum_quote[end_idx]
        bars['cs_signed_qty'] = csum_signed_qty[end_idx]
        bars['cs_signed_quote'] = csum_signed_quote[end_idx]
        bars['cs_pxqty'] = csum_pxqty[end_idx]
        bars['cs_ret2'] = csum_ret2[end_idx]
        bars['cs_abs_r'] = csum_abs_r[end_idx]
        bars['cs_bpv'] = csum_bpv[end_idx]
    
    def _add_large_order_stats(self, df: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
        """添加大单统计"""
        # 首先需要为 df 添加 bar_id 以便分组
        # 使用 bars 的时间边界为每条交易分配 bar_id
        df_with_bar = self._assign_bar_ids_to_trades(df, bars)
        
        abs_thresholds = [100.0, 1000.0, 10000.0, 100000.0]
        bar_ids_arr = df_with_bar['bar_id'].to_numpy(dtype=np.int64)
        quote_arr = df_with_bar['quote_qty'].to_numpy(dtype=np.float64)
        sign_arr = np.where(df_with_bar['trade_sign'].to_numpy() > 0, 1.0, -1.0)
        n_bars = int(bars.index.max()) + 1 if len(bars.index) > 0 else 0
        
        for thr in abs_thresholds:
            m = quote_arr >= thr
            if n_bars == 0 or not np.any(m):
                tag = f"abs_{int(thr)}"
                self._add_zero_large_order_columns(bars, tag)
                continue

            # 计算大单统计
            self._compute_large_order_stats(bars, bar_ids_arr, quote_arr, sign_arr, m, thr, n_bars)
        
        return bars
    
    def _assign_bar_ids_to_trades(self, df: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
        """为每条交易分配对应的 bar_id"""
        df_copy = df.copy()
        df_copy['bar_id'] = -1
        
        for idx, bar_row in bars.iterrows():
            start_idx = int(bar_row['start_trade_idx'])
            end_idx = int(bar_row['end_trade_idx'])
            df_copy.loc[start_idx:end_idx, 'bar_id'] = idx
        
        return df_copy
    
    def _add_zero_large_order_columns(self, bars: pd.DataFrame, tag: str):
        """添加零值大单列"""
        columns = ['dollar_sum', 'count', 'buy_dollar', 'sell_dollar', 'buy_count', 'sell_count']
        for col in columns:
            bars[f'large_{tag}_{col}'] = 0.0
    
    def _compute_large_order_stats(self, bars: pd.DataFrame, bar_ids_arr: np.ndarray, 
                                 quote_arr: np.ndarray, sign_arr: np.ndarray, 
                                 mask: np.ndarray, threshold: float, n_bars: int):
        """计算大单统计指标"""
        dollar_sum = np.bincount(bar_ids_arr[mask], weights=quote_arr[mask], minlength=n_bars)
        count_sum = np.bincount(bar_ids_arr[mask], minlength=n_bars)
        
        mb = mask & (sign_arr > 0)
        ms = mask & (sign_arr < 0)
        buy_dollar = np.bincount(bar_ids_arr[mb], weights=quote_arr[mb], minlength=n_bars)
        sell_dollar = np.bincount(bar_ids_arr[ms], weights=quote_arr[ms], minlength=n_bars)
        buy_count = np.bincount(bar_ids_arr[mb], minlength=n_bars)
        sell_count = np.bincount(bar_ids_arr[ms], minlength=n_bars)

        tag = f"abs_{int(threshold)}"
        bars[f'large_{tag}_dollar_sum'] = dollar_sum.astype(float)
        bars[f'large_{tag}_count'] = count_sum.astype(float)
        bars[f'large_{tag}_buy_dollar'] = buy_dollar.astype(float)
        bars[f'large_{tag}_sell_dollar'] = sell_dollar.astype(float)
        bars[f'large_{tag}_buy_count'] = buy_count.astype(float)
        bars[f'large_{tag}_sell_count'] = sell_count.astype(float)

    def _add_technical_indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        """复用与 DollarBarBuilder 基本一致的技术指标计算"""
        eps = 1e-12
        close = bars['close'].astype(float)
        open_ = bars['open'].astype(float)
        high = bars['high'].astype(float)
        low = bars['low'].astype(float)
        prev_close = close.shift(1)

        # 基础对数收益
        r = np.log((close + eps) / (prev_close + eps))
        bars['bar_logret'] = r
        bars['bar_abs_logret'] = r.abs()
        bars['bar_logret2'] = r * r
        bars['bar_logret4'] = (r * r) * (r * r)

        # 高低对数幅度
        log_hl = np.log((high + eps) / (low + eps))
        bars['bar_log_hl'] = log_hl

        # 波动率估计
        bars['bar_parkinson_var'] = (log_hl ** 2) / (4.0 * np.log(2.0))
        log_co = np.log((close + eps) / (open_ + eps))
        bars['bar_gk_var'] = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
        bars['bar_rs_var'] = (
            np.log((high + eps) / (close + eps)) * np.log((high + eps) / (open_ + eps)) +
            np.log((low + eps) / (close + eps)) * np.log((low + eps) / (open_ + eps))
        )

        # True Range
        tr_candidates = np.vstack([
            (high - low).to_numpy(dtype=float),
            (high - prev_close).abs().to_numpy(dtype=float),
            (low - prev_close).abs().to_numpy(dtype=float),
        ])
        bars['bar_tr'] = np.nanmax(tr_candidates, axis=0)
        bars['bar_tr_norm'] = bars['bar_tr'] / (close + eps)

        # 活跃度与金额速度（注意 time bar 下 duration 更稳定）
        st = pd.to_datetime(bars['start_time'])
        et = pd.to_datetime(bars['end_time'])
        duration_s = (et - st).dt.total_seconds().clip(lower=1.0)
        bars['bar_duration_s'] = duration_s
        bars['bar_intensity_trade_per_s'] = bars['trades'] / duration_s
        bars['bar_dollar_per_s'] = bars['dollar_value'] / duration_s

        # Amihud 流动性指标
        bars['bar_amihud'] = bars['bar_abs_logret'] / (bars['dollar_value'].replace(0, np.nan))
        bars['bar_amihud'] = bars['bar_amihud'].fillna(0.0)

        # 累计列
        self._add_cumulative_columns(bars)

        return bars

    def _add_cumulative_columns(self, bars: pd.DataFrame):
        """添加累计列"""
        cs_cols = [
            'bar_logret', 'bar_abs_logret', 'bar_logret2', 'bar_logret4',
            'bar_tr', 'bar_log_hl', 'bar_parkinson_var', 'bar_gk_var', 'bar_rs_var',
            'bar_duration_s', 'dollar_value', 'volume', 'trades'
        ]
        
        for col in cs_cols:
            if col in bars.columns:
                bars[f'cs_{col}'] = bars[col].fillna(0.0).cumsum()
        
        # 为大单列添加累计前缀和
        abs_thresholds = [100.0, 1000.0, 10000.0, 100000.0]
        for thr in abs_thresholds:
            tag = f"abs_{int(thr)}"
            for col in ['dollar_sum', 'count', 'buy_dollar', 'sell_dollar', 'buy_count', 'sell_count']:
                colname = f'large_{tag}_{col}'
                if colname in bars.columns:
                    bars[f'cs_{colname}'] = bars[colname].cumsum()


