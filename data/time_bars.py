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

        grouped = df.resample(freq, label='right', closed='right').agg(agg)

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
        trade_counts = df['price'].resample(freq, label='right', closed='right').count().rename('trades')
        bars = grouped.join(trade_counts)

        # 丢弃空桶
        bars = bars[bars['trades'] > 0]

        # 使用桶的右端点作为 end_time，左端点作为 start_time
        try:
            offset = pd.tseries.frequencies.to_offset(freq)
        except Exception:
            offset = pd.tseries.frequencies.to_offset(pd.tseries.frequencies.to_offset(freq))
        bars['end_time'] = bars.index
        bars['start_time'] = bars['end_time'] - offset

        # 重置 bar_id，连续整数
        bars = bars.reset_index(drop=True)
        bars['bar_id'] = bars.index

        # 与 DollarBarBuilder 保持一致的技术指标和累计列
        bars = self._add_technical_indicators(bars)

        return bars

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
        cs_cols = [
            'bar_logret', 'bar_abs_logret', 'bar_logret2', 'bar_logret4',
            'bar_tr', 'bar_log_hl', 'bar_parkinson_var', 'bar_gk_var', 'bar_rs_var',
            'bar_duration_s', 'dollar_value', 'volume', 'trades'
        ]
        for col in cs_cols:
            if col in bars.columns:
                bars[f'cs_{col}'] = bars[col].fillna(0.0).cumsum()


