"""
基础特征提取器
包含VWAP、交易量、强度、买量占比等基础统计特征
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .base_extractor import MicrostructureBaseExtractor
from data.trades_processor import TradesContext


class BasicFeatureExtractor(MicrostructureBaseExtractor):
    """基础特征提取器"""
    
    def _get_default_config(self) -> Dict[str, bool]:
        """获取默认特征配置"""
        return {
            'enabled': True,
        }
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return [
            'int_trade_vwap',           # 成交量加权平均价
            'int_trade_volume_sum',     # 总成交量
            'int_trade_dollar_sum',     # 总成交金额
            'int_trade_signed_volume',  # 有向成交量
            'int_trade_buy_ratio',      # 买量占比
            'int_trade_intensity',      # 交易强度
            'int_trade_rv',             # 已实现波动率
        ]
    
    def _extract_features(self, ctx: TradesContext, s: int, e: int,
                         start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Dict[str, float]:
        """提取基础特征"""
        if not self.feature_config.get('enabled', True):
            return {}
        
        sum_qty = self._sum_range(ctx.csum_qty, s, e)
        sum_quote = self._sum_range(ctx.csum_quote, s, e)
        sum_signed_qty = self._sum_range(ctx.csum_signed_qty, s, e)
        sum_pxqty = self._sum_range(ctx.csum_pxqty, s, e)

        # VWAP计算
        vwap = sum_pxqty / sum_qty if sum_qty > 0 else np.nan
        
        # 交易强度（单位时间内的交易次数）
        duration = max(1.0, (end_ts - start_ts).total_seconds())
        intensity = (e - s) / duration

        # 买量占比
        buy_mask = ctx.sign[s:e] > 0
        buy_qty = float(ctx.qty[s:e][buy_mask].sum()) if (e - s) > 0 else 0.0
        trade_buy_ratio = (buy_qty / sum_qty) if sum_qty > 0 else np.nan

        # 已实现波动率
        rv = self._sum_range(ctx.csum_ret2, s, e)

        return {
            'int_trade_vwap': vwap,
            'int_trade_volume_sum': sum_qty,
            'int_trade_dollar_sum': sum_quote,
            'int_trade_signed_volume': sum_signed_qty,
            'int_trade_buy_ratio': trade_buy_ratio,
            'int_trade_intensity': intensity,
            'int_trade_rv': rv,
        }
