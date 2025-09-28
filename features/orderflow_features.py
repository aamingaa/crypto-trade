"""
订单流特征提取器
包含OFI、GOF、签名不平衡等订单流相关特征
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .base_extractor import MicrostructureBaseExtractor
from data.trades_processor import TradesContext


class OrderFlowFeatureExtractor(MicrostructureBaseExtractor):
    """订单流特征提取器"""
    
    def _get_default_config(self) -> Dict[str, bool]:
        """获取默认特征配置"""
        return {
            'enabled': True,
        }
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return [
            'ofi_signed_qty_sum',      # 有向成交量和（OFI相关）
            'ofi_signed_quote_sum',    # 有向成交金额和（OFI相关）
            'gof_by_count',            # 按交易次数的Garman订单流
            'gof_by_volume',           # 按成交量的Garman订单流
        ]
    
    def _extract_features(self, ctx: TradesContext, s: int, e: int,
                         start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Dict[str, float]:
        """提取订单流特征"""
        if not self.feature_config.get('enabled', True):
            return {}
        
        sum_qty = self._sum_range(ctx.csum_qty, s, e)
        sum_signed_qty = self._sum_range(ctx.csum_signed_qty, s, e)
        sum_signed_quote = self._sum_range(ctx.csum_signed_quote, s, e)

        # Garman Order Flow (GOF) 特征
        n = float(e - s)
        
        # 按交易次数的GOF：平均签名方向
        gof_by_count = float(np.mean(np.sign(ctx.sign[s:e]))) if n > 0 else np.nan
        
        # 按成交量的GOF：有向成交量占比
        gof_by_volume = (sum_signed_qty / sum_qty) if sum_qty > 0 else np.nan

        return {
            'ofi_signed_qty_sum': sum_signed_qty,
            'ofi_signed_quote_sum': sum_signed_quote,
            'gof_by_count': gof_by_count,
            'gof_by_volume': gof_by_volume,
        }
