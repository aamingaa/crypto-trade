"""
路径形状特征提取器
包含VWAP偏离、累计流量与价格相关性等路径形状相关特征
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .base_extractor import MicrostructureBaseExtractor
from data.trades_processor import TradesContext


class PathShapeFeatureExtractor(MicrostructureBaseExtractor):
    """路径形状特征提取器"""
    
    def _get_default_config(self) -> Dict[str, bool]:
        """获取默认特征配置"""
        return {
            'enabled': True,
            'min_trades_for_correlation': 3,  # 计算相关性的最小交易数
        }
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return [
            'signed_vwap_deviation',              # 有向VWAP偏离
            'vwap_deviation',                     # VWAP偏离
            'corr_cumsum_signed_qty_logp',       # 累计有向成交量与对数价格相关性
            'corr_cumsum_signed_dollar_logp',    # 累计有向成交金额与对数价格相关性
        ]
    
    def _extract_features(self, ctx: TradesContext, s: int, e: int,
                         start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Dict[str, float]:
        """提取路径形状特征"""
        if not self.feature_config.get('enabled', True):
            return {}
        
        # VWAP偏离特征
        vwap_features = self._extract_vwap_deviation_features(ctx, s, e)
        
        # 累计流量与价格相关性特征
        correlation_features = self._extract_correlation_features(ctx, s, e)
        
        return {
            **vwap_features,
            **correlation_features,
        }
    
    def _extract_vwap_deviation_features(self, ctx: TradesContext, s: int, e: int) -> Dict[str, float]:
        """提取VWAP偏离特征"""
        sum_qty = self._sum_range(ctx.csum_qty, s, e)
        sum_signed_qty = self._sum_range(ctx.csum_signed_qty, s, e)
        sum_pxqty = self._sum_range(ctx.csum_pxqty, s, e)

        # 计算VWAP
        vwap = sum_pxqty / sum_qty if sum_qty > 0 else np.nan
        
        if e > s:
            p_last = ctx.price[e - 1]
        else:
            p_last = np.nan

        # VWAP偏离
        if vwap != 0 and np.isfinite(vwap) and np.isfinite(p_last):
            dev = (p_last - vwap) / vwap
            
            # 有向VWAP偏离（根据净流量方向调整符号）
            if sum_signed_qty > 0:
                signed_dev = dev
            elif sum_signed_qty < 0:
                signed_dev = -dev
            else:
                signed_dev = 0.0
        else:
            dev = np.nan
            signed_dev = np.nan

        return {
            'signed_vwap_deviation': signed_dev,
            'vwap_deviation': dev,
        }
    
    def _extract_correlation_features(self, ctx: TradesContext, s: int, e: int) -> Dict[str, float]:
        """提取累计流量与价格相关性特征"""
        min_trades = self.feature_config.get('min_trades_for_correlation', 3)
        
        if e - s <= min_trades:
            return {
                'corr_cumsum_signed_qty_logp': np.nan,
                'corr_cumsum_signed_dollar_logp': np.nan,
            }
        
        # 相对于起始点的对数价格变化
        logp_rel = ctx.logp[s:e] - ctx.logp[s]
        
        # 累计有向成交量（相对于起始点）
        cs_signed_qty = (ctx.csum_signed_qty[s:e] - 
                        (ctx.csum_signed_qty[s] if s < ctx.csum_signed_qty.size else 0.0))
        
        # 累计有向成交金额（相对于起始点）
        cs_signed_dollar = (ctx.csum_signed_quote[s:e] - 
                           (ctx.csum_signed_quote[s] if s < ctx.csum_signed_quote.size else 0.0))
        
        # 计算相关性
        corr_qty = self._compute_correlation(cs_signed_qty, logp_rel)
        corr_dollar = self._compute_correlation(cs_signed_dollar, logp_rel)

        return {
            'corr_cumsum_signed_qty_logp': corr_qty,
            'corr_cumsum_signed_dollar_logp': corr_dollar,
        }
