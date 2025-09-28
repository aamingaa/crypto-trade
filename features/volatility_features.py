"""
波动率特征提取器
包含RV、BPV、Jump、高低幅度比等波动率相关特征
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .base_extractor import MicrostructureBaseExtractor
from data.trades_processor import TradesContext


class VolatilityFeatureExtractor(MicrostructureBaseExtractor):
    """波动率特征提取器"""
    
    def _get_default_config(self) -> Dict[str, bool]:
        """获取默认特征配置"""
        return {
            'enabled': True,
            'micro_momentum_window': 20,  # 微动量计算窗口
        }
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return [
            'rv',                        # 已实现波动率
            'bpv',                       # 双幂变差
            'jump_rv_bpv',              # 跳跃组件 (RV - BPV)
            'micro_dp_short',           # 微动量（短期价格变化）
            'micro_dp_zscore',          # 微动量Z-score
            'hl_amplitude_ratio',       # 高低幅度比
        ]
    
    def _extract_features(self, ctx: TradesContext, s: int, e: int,
                         start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Dict[str, float]:
        """提取波动率特征"""
        if not self.feature_config.get('enabled', True):
            return {}
        
        # RV (已实现波动率) / BPV (双幂变差) / Jump (跳跃组件)
        rv = self._sum_range(ctx.csum_ret2, s, e)
        bpv = self._sum_range(ctx.csum_bpv, s, e)
        jump = max(rv - bpv, 0.0) if (np.isfinite(rv) and np.isfinite(bpv)) else np.nan

        # 微动量特征
        micro_features = self._extract_micro_momentum_features(ctx, s, e)
        
        # 高低幅度比
        hl_amplitude_ratio = self._extract_high_low_amplitude_ratio(ctx, s, e)

        return {
            'rv': rv,
            'bpv': bpv,
            'jump_rv_bpv': jump,
            'micro_dp_short': micro_features['micro_dp_short'],
            'micro_dp_zscore': micro_features['micro_dp_zscore'],
            'hl_amplitude_ratio': hl_amplitude_ratio,
        }
    
    def _extract_micro_momentum_features(self, ctx: TradesContext, s: int, e: int) -> Dict[str, float]:
        """提取微动量特征"""
        W = min(self.feature_config.get('micro_momentum_window', 20), e - s)
        
        if W >= 2:
            # 取最后W个交易的对数价格
            lp = ctx.logp[max(s, e - W):e]
            dp_short = float(lp[-1] - lp[0])  # 短期价格变化
            
            # 计算Z-score
            mu = float(np.mean(lp))
            sd = float(np.std(lp))
            z = (float(lp[-1]) - mu) / sd if sd > 0 else np.nan
        else:
            dp_short = np.nan
            z = np.nan
        
        return {
            'micro_dp_short': dp_short,
            'micro_dp_zscore': z,
        }
    
    def _extract_high_low_amplitude_ratio(self, ctx: TradesContext, s: int, e: int) -> float:
        """提取高低幅度比特征"""
        if (e - s) > 0:
            hi = float(np.max(ctx.price[s:e]))
            lo = float(np.min(ctx.price[s:e]))
            mid = (hi + lo) / 2.0
            return float((hi - lo) / mid) if mid != 0 else np.nan
        else:
            return np.nan
