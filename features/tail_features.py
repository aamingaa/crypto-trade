"""
大单特征提取器
包含大单方向性特征、尾部分布特征等
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .base_extractor import MicrostructureBaseExtractor
from data.trades_processor import TradesContext


class TailFeatureExtractor(MicrostructureBaseExtractor):
    """大单特征提取器"""
    
    def _get_default_config(self) -> Dict[str, bool]:
        """获取默认特征配置"""
        return {
            'enabled': True,
            'quantiles': [0.9, 0.95],  # 大单定义的分位数阈值
        }
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        quantiles = self.feature_config.get('quantiles', [0.9, 0.95])
        names = []
        
        for q in quantiles:
            tag = f"q{int(round(q * 100))}"
            names.extend([
                f'large_{tag}_buy_dollar_sum',   # 大单买方金额
                f'large_{tag}_sell_dollar_sum',  # 大单卖方金额
                f'large_{tag}_lti',              # 大单流动性接受不平衡指标
            ])
        
        return names
    
    def _extract_features(self, ctx: TradesContext, s: int, e: int,
                         start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Dict[str, float]:
        """提取大单特征"""
        if not self.feature_config.get('enabled', True):
            return {}
        
        if e - s <= 0:
            return {}
        
        # 获取交易金额和方向
        dv = ctx.quote[s:e]
        if dv.size == 0:
            return {}
        
        sign = ctx.sign[s:e]
        eps = 1e-12  # 防止除零
        
        quantiles = self.feature_config.get('quantiles', [0.9, 0.95])
        features = {}
        
        for q in quantiles:
            # 计算分位数阈值
            thr = float(np.quantile(dv, q)) if np.isfinite(dv).all() else np.nan
            if not np.isfinite(thr) or thr <= 0:
                continue
            
            # 识别大单
            mask = dv >= thr
            if not mask.any():
                continue
            
            # 计算大单买卖金额
            buy_dollar = float(dv[mask][sign[mask] > 0].sum())
            sell_dollar = float(dv[mask][sign[mask] < 0].sum())
            
            # 流动性接受不平衡指标 (Liquidity Taker Imbalance)
            lti = (buy_dollar - sell_dollar) / (buy_dollar + sell_dollar + eps)
            
            tag = f"q{int(round(q * 100))}"
            features.update({
                f'large_{tag}_buy_dollar_sum': buy_dollar,
                f'large_{tag}_sell_dollar_sum': sell_dollar,
                f'large_{tag}_lti': lti,
            })
        
        return features
