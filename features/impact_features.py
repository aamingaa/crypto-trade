"""
价格冲击特征提取器
包含Kyle Lambda、Amihud Lambda、Hasbrouck Lambda、半衰期等价格冲击相关特征
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .base_extractor import MicrostructureBaseExtractor
from data.trades_processor import TradesContext


class PriceImpactFeatureExtractor(MicrostructureBaseExtractor):
    """价格冲击特征提取器"""
    
    def _get_default_config(self) -> Dict[str, bool]:
        """获取默认特征配置"""
        return {
            'enabled': True,
            'min_trades': 3,  # 最小交易数量要求
        }
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return [
            'kyle_lambda',              # Kyle Lambda (价格冲击系数)
            'amihud_lambda',            # Amihud Lambda (非流动性指标)
            'hasbrouck_lambda',         # Hasbrouck Lambda (信息冲击系数)
            'impact_half_life',         # 价格冲击半衰期
            'impact_perm_share',        # 永久冲击占比
            'impact_transient_share',   # 暂时冲击占比
        ]
    
    def _extract_features(self, ctx: TradesContext, s: int, e: int,
                         start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Dict[str, float]:
        """提取价格冲击特征"""
        if not self.feature_config.get('enabled', True):
            return {}
        
        min_trades = self.feature_config.get('min_trades', 3)
        if e - s < min_trades:
            return {
                'kyle_lambda': np.nan,
                'amihud_lambda': np.nan, 
                'hasbrouck_lambda': np.nan,
                'impact_half_life': np.nan,
                'impact_perm_share': np.nan,
                'impact_transient_share': np.nan
            }
        
        # 计算收益率和有向金额
        r = np.diff(ctx.logp[s:e])
        sdollar = ctx.sign[s:e] * ctx.quote[s:e]
        x = sdollar[1:]  # 对齐收益率
        
        # Kyle Lambda (价格冲击系数)
        kyle_lambda = self._calculate_kyle_lambda(x, r)
        
        # Amihud Lambda (非流动性指标)
        amihud_lambda = self._calculate_amihud_lambda(r, ctx.quote[s+1:e])
        
        # Hasbrouck Lambda (信息冲击系数)
        hasbrouck_lambda = self._calculate_hasbrouck_lambda(r, ctx.quote[s+1:e])
        
        # 价格冲击半衰期
        impact_half_life = self._calculate_impact_half_life(r)
        
        # 永久vs暂时冲击占比
        perm_share, trans_share = self._calculate_impact_shares(ctx.price[s:e])

        return {
            'kyle_lambda': kyle_lambda,
            'amihud_lambda': amihud_lambda,
            'hasbrouck_lambda': hasbrouck_lambda,
            'impact_half_life': impact_half_life,
            'impact_perm_share': perm_share,
            'impact_transient_share': trans_share,
        }
    
    def _calculate_kyle_lambda(self, x: np.ndarray, r: np.ndarray) -> float:
        """计算Kyle Lambda"""
        varx = float(np.var(x))
        if varx > 0:
            return float(np.cov(x, r, ddof=1)[0, 1] / varx)
        else:
            return np.nan
    
    def _calculate_amihud_lambda(self, r: np.ndarray, quotes: np.ndarray) -> float:
        """计算Amihud Lambda"""
        if np.all(quotes > 0):
            return float((np.abs(r) / quotes).mean())
        else:
            return np.nan
    
    def _calculate_hasbrouck_lambda(self, r: np.ndarray, quotes: np.ndarray) -> float:
        """计算Hasbrouck Lambda"""
        xh = np.sign(r) * np.sqrt(quotes)
        varxh = float(np.var(xh))
        if varxh > 0 and len(r) > 1:
            return float(np.cov(xh, r, ddof=1)[0, 1] / varxh)
        else:
            return np.nan
    
    def _calculate_impact_half_life(self, r: np.ndarray) -> float:
        """计算价格冲击半衰期"""
        if len(r) < 2:
            return np.nan
        
        # 计算一阶自相关系数
        r0 = r[:-1] - np.mean(r[:-1])
        r1 = r[1:] - np.mean(r[1:])
        denom = np.sqrt(np.sum(r0**2) * np.sum(r1**2))
        
        if denom > 0:
            rho = float(np.sum(r0 * r1) / denom)
            if 0 < rho < 1:
                return float(np.log(2.0) / (-np.log(rho)))
            else:
                return np.nan
        else:
            return np.nan
    
    def _calculate_impact_shares(self, prices: np.ndarray) -> tuple:
        """计算永久冲击和暂时冲击占比"""
        if len(prices) < 2:
            return np.nan, np.nan
        
        dp = np.diff(prices)
        denom = float(np.sum(np.abs(dp)))
        
        if denom > 0:
            # 永久冲击占比：净价格变化 / 总价格变化
            perm = float(np.abs(prices[-1] - prices[0]) / denom)
            perm = float(np.clip(perm, 0.0, 1.0))
            trans = float(1.0 - perm)
            return perm, trans
        else:
            return np.nan, np.nan
