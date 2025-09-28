"""
动量与反转特征提取器
包含均值回复强度、自相关性等动量和反转相关特征
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .base_extractor import MicrostructureBaseExtractor
from data.trades_processor import TradesContext


class MomentumFeatureExtractor(MicrostructureBaseExtractor):
    """动量与反转特征提取器"""
    
    def _get_default_config(self) -> Dict[str, bool]:
        """获取默认特征配置"""
        return {
            'enabled': True,
        }
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return [
            'mr_rho1',          # 一阶自相关系数
            'mr_strength',      # 均值回复强度 (负的自相关系数)
        ]
    
    def _extract_features(self, ctx: TradesContext, s: int, e: int,
                         start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Dict[str, float]:
        """提取动量与反转特征"""
        if not self.feature_config.get('enabled', True):
            return {}
        
        # 计算收益率序列的均值回复特征
        r_slice = np.diff(ctx.logp[s:e])
        
        if r_slice.size >= 2:
            # 计算去中心化的收益率
            x0 = r_slice[:-1] - np.mean(r_slice[:-1])
            x1 = r_slice[1:] - np.mean(r_slice[1:])
            
            # 计算一阶自相关系数
            denom = np.sqrt(np.sum(x0**2) * np.sum(x1**2))
            mr_rho1 = float(np.sum(x0 * x1) / denom) if denom > 0 else np.nan
            
            # 均值回复强度（负的自相关系数表示更强的均值回复）
            mr_strength = -mr_rho1 if pd.notna(mr_rho1) else np.nan
        else:
            mr_rho1 = np.nan
            mr_strength = np.nan

        return {
            'mr_rho1': mr_rho1,
            'mr_strength': mr_strength,
        }
