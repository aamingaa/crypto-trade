"""
特征提取器基类
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from core.base import BaseFeatureExtractor
from data.trades_processor import TradesContext


class MicrostructureBaseExtractor(BaseFeatureExtractor):
    """微观结构特征提取器基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.feature_config = self._get_default_config()
        if config and isinstance(config, dict):
            self.feature_config.update(config)
    
    def _get_default_config(self) -> Dict[str, bool]:
        """获取默认特征配置"""
        return {}
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        raise NotImplementedError("子类必须实现此方法")
    
    def extract(self, data: pd.DataFrame) -> Dict[str, float]:
        """从数据中提取特征（这里需要TradesContext）"""
        raise NotImplementedError("请使用 extract_from_context 方法")
    
    def extract_from_context(self, ctx: TradesContext, start_ts: pd.Timestamp, 
                           end_ts: pd.Timestamp, bars: Optional[pd.DataFrame] = None,
                           bar_window_start_idx: Optional[int] = None,
                           bar_window_end_idx: Optional[int] = None) -> Dict[str, float]:
        """从交易上下文中提取特征"""
        if bars is not None and bar_window_start_idx is not None and bar_window_end_idx is not None:
            s, e = bars.loc[bar_window_start_idx, 'start_trade_idx'], bars.loc[bar_window_end_idx, 'end_trade_idx'] + 1
        else:
            s, e = ctx.locate(start_ts, end_ts)
        
        if e - s <= 0:
            return {}
        
        return self._extract_features(ctx, s, e, start_ts, end_ts)
    
    def _extract_features(self, ctx: TradesContext, s: int, e: int,
                         start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Dict[str, float]:
        """提取特征的核心方法，由子类实现"""
        raise NotImplementedError("子类必须实现此方法")
    
    def _sum_range(self, prefix: np.ndarray, s: int, e: int) -> float:
        """计算前缀和的区间和"""
        if e <= s:
            return 0.0
        return float(prefix[e - 1] - (prefix[s - 1] if s > 0 else 0.0))
    
    def _compute_correlation(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算相关系数"""
        if a.size != b.size or a.size < 3:
            return np.nan
        sa = np.std(a)
        sb = np.std(b)
        if sa == 0 or sb == 0:
            return np.nan
        c = np.corrcoef(a, b)[0, 1]
        return float(c) if np.isfinite(c) else np.nan
