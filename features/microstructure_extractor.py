"""
微观结构特征提取器主模块
整合所有子特征提取器，提供统一的接口
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from core.base import BaseFeatureExtractor
from data.trades_processor import TradesContext

from .basic_features import BasicFeatureExtractor
from .volatility_features import VolatilityFeatureExtractor
from .momentum_features import MomentumFeatureExtractor
from .orderflow_features import OrderFlowFeatureExtractor
from .impact_features import PriceImpactFeatureExtractor
from .tail_features import TailFeatureExtractor
from .path_shape_features import PathShapeFeatureExtractor


class MicrostructureFeatureExtractor(BaseFeatureExtractor):
    """微观结构特征提取器主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.feature_config = self._get_default_config()
        if config:
            self.feature_config.update(config)
        
        # 初始化各个子特征提取器
        self.extractors = {
            'basic': BasicFeatureExtractor(self._get_extractor_config(config, 'basic')),
            'volatility': VolatilityFeatureExtractor(self._get_extractor_config(config, 'volatility')),
            'momentum': MomentumFeatureExtractor(self._get_extractor_config(config, 'momentum')),
            'orderflow': OrderFlowFeatureExtractor(self._get_extractor_config(config, 'orderflow')),
            'impact': PriceImpactFeatureExtractor(self._get_extractor_config(config, 'impact')),
            'tail': TailFeatureExtractor(self._get_extractor_config(config, 'tail')),
            'path_shape': PathShapeFeatureExtractor(self._get_extractor_config(config, 'path_shape')),
        }
    
    def _get_default_config(self) -> Dict[str, bool]:
        """获取默认特征配置"""
        return {
            'basic': False,                    # 基础汇总/VWAP/强度/买量占比
            'orderflow': False,                # GOF/签名不平衡
            'impact': False,                   # Kyle/Amihud/Hasbrouck/半衰期/占比
            'volatility': True,               # RV/BPV/Jump/微动量/高低幅比
            'momentum': False,                 # 均值回复强度
            'path_shape': False,               # 协动相关/VWAP偏离
            'tail': False,                     # 大单买卖方向性（分位阈值法）
            
            # 暂未实现的特征类型
            'arrival_stats': False,           # 到达间隔统计
            'run_markov': False,              # run-length/Markov/翻转率
            'rolling_ofi': False,             # 滚动OFI
            'hawkes': False,                  # 聚簇（Hawkes近似）
            'bar_large_abs': False,           # 纯 bar 级绝对阈值大单特征
        }
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        names = []
        
        for extractor_name, enabled in self.feature_config.items():
            if enabled and extractor_name in self.extractors:
                names.extend(self.extractors[extractor_name].get_feature_names())
        
        return names
    
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
        
        features = {}
        
        # 提取各类特征
        for extractor_name, enabled in self.feature_config.items():
            if enabled and extractor_name in self.extractors:
                try:
                    extractor_features = self.extractors[extractor_name].extract_from_context(
                        ctx, start_ts, end_ts, bars, bar_window_start_idx, bar_window_end_idx
                    )
                    features.update(extractor_features)
                except Exception as e:
                    # 记录错误但继续执行其他特征提取
                    print(f"警告：提取{extractor_name}特征时出错: {e}")
                    continue
        
        return features
    
    def get_extractor(self, extractor_name: str):
        """获取指定的子特征提取器"""
        return self.extractors.get(extractor_name)
    
    def enable_feature_group(self, group_name: str, enabled: bool = True):
        """启用或禁用指定特征组"""
        if group_name in self.feature_config:
            self.feature_config[group_name] = enabled
            if group_name in self.extractors:
                self.extractors[group_name].feature_config['enabled'] = enabled
    
    def get_feature_groups(self) -> List[str]:
        """获取所有特征组名称"""
        return list(self.extractors.keys())
    
    def _get_extractor_config(self, config: Optional[Dict[str, Any]], key: str) -> Dict[str, Any]:
        """获取子特征提取器的配置"""
        if not config:
            return {}
        
        extractor_config = config.get(key, {})
        
        # 如果配置是布尔值，转换为字典格式
        if isinstance(extractor_config, bool):
            return {'enabled': extractor_config}
        elif isinstance(extractor_config, dict):
            return extractor_config
        else:
            return {}
