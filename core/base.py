"""
核心基础类和接口定义
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


class BaseDataProcessor(ABC):
    """数据处理器基类"""
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理数据的抽象方法"""
        pass
    
    @abstractmethod
    def validate_input(self, data: pd.DataFrame) -> bool:
        """验证输入数据格式"""
        pass


class BaseFeatureExtractor(ABC):
    """特征提取器基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    def extract(self, data: pd.DataFrame) -> Dict[str, float]:
        """提取特征的抽象方法"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        pass


class BaseModel(ABC):
    """机器学习模型基类"""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """预测"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        pass


class BaseValidator(ABC):
    """交叉验证器基类"""
    
    @abstractmethod
    def split(self, X: pd.DataFrame, y: pd.Series) -> List[tuple]:
        """分割数据"""
        pass
    
    @abstractmethod
    def evaluate(self, model: BaseModel, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """评估模型"""
        pass


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self._configs = {}
    
    def set_config(self, module: str, config: Dict[str, Any]):
        """设置模块配置"""
        self._configs[module] = config
    
    def get_config(self, module: str) -> Dict[str, Any]:
        """获取模块配置"""
        return self._configs.get(module, {})
    
    def update_config(self, module: str, updates: Dict[str, Any]):
        """更新模块配置"""
        if module not in self._configs:
            self._configs[module] = {}
        self._configs[module].update(updates)
