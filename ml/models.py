"""
机器学习模型模块 - 修复NaN处理问题
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from core.base import BaseModel


class RidgeModel(BaseModel):
    """Ridge回归模型"""
    
    def __init__(self, alpha: float = 1.0, random_state: int = 42):
        self.alpha = alpha
        self.random_state = random_state
        self.model = Ridge(alpha=alpha, random_state=random_state)
        self.scaler = StandardScaler()
        self.feature_names: Optional[list] = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RidgeModel':
        """训练模型"""
        self.feature_names = list(X.columns)
        
        # 处理NaN值 - 用0填充
        X_clean = X.fillna(0)
        
        # 标准化特征
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_clean.values),
            index=X_clean.index,
            columns=X_clean.columns
        )
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        # 处理NaN值 - 用0填充（与训练时一致）
        X_clean = X.fillna(0)
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_clean.values),
            index=X_clean.index,
            columns=X_clean.columns
        )
        
        predictions = self.model.predict(X_scaled)
        return pd.Series(predictions, index=X.index)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性（系数绝对值）"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        coefficients = np.abs(self.model.coef_)
        return dict(zip(self.feature_names, coefficients))


class RandomForestModel(BaseModel):
    """随机森林模型"""
    
    def __init__(self, n_estimators: int = 300, max_depth: int = 8, 
                 random_state: int = 42, n_jobs: int = -1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.feature_names: Optional[list] = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomForestModel':
        """训练模型"""
        self.feature_names = list(X.columns)
        
        # 处理NaN值 - RandomForest可以处理NaN，但为了一致性也进行处理
        X_clean = X.fillna(0)
        
        self.model.fit(X_clean, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        # 处理NaN值 - 用0填充（与训练时一致）
        X_clean = X.fillna(0)
        
        predictions = self.model.predict(X_clean)
        return pd.Series(predictions, index=X.index)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


class LinearModel(BaseModel):
    """线性回归模型"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_names: Optional[list] = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LinearModel':
        """训练模型"""
        self.feature_names = list(X.columns)
        
        # 处理NaN值 - 用0填充
        X_clean = X.fillna(0)
        
        # 标准化特征
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_clean.values),
            index=X_clean.index,
            columns=X_clean.columns
        )
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        # 处理NaN值 - 用0填充（与训练时一致）
        X_clean = X.fillna(0)
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_clean.values),
            index=X_clean.index,
            columns=X_clean.columns
        )
        
        predictions = self.model.predict(X_scaled)
        return pd.Series(predictions, index=X.index)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性（系数绝对值）"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        coefficients = np.abs(self.model.coef_)
        return dict(zip(self.feature_names, coefficients))


class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """创建模型实例"""
        if model_type in ('ridge', 'ridge_reg'):
            # Ridge模型支持的参数
            valid_params = {'alpha', 'random_state'}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            return RidgeModel(**filtered_kwargs)
        elif model_type == 'rf':
            # RandomForest模型支持的参数
            valid_params = {'n_estimators', 'max_depth', 'random_state', 'n_jobs'}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            return RandomForestModel(**filtered_kwargs)
        elif model_type in ('linear', 'ols', 'linreg'):
            # LinearModel不需要额外参数
            return LinearModel()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    @staticmethod
    def get_available_models() -> list:
        """获取可用的模型类型"""
        return ['ridge', 'rf', 'linear']
    
    @staticmethod
    def get_model_params(model_type: str) -> list:
        """获取模型支持的参数列表"""
        param_mapping = {
            'ridge': ['alpha', 'random_state'],
            'rf': ['n_estimators', 'max_depth', 'random_state', 'n_jobs'],
            'linear': []
        }
        return param_mapping.get(model_type, [])
