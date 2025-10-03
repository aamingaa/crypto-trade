"""
交易分析管道主控制器
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

from core.base import ConfigManager
from data.trades_processor import TradesProcessor
from data.dollar_bars import DollarBarBuilder
# from features import MicrostructureFeatureExtractor
from features.microstructure_extractor import MicrostructureFeatureExtractor
from ml.models import ModelFactory
from ml.validators import PurgedBarValidator
from utils.visualization import TradingVisualizer


class TradingPipeline:
    """交易分析管道主控制器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config_manager = ConfigManager()
        if config:
            for module, module_config in config.items():
                self.config_manager.set_config(module, module_config)
        
        # 初始化组件
        self.trades_processor = TradesProcessor()
        self.feature_extractor = MicrostructureFeatureExtractor(
            self.config_manager.get_config('features')
        )
        self.visualizer = TradingVisualizer()
        
        # 运行时数据
        self.trades_context = None
        self.bars = None
        self.features = None
        self.labels = None
        self.model = None
        self.evaluation_results = None
    
    def load_data(self, trades_data: Optional[pd.DataFrame] = None,
                  trades_zip_path: Optional[str] = None,
                  date_range: Optional[Tuple[str, str]] = None,
                  data_path_template: Optional[str] = None) -> pd.DataFrame:
        """加载交易数据"""
        if trades_data is not None:
            return trades_data
        
        if trades_zip_path and os.path.exists(trades_zip_path):
            print(f"从缓存加载数据: {trades_zip_path}")
            return pd.read_csv(trades_zip_path)
        
        if date_range and data_path_template:
            start_date, end_date = date_range
            date_list = self._generate_date_range(start_date, end_date)
            
            raw_df = []
            for date in date_list:
                file_path = data_path_template.format(date=date)
                if os.path.exists(file_path):
                    raw_df.append(pd.read_csv(file_path))
                else:
                    print(f"警告: 文件不存在 {file_path}")
            
            if not raw_df:
                raise ValueError("未找到任何数据文件")
            
            trades_df = pd.concat(raw_df, ignore_index=True)
            
            # 保存到缓存
            if trades_zip_path:
                os.makedirs(os.path.dirname(trades_zip_path), exist_ok=True)
                trades_df.to_csv(
                    trades_zip_path,
                    index=False,
                    compression={'method': 'zip', 'archive_name': 'trades_df.csv'}
                )
            
            return trades_df
        
        raise ValueError("必须提供trades_data、trades_zip_path或date_range+data_path_template")
    
    def build_bars(self, trades_df: pd.DataFrame, dollar_threshold: float,
                   bar_zip_path: Optional[str] = None) -> pd.DataFrame:
        """构建Dollar Bars"""
        if bar_zip_path and os.path.exists(bar_zip_path):
            print(f"从缓存加载bars: {bar_zip_path}")
            bars = pd.read_csv(bar_zip_path)
        else:
            print("构建Dollar Bars...")
            bar_builder = DollarBarBuilder(dollar_threshold)
            bars = bar_builder.process(trades_df)
            
            if bar_zip_path:
                os.makedirs(os.path.dirname(bar_zip_path), exist_ok=True)
                bars.to_csv(
                    bar_zip_path, 
                    index=False,
                    compression={'method': 'zip', 'archive_name': 'bars.csv'}
                )
        
        self.bars = bars
        return bars
    
    def extract_features(self, trades_df: pd.DataFrame, bars: pd.DataFrame,
                        feature_window_bars: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """提取特征和标签"""
        print("构建交易上下文...")
        self.trades_context = self.trades_processor.build_context(trades_df)
        
        print("提取特征...")
        features_list = []
        labels_list = []
        
        bars = bars.reset_index(drop=True)
        bars['bar_id'] = bars.index
        bars['start_time'] = pd.to_datetime(bars['start_time'])
        bars['end_time'] = pd.to_datetime(bars['end_time'])
        
        close_prices = bars.set_index('bar_id')['close']
        end_times = bars.set_index('bar_id')['end_time']
        
        # 计算标签（多个持有期）
        horizon_bars = [1, 3, 5, 10]
        labels_df = pd.DataFrame(index=close_prices.index)
        
        for horizon in horizon_bars:
            log_return = np.log(close_prices.shift(-horizon) / close_prices)
            labels_df[f'log_return_{horizon}'] = log_return
            labels_df[f't0_time_{horizon}'] = end_times
            labels_df[f'tH_time_{horizon}'] = end_times.shift(-horizon)
        
        # 提取特征
        idx = 1
        for bar_id in close_prices.index:
            bar_window_start_idx = bar_id - feature_window_bars
            if bar_window_start_idx < 0:
                continue
            
            bar_window_end_idx = bar_id - 1
            
            feature_start_ts = bars.loc[bar_window_start_idx, 'start_time']
            feature_end_ts = bars.loc[bar_window_end_idx, 'end_time']
            
            # 提取微观结构特征
            features = self.feature_extractor.extract_from_context(
                ctx=self.trades_context,
                start_ts=feature_start_ts,
                end_ts=feature_end_ts,
                bars=bars,
                bar_window_start_idx=bar_window_start_idx,
                bar_window_end_idx=bar_window_end_idx
            )
            
            if idx % 100 == 0:
                print(f"处理进度: {idx}/{len(close_prices.index)}")
            idx += 1
            
            features['bar_id'] = bar_id
            features['feature_start'] = feature_start_ts
            features['feature_end'] = feature_end_ts
            features['prediction_time'] = bars.loc[bar_id, 'end_time']
            
            features_list.append(features)
        
        # 构建特征DataFrame
        X = pd.DataFrame(features_list).set_index('bar_id')
        y = labels_df.loc[X.index]
        
        # 过滤无效数据
        mask = y['log_return_5'].notna() & np.isfinite(y['log_return_5'].values)
        X = X.loc[mask].replace([np.inf, -np.inf], np.nan)
        y = y.loc[X.index]
        
        self.features = X
        self.labels = y
        
        return X, y
    
    def train_and_evaluate(self, X: pd.DataFrame, y: pd.DataFrame,
                          model_type: str = 'ridge',
                          target_horizon: int = 5,
                          n_splits: int = 5,
                          embargo_bars: int = 3) -> Dict:
        """训练模型并评估"""
        print(f"开始训练{model_type}模型...")
        
        # 选择目标标签
        target_col = f'log_return_{target_horizon}'
        if target_col not in y.columns:
            raise ValueError(f"标签列{target_col}不存在")
        
        y_target = y[target_col]
        
        # 过滤特征列（排除时间相关列）
        feature_cols = [col for col in X.columns 
                       if not any(time_word in col.lower() 
                                for time_word in ['time', 'start', 'end', 'settle'])]
        X_features = X[feature_cols]
        
        # 创建模型
        model_config = self.config_manager.get_config('model')
        self.model = ModelFactory.create_model(model_type, **model_config)
        
        # 交叉验证
        validator = PurgedBarValidator(n_splits=n_splits, embargo_bars=embargo_bars)
        results = validator.evaluate(self.model, X_features, y_target)
        
        self.evaluation_results = results
        
        print("模型评估完成!")
        print(f"平均Pearson IC: {results['summary']['pearson_ic_mean']:.4f}")
        print(f"平均Spearman IC: {results['summary']['spearman_ic_mean']:.4f}")
        print(f"平均RMSE: {results['summary']['rmse_mean']:.4f}")
        print(f"平均方向准确率: {results['summary']['dir_acc_mean']:.4f}")
        
        return results
    
    def visualize_results(self, save_dir: Optional[str] = None):
        """可视化结果"""
        if self.evaluation_results is None:
            print("尚未进行模型评估")
            return
        
        predictions = self.evaluation_results['predictions']
        target_col = f'log_return_5'  # 假设使用5期作为目标
        y_true = self.labels[target_col].loc[predictions.index]
        
        self.visualizer.plot_predictions_vs_truth(
            predictions, y_true, 
            title="预测值 vs 真实值",
            save_path=os.path.join(save_dir, "predictions_vs_truth.png") if save_dir else None
        )
        
        # 绘制特征重要性
        if hasattr(self.model, 'get_feature_importance'):
            importance = self.model.get_feature_importance()
            self.visualizer.plot_feature_importance(
                importance,
                title="特征重要性",
                save_path=os.path.join(save_dir, "feature_importance.png") if save_dir else None
            )
    
    def run_full_pipeline(self, **kwargs) -> Dict:
        """运行完整的分析管道"""
        # 加载数据
        trades_df = self.load_data(**kwargs.get('data_config', {}))
        print(f"加载了{len(trades_df)}条交易记录")
        
        # 构建bars
        dollar_threshold = kwargs.get('dollar_threshold', 60000000)
        bars = self.build_bars(trades_df, dollar_threshold, 
                              kwargs.get('bar_zip_path'))
        print(f"构建了{len(bars)}个Dollar Bars")
        
        # 提取特征
        feature_window_bars = kwargs.get('feature_window_bars', 10)
        X, y = self.extract_features(trades_df, bars, feature_window_bars)
        print(f"提取了{len(X)}个样本，{len(X.columns)}个特征")
        
        # 训练评估
        results = self.train_and_evaluate(
            X, y,
            model_type=kwargs.get('model_type', 'ridge'),
            target_horizon=kwargs.get('target_horizon', 5),
            n_splits=kwargs.get('n_splits', 5),
            embargo_bars=kwargs.get('embargo_bars', 3)
        )
        
        # 可视化
        if kwargs.get('save_plots'):
            self.visualize_results(kwargs.get('plot_save_dir'))
        
        return {
            'evaluation': results,
            'features': X,
            'labels': y,
            'bars': bars,
            'model': self.model
        }
    
    def _generate_date_range(self, start_date: str, end_date: str) -> List[str]:
        """生成日期范围"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        date_list = []
        current = start
        while current <= end:
            date_list.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        return date_list
