"""
交叉验证模块
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import mean_squared_error
from core.base import BaseValidator, BaseModel


class PurgedTimeSeriesValidator(BaseValidator):
    """Purged时间序列交叉验证器"""
    
    def __init__(self, n_splits: int = 5, embargo: str = '1H'):
        self.n_splits = n_splits
        self.embargo = embargo
    
    def split(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """分割数据"""
        return self._time_splits_purged(X.index, self.n_splits, self.embargo)
    
    def evaluate(self, model: BaseModel, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """评估模型"""
        splits = self.split(X, y)
        fold_results = []
        predictions_all = pd.Series(index=X.index, dtype=float)
        
        for fold_id, (train_idx, test_idx) in enumerate(splits):
            X_train, y_train = X.loc[train_idx], y.loc[train_idx]
            X_test, y_test = X.loc[test_idx], y.loc[test_idx]
            
            # 训练模型
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions_all.loc[test_idx] = y_pred
            
            # 计算指标
            fold_metrics = self._compute_metrics(y_test, y_pred)
            fold_metrics['fold'] = fold_id
            fold_metrics['n_train'] = len(X_train)
            fold_metrics['n_test'] = len(X_test)
            fold_results.append(fold_metrics)
        
        # 汇总结果
        df_folds = pd.DataFrame(fold_results)
        summary = self._summarize_results(df_folds)
        
        return {
            'by_fold': fold_results,
            'summary': summary,
            'predictions': predictions_all,
        }
    
    def _time_splits_purged(self, idx: pd.DatetimeIndex, n_splits: int, 
                           embargo: str) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """生成时间连续的折，返回 (train_index, test_index) 对列表"""
        times = pd.Series(index=idx.unique().sort_values(), data=np.arange(len(idx.unique())))
        n = len(times)
        if n_splits < 2 or n < n_splits:
            raise ValueError('样本过少，无法进行时间序列CV')

        fold_sizes = [n // n_splits] * n_splits
        for i in range(n % n_splits):
            fold_sizes[i] += 1

        # 计算各折在时间索引上的切片范围
        boundaries = []
        start = 0
        for sz in fold_sizes:
            end = start + sz
            boundaries.append((start, end))
            start = end

        embargo_td = pd.Timedelta(embargo)
        out: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
        
        for (s, e) in boundaries:
            test_times = times.index[s:e]
            test_mask = idx.isin(test_times)

            test_start = test_times.min()
            test_end = test_times.max()

            left_block = (idx >= (test_start - embargo_td)) & (idx < test_start)
            right_block = (idx > test_end) & (idx <= (test_end + embargo_td))
            exclude = left_block | right_block | test_mask
            
            train_idx = idx[~exclude]
            test_idx = idx[test_mask]
            
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            out.append((train_idx, test_idx))
        
        return out
    
    def _compute_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """计算评估指标"""
        # 相关性指标
        pearson_ic = y_pred.corr(y_true)
        spearman_ic = y_pred.corr(y_true, method='spearman')
        
        # 误差指标
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        
        # 方向准确率
        dir_acc = (np.sign(y_pred) == np.sign(y_true)).mean()
        
        # 简单交易指标
        pos = np.sign(y_pred).fillna(0.0)
        ret_gross = (pos * y_true).astype(float)
        turnover = pos.diff().abs().fillna(np.abs(pos.iloc[0]))
        fee_rate = 1e-4
        ret_net = ret_gross - fee_rate * turnover
        sharpe_net = float(ret_net.mean() / ret_net.std()) if ret_net.std() > 0 else np.nan
        
        return {
            'pearson_ic': float(pearson_ic) if pd.notna(pearson_ic) else np.nan,
            'spearman_ic': float(spearman_ic) if pd.notna(spearman_ic) else np.nan,
            'rmse': float(rmse),
            'dir_acc': float(dir_acc),
            'ret_gross_mean': float(ret_gross.mean()),
            'ret_net_mean': float(ret_net.mean()),
            'ret_net_std': float(ret_net.std()) if ret_net.std() > 0 else np.nan,
            'sharpe_net': sharpe_net,
        }
    
    def _summarize_results(self, df_folds: pd.DataFrame) -> Dict[str, float]:
        """汇总结果"""
        return {
            'pearson_ic_mean': float(df_folds['pearson_ic'].mean()) if not df_folds.empty else np.nan,
            'spearman_ic_mean': float(df_folds['spearman_ic'].mean()) if not df_folds.empty else np.nan,
            'rmse_mean': float(df_folds['rmse'].mean()) if not df_folds.empty else np.nan,
            'dir_acc_mean': float(df_folds['dir_acc'].mean()) if not df_folds.empty else np.nan,
            'ret_gross_mean_mean': float(df_folds['ret_gross_mean'].mean()) if 'ret_gross_mean' in df_folds else np.nan,
            'ret_net_mean_mean': float(df_folds['ret_net_mean'].mean()) if 'ret_net_mean' in df_folds else np.nan,
            'sharpe_net_mean': float(df_folds['sharpe_net'].mean()) if 'sharpe_net' in df_folds else np.nan,
            'n_splits_effective': int(len(df_folds)),
        }


class PurgedBarValidator(BaseValidator):
    """基于Bar计数的Purged交叉验证器"""
    
    def __init__(self, n_splits: int = 5, embargo_bars: int = 0):
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars
    
    def split(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """分割数据"""
        return self._bar_splits_purged(X.index, self.n_splits, self.embargo_bars)
    
    def evaluate(self, model: BaseModel, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """评估模型"""
        splits = self.split(X, y)
        fold_results = []
        predictions_all = pd.Series(index=X.index, dtype=float)
        
        for fold_id, (train_idx, test_idx) in enumerate(splits):
            X_train, y_train = X.loc[train_idx], y.loc[train_idx]
            X_test, y_test = X.loc[test_idx], y.loc[test_idx]
            
            # 训练模型
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions_all.loc[test_idx] = y_pred
            
            # 计算指标
            fold_metrics = self._compute_metrics(y_test, y_pred)
            fold_metrics['fold'] = fold_id
            fold_metrics['n_train'] = len(X_train)
            fold_metrics['n_test'] = len(X_test)
            fold_results.append(fold_metrics)
        
        # 汇总结果
        df_folds = pd.DataFrame(fold_results)
        summary = self._summarize_results(df_folds)
        
        return {
            'by_fold': fold_results,
            'summary': summary,
            'predictions': predictions_all,
        }
    
    def _bar_splits_purged(self, idx: pd.DatetimeIndex, n_splits: int, 
                          embargo_bars: int) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """基于bar顺序的Purged K-Fold划分"""
        n = len(idx)
        if n_splits < 2 or n < n_splits:
            raise ValueError('样本过少，无法进行时间序列CV')

        # 均分折大小
        fold_sizes = [n // n_splits] * n_splits
        for i in range(n % n_splits):
            fold_sizes[i] += 1

        # 累积得到每折的位置范围
        boundaries: List[Tuple[int, int]] = []
        start = 0
        for sz in fold_sizes:
            end = start + sz
            boundaries.append((start, end))
            start = end

        embargo_bars = int(max(0, embargo_bars))
        positions = np.arange(n)
        out: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
        
        for (s, e) in boundaries:
            test_mask_pos = (positions >= s) & (positions < e)
            # 左右屏蔽区
            left_start = max(0, s - embargo_bars)
            left_end = s
            right_start = e
            right_end = min(n, e + embargo_bars)
            left_block_pos = (positions >= left_start) & (positions < left_end)
            right_block_pos = (positions >= right_start) & (positions < right_end)

            exclude_pos = left_block_pos | right_block_pos | test_mask_pos
            train_idx = idx[~exclude_pos]
            test_idx = idx[test_mask_pos]
            
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            out.append((train_idx, test_idx))
        
        return out
    
    def _compute_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """计算评估指标"""
        # 相关性指标
        pearson_ic = y_pred.corr(y_true)
        spearman_ic = y_pred.corr(y_true, method='spearman')
        
        # 误差指标
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        
        # 方向准确率
        dir_acc = (np.sign(y_pred) == np.sign(y_true)).mean()
        
        # 简单交易指标
        pos = np.sign(y_pred).fillna(0.0)
        ret_gross = (pos * y_true).astype(float)
        turnover = pos.diff().abs().fillna(np.abs(pos.iloc[0]))
        fee_rate = 1e-4
        ret_net = ret_gross - fee_rate * turnover
        sharpe_net = float(ret_net.mean() / ret_net.std()) if ret_net.std() > 0 else np.nan
        
        return {
            'pearson_ic': float(pearson_ic) if pd.notna(pearson_ic) else np.nan,
            'spearman_ic': float(spearman_ic) if pd.notna(spearman_ic) else np.nan,
            'rmse': float(rmse),
            'dir_acc': float(dir_acc),
            'ret_gross_mean': float(ret_gross.mean()),
            'ret_net_mean': float(ret_net.mean()),
            'ret_net_std': float(ret_net.std()) if ret_net.std() > 0 else np.nan,
            'sharpe_net': sharpe_net,
        }
    
    def _summarize_results(self, df_folds: pd.DataFrame) -> Dict[str, float]:
        """汇总结果"""
        return {
            'pearson_ic_mean': float(df_folds['pearson_ic'].mean()) if not df_folds.empty else np.nan,
            'spearman_ic_mean': float(df_folds['spearman_ic'].mean()) if not df_folds.empty else np.nan,
            'rmse_mean': float(df_folds['rmse'].mean()) if not df_folds.empty else np.nan,
            'dir_acc_mean': float(df_folds['dir_acc'].mean()) if not df_folds.empty else np.nan,
            'ret_gross_mean_mean': float(df_folds['ret_gross_mean'].mean()) if 'ret_gross_mean' in df_folds else np.nan,
            'ret_net_mean_mean': float(df_folds['ret_net_mean'].mean()) if 'ret_net_mean' in df_folds else np.nan,
            'sharpe_net_mean': float(df_folds['sharpe_net'].mean()) if 'sharpe_net' in df_folds else np.nan,
            'n_splits_effective': int(len(df_folds)),
        }
