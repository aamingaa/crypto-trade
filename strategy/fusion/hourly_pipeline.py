import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from util.dollarbar_fusion import (
    build_dollar_bars,
    aggregate_trade_features_on_bars,
)


 


def _add_bar_lags_and_rollings(
    Xb: pd.DataFrame,
    add_lags: int = 2,
    rolling_windows: Optional[List[int]] = None,
    rolling_stats: Optional[List[str]] = None,
) -> pd.DataFrame:
    X = Xb.copy().sort_index()

    feature_cols = list(X.columns)

    # lags
    for k in range(1, add_lags + 1):
        for col in feature_cols:
            X[f'{col}_lag{k}'] = X[col].shift(k)

    # rollings
    if rolling_windows:
        stats = rolling_stats or ['mean', 'std', 'sum']
        for w in rolling_windows:
            roll = X[feature_cols].rolling(window=w, min_periods=w)
            for stat in stats:
                if stat == 'mean':
                    tmp = roll.mean()
                elif stat == 'std':
                    tmp = roll.std()
                elif stat == 'sum':
                    tmp = roll.sum()
                elif stat == 'min':
                    tmp = roll.min()
                elif stat == 'max':
                    tmp = roll.max()
                else:
                    continue
                tmp.columns = [f'{col}_roll{w}_{stat}' for col in X[feature_cols].columns]
                X = X.join(tmp)

    return X


 


def _time_splits_purged(
    idx: pd.DatetimeIndex,
    n_splits: int = 5,
    embargo: str = '0H',
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    生成时间连续的折，返回 (train_index, test_index) 对列表。
    训练集对测试集边界施加 Embargo，避免信息泄露。
    """
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

        left_block = (idx <= (test_start + embargo_td))
        right_block = (idx >= (test_end - embargo_td))
        exclude = left_block | right_block | test_mask
        train_idx = idx[~exclude]
        test_idx = idx[test_mask]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        out.append((train_idx, test_idx))
    return out


def purged_cv_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    embargo: str = '1H',
    model_type: str = 'ridge',
    random_state: int = 42,
) -> Dict:
    """
    使用 Purged 时间序列 CV 进行回归评估，返回按折与汇总的指标。
    指标：Pearson IC、Spearman IC、RMSE、方向准确率。
    """
    assert X.index.equals(y.index)

    # 选择模型
    if model_type == 'rf':
        base_model = RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=random_state, n_jobs=-1
        )
        use_scaler = False
    else:
        base_model = Ridge(alpha=1.0, random_state=random_state)
        use_scaler = True

    splits = _time_splits_purged(X.index, n_splits=n_splits, embargo=embargo)
    by_fold = []
    preds_all = pd.Series(index=X.index, dtype=float)

    for fold_id, (tr_idx, te_idx) in enumerate(splits):
        Xtr, ytr = X.loc[tr_idx], y.loc[tr_idx]
        Xte, yte = X.loc[te_idx], y.loc[te_idx]

        if use_scaler:
            scaler = StandardScaler()
            Xtr_scaled = pd.DataFrame(
                scaler.fit_transform(Xtr.values), index=Xtr.index, columns=Xtr.columns
            )
            Xte_scaled = pd.DataFrame(
                scaler.transform(Xte.values), index=Xte.index, columns=Xte.columns
            )
        else:
            Xtr_scaled, Xte_scaled = Xtr, Xte

        model = base_model
        model.fit(Xtr_scaled, ytr)
        yhat = pd.Series(model.predict(Xte_scaled), index=te_idx)
        preds_all.loc[te_idx] = yhat

        # 指标
        pearson_ic = yhat.corr(yte)
        spearman_ic = yhat.corr(yte, method='spearman')
        rmse = mean_squared_error(yte, yhat) ** 0.5
        dir_acc = (np.sign(yhat) == np.sign(yte)).mean()
        by_fold.append({
            'fold': fold_id,
            'pearson_ic': float(pearson_ic),
            'spearman_ic': float(spearman_ic),
            'rmse': float(rmse),
            'dir_acc': float(dir_acc),
            'n_train': int(len(Xtr)),
            'n_test': int(len(Xte)),
        })

    # 汇总
    df_folds = pd.DataFrame(by_fold)
    summary = {
        'pearson_ic_mean': float(df_folds['pearson_ic'].mean()) if not df_folds.empty else np.nan,
        'spearman_ic_mean': float(df_folds['spearman_ic'].mean()) if not df_folds.empty else np.nan,
        'rmse_mean': float(df_folds['rmse'].mean()) if not df_folds.empty else np.nan,
        'dir_acc_mean': float(df_folds['dir_acc'].mean()) if not df_folds.empty else np.nan,
        'n_splits_effective': int(len(df_folds)),
    }
    return {
        'by_fold': by_fold,
        'summary': summary,
        'predictions': preds_all,
    }


def make_barlevel_dataset(
    trades: pd.DataFrame,
    dollar_threshold: float,
    horizon_bars: int = 1,
    add_lags: int = 2,
    rolling_windows: Optional[List[int]] = None,
    rolling_stats: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    逐笔 -> Dollar Bars -> Bar特征 -> N-bar 标签（不做小时对齐）。
    返回：X_bar, y_bar, bars, bar_features
    """
    # 1) 事件Bar
    bars = build_dollar_bars(trades, dollar_threshold=dollar_threshold)
    if bars.empty:
        return pd.DataFrame(), pd.Series(dtype=float), bars, pd.DataFrame()

    # 2) Bar级特征（逐笔交易侧）
    bar_feat = aggregate_trade_features_on_bars(trades, bars)

    # 3) 标签：未来 N 根 bar 的对数收益
    close = bar_feat['close'] if 'close' in bar_feat.columns else bars.set_index('bar_id')['close']
    y_bar = np.log(close.shift(-horizon_bars) / close)

    # 4) 特征工程：去掉时间列，仅保留数值特征
    keep_cols = [c for c in bar_feat.columns if c not in ['start_time', 'end_time']]
    X_bar = bar_feat[keep_cols]
    X_bar = _add_bar_lags_and_rollings(
        X_bar,
        add_lags=add_lags,
        rolling_windows=rolling_windows,
        rolling_stats=rolling_stats,
    )

    # 5) 对齐与去NaN
    X_bar = X_bar.dropna()
    y_bar = y_bar.loc[X_bar.index]

    return X_bar, y_bar, bars, bar_feat


def run_barlevel_pipeline(
    trades: pd.DataFrame,
    dollar_threshold: float,
    horizon_bars: int = 1,
    add_lags: int = 2,
    rolling_windows: Optional[List[int]] = None,
    rolling_stats: Optional[List[str]] = None,
    n_splits: int = 5,
    embargo_bars: int = 0,
    model_type: str = 'ridge',
    random_state: int = 42,
) -> Dict:
    """
    端到端：逐笔 -> 事件Bar -> Bar级特征 -> Purged CV（按bar索引，使用等距切片近似）
    注意：这里的 purged 分割在 bar 索引上做，embargo_bars 控制左右留白根数。
    """
    Xb, yb, bars, bar_feat = make_barlevel_dataset(
        trades=trades,
        dollar_threshold=dollar_threshold,
        horizon_bars=horizon_bars,
        add_lags=add_lags,
        rolling_windows=rolling_windows,
        rolling_stats=rolling_stats,
    )

    if Xb.empty or yb.empty:
        return {
            'error': '数据不足或阈值设置过大，无法构造训练集（bar级）',
            'X_bar': Xb,
            'y_bar': yb,
            'bars': bars,
            'bar_features': bar_feat,
        }

    # 使用真实 bar 的 end_time 作为时间索引
    bar_times = bars.set_index('bar_id')['end_time']
    idx_time = pd.to_datetime(bar_times.loc[Xb.index])
    Xb2 = Xb.copy(); Xb2.index = idx_time
    yb2 = yb.copy(); yb2.index = idx_time

    # 将 embargo_bars 转换为时间间隔：使用 bar 的中位时长作为单位
    bar_durations = (bars['end_time'] - bars['start_time']).dropna()
    median_duration = bar_durations.median() if not bar_durations.empty else pd.Timedelta(0)
    embargo = median_duration * int(max(0, embargo_bars))
    eval_result = purged_cv_evaluate(
        X=Xb2,
        y=yb2,
        n_splits=n_splits,
        embargo=embargo,
        model_type=model_type,
        random_state=random_state,
    )

    out = {
        'eval': eval_result,
        'X_bar': Xb,
        'y_bar': yb,
        'bars': bars,
        'bar_features': bar_feat,
    }
    return out


def _factor_int_trade_stats(seg: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Dict[str, float]:
    if seg.empty:
        duration_seconds = max(1.0, (end_ts - start_ts).total_seconds())
        return {
            'int_trade_vwap': np.nan,
            'int_trade_volume_sum': 0.0,
            'int_trade_dollar_sum': 0.0,
            'int_trade_signed_volume': 0.0,
            'int_trade_buy_ratio': np.nan,
            'int_trade_intensity': 0.0,
            'int_trade_rv': np.nan,
        }
    dollar = seg['quote_qty'].sum()
    qty_sum = seg['qty'].sum()
    vwap = (seg['price'] * seg['qty']).sum() / qty_sum if qty_sum > 0 else np.nan
    signed_volume = (seg['qty'] * seg['trade_sign']).sum()
    buy_ratio = seg.loc[seg['trade_sign'] > 0, 'qty'].sum() / qty_sum if qty_sum > 0 else np.nan
    seg = seg.copy()
    seg['logp'] = np.log(seg['price'])
    rv = (seg['logp'].diff().dropna() ** 2).sum()
    duration_seconds = max(1.0, (end_ts - start_ts).total_seconds())
    intensity = len(seg) / duration_seconds
    return {
        'int_trade_vwap': float(vwap) if pd.notna(vwap) else np.nan,
        'int_trade_volume_sum': float(qty_sum),
        'int_trade_dollar_sum': float(dollar),
        'int_trade_signed_volume': float(signed_volume),
        'int_trade_buy_ratio': float(buy_ratio) if pd.notna(buy_ratio) else np.nan,
        'int_trade_intensity': float(intensity),
        'int_trade_rv': float(rv) if pd.notna(rv) else np.nan,
    }


def _factor_order_flow_imbalance(seg: pd.DataFrame) -> Dict[str, float]:
    if seg.empty:
        return {
            'ofi_signed_qty_sum': 0.0,
            'ofi_signed_quote_sum': 0.0,
            'ofi_trade_count_diff': 0.0,
        }
    signed_qty = float((seg['qty'] * seg['trade_sign']).sum())
    signed_quote = float((seg['quote_qty'] * seg['trade_sign']).sum())
    trade_count_diff = float((seg['trade_sign'] > 0).sum() - (seg['trade_sign'] < 0).sum())
    return {
        'ofi_signed_qty_sum': signed_qty,
        'ofi_signed_quote_sum': signed_quote,
        'ofi_trade_count_diff': trade_count_diff,
    }


def _factor_garman_order_flow(seg: pd.DataFrame) -> Dict[str, float]:
    if seg.empty:
        return {'gof_by_count': np.nan, 'gof_by_volume': np.nan}
    n = len(seg)
    qty_sum = seg['qty'].sum()
    gof_by_count = float(seg['trade_sign'].sum()) / n if n > 0 else np.nan
    gof_by_volume = float((seg['trade_sign'] * seg['qty']).sum()) / qty_sum if qty_sum > 0 else np.nan
    return {
        'gof_by_count': gof_by_count,
        'gof_by_volume': gof_by_volume,
    }


def _factor_rolling_ofi(seg: pd.DataFrame) -> Dict[str, float]:
    if seg.empty:
        return {'ofi_roll_sum_max': 0.0, 'ofi_roll_sum_std': 0.0}
    n = len(seg)
    roll_n = max(1, min(20, n))
    signed_qty_series = (seg['trade_sign'] * seg['qty']).astype(float).reset_index(drop=True)
    ofi_roll = signed_qty_series.rolling(window=roll_n, min_periods=roll_n).sum()
    ofi_roll_max = float(ofi_roll.max()) if len(ofi_roll.dropna()) else 0.0
    ofi_roll_std = float(ofi_roll.std()) if len(ofi_roll.dropna()) else 0.0
    return {'ofi_roll_sum_max': ofi_roll_max, 'ofi_roll_sum_std': ofi_roll_std}


def _factor_activity_and_persistence(seg: pd.DataFrame) -> Dict[str, float]:
    if seg.empty:
        return {
            'runlen_buy_max': 0.0,
            'runlen_sell_max': 0.0,
            'runlen_buy_mean': np.nan,
            'runlen_sell_mean': np.nan,
            'alt_frequency': np.nan,
            'p_pos_to_pos': np.nan,
            'p_neg_to_neg': np.nan,
        }
    sgn = seg['trade_sign'].astype(int).to_numpy()
    n = len(sgn)
    # run-lengths
    run_lengths = []
    cur = sgn[0]
    length = 1
    for val in sgn[1:]:
        if val == cur:
            length += 1
        else:
            run_lengths.append((cur, length))
            cur = val
            length = 1
    run_lengths.append((cur, length))
    buy_runs = [l for s, l in run_lengths if s > 0]
    sell_runs = [l for s, l in run_lengths if s < 0]
    runlen_buy_max = float(max(buy_runs)) if buy_runs else 0.0
    runlen_sell_max = float(max(sell_runs)) if sell_runs else 0.0
    runlen_buy_mean = float(np.mean(buy_runs)) if buy_runs else np.nan
    runlen_sell_mean = float(np.mean(sell_runs)) if sell_runs else np.nan
    # 交替频率
    flips = int((np.diff(sgn) != 0).sum()) if n > 1 else 0
    alt_frequency = flips / (n - 1) if n > 1 else np.nan
    # Markov 转移概率
    p_pos_to_pos = np.nan
    p_neg_to_neg = np.nan
    if n > 1:
        from_pos = sgn[:-1] > 0
        to_pos = sgn[1:] > 0
        from_neg = sgn[:-1] < 0
        to_neg = sgn[1:] < 0
        pos_count = int(from_pos.sum())
        neg_count = int(from_neg.sum())
        p_pos_to_pos = float((from_pos & to_pos).sum() / pos_count) if pos_count > 0 else np.nan
        p_neg_to_neg = float((from_neg & to_neg).sum() / neg_count) if neg_count > 0 else np.nan
    return {
        'runlen_buy_max': runlen_buy_max,
        'runlen_sell_max': runlen_sell_max,
        'runlen_buy_mean': runlen_buy_mean,
        'runlen_sell_mean': runlen_sell_mean,
        'alt_frequency': float(alt_frequency) if pd.notna(alt_frequency) else np.nan,
        'p_pos_to_pos': p_pos_to_pos,
        'p_neg_to_neg': p_neg_to_neg,
    }


def _compute_interval_trade_features(trades: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Dict[str, float]:
    """
    在给定时间区间 [start_ts, end_ts) 内，逐笔计算区间特征。
    每个特征由独立函数实现并组合返回。
    """
    df = trades.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    if 'quote_qty' not in df.columns or df['quote_qty'].isna().all():
        df['quote_qty'] = df['price'] * df['qty']
    df['trade_sign'] = np.where(df['is_buyer_maker'], -1, 1)
    seg = df.loc[(df['time'] >= start_ts) & (df['time'] < end_ts)].copy()

    out = {}
    out.update(_factor_int_trade_stats(seg, start_ts, end_ts))
    out.update(_factor_order_flow_imbalance(seg))
    out.update(_factor_garman_order_flow(seg))
    out.update(_factor_rolling_ofi(seg))
    out.update(_factor_activity_and_persistence(seg))
    return out


def make_interval_feature_dataset(
    trades: pd.DataFrame,
    dollar_threshold: float,
    horizon_bars: int = 3,
    window_mode: str = 'past',  # 'past' 使用过去N个bar的[start,end)，'future' 使用未来N个bar（注意可能泄露）
    add_lags: int = 0,
    rolling_windows: Optional[List[int]] = None,
    rolling_stats: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    标签：由 dollar bars 生成的未来 N-bar 对数收益。
    因子：对齐到对应区间 [start_time, end_time)（过去或未来N个bar）上基于逐笔成交直接计算。
    返回：X_interval, y, bars
    """
    bars = build_dollar_bars(trades, dollar_threshold=dollar_threshold)
    if bars.empty:
        return pd.DataFrame(), pd.Series(dtype=float), bars

    bars = bars.reset_index(drop=True)
    bars['bar_id'] = bars.index
    close_s = bars.set_index('bar_id')['close']

    # 未来 N-bar 对数收益
    y = np.log(close_s.shift(-horizon_bars) / close_s)

    # 计算每个样本的区间
    features = []
    for bar_id in close_s.index:
        if window_mode == 'past':
            start_idx = bar_id - horizon_bars
            end_idx = bar_id - 1
            if start_idx < 0:
                features.append({'bar_id': bar_id, '_skip': True})
                continue
        else:  # future 区间（注意信息泄露，仅在需要时使用）
            start_idx = bar_id
            end_idx = bar_id + horizon_bars - 1
            if end_idx >= len(bars):
                features.append({'bar_id': bar_id, '_skip': True})
                continue

        start_ts = bars.loc[start_idx, 'start_time']
        end_ts = bars.loc[end_idx, 'end_time']
        feat = _compute_interval_trade_features(trades, start_ts, end_ts)
        feat['bar_id'] = bar_id
        feat['interval_start'] = start_ts
        feat['interval_end'] = end_ts
        features.append(feat)

    X = pd.DataFrame(features).set_index('bar_id')
    if '_skip' in X.columns:
        keep_idx = X['_skip'] != True
        X = X.loc[keep_idx].drop(columns=['_skip'])
    
    # 对齐标签
    y = y.loc[X.index]

    # 可选：对区间因子再做滞后/滚动（通常不需要，默认不加）
    if add_lags or rolling_windows:
        X = _add_bar_lags_and_rollings(
            X,
            add_lags=add_lags,
            rolling_windows=rolling_windows,
            rolling_stats=rolling_stats,
        ).dropna()
        y = y.loc[X.index]

    return X, y, bars


def run_bar_interval_pipeline(
    trades: pd.DataFrame,
    dollar_threshold: float,
    horizon_bars: int = 3,
    window_mode: str = 'past',
    n_splits: int = 5,
    embargo_bars: int = 1,
    model_type: str = 'ridge',
    random_state: int = 42,
) -> Dict:
    """
    按 N 个 dollar bar 定义的时间区间计算因子，标签为未来 N-bar 对数收益；
    使用区间的真实 end_time 作为索引做 Purged K-Fold，embargo 由 bar 中位时长换算。
    """
    X, y, bars = make_interval_feature_dataset(
        trades=trades,
        dollar_threshold=dollar_threshold,
        horizon_bars=horizon_bars,
        window_mode=window_mode,
    )

    if X.empty or y.empty:
        return {'error': '数据不足或阈值设置过大，无法构造区间数据集', 'X': X, 'y': y, 'bars': bars}

    # 用对应样本的区间结束时间作为索引
    # 若为 past 模式：使用当前锚点 bar 的 end_time 代表预测时点
    end_times = bars.set_index('bar_id')['end_time']
    idx_time = pd.to_datetime(end_times.loc[X.index])
    X2 = X.copy(); X2.index = idx_time
    y2 = y.copy(); y2.index = idx_time

    # embargo 转换为时间长度
    durations = (bars['end_time'] - bars['start_time']).dropna()
    median_duration = durations.median() if not durations.empty else pd.Timedelta(0)
    embargo_td = median_duration * int(max(0, embargo_bars))

    eval_result = purged_cv_evaluate(
        X=X2,
        y=y2,
        n_splits=n_splits,
        embargo=embargo_td,
        model_type=model_type,
        random_state=random_state,
    )

    return {'eval': eval_result, 'X': X, 'y': y, 'bars': bars}

 


__all__ = [
    'purged_cv_evaluate',
    'make_barlevel_dataset',
    'run_barlevel_pipeline',
    'make_interval_feature_dataset',
    'run_bar_interval_pipeline',
]

def generate_date_range(start_date, end_date):    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    return date_list

def main():    
    raw_df = []
    date_list = generate_date_range('2025-01-01', '2025-01-02')
    for date in date_list:
        raw_df.append(pd.read_csv(f'/Volumes/Ext-Disk/data/futures/um/daily/trades/ETHUSDT/ETHUSDT-trades-{date}.zip'))
        
    # dollar_bar = build_dollar_bars(raw_df, 10000 * 2000)
    # print(dollar_bar)

    trades_df = pd.concat(raw_df, ignore_index=True)
    res = run_bar_interval_pipeline(
        trades=trades_df,
        dollar_threshold=10000 * 2000,
        horizon_bars=3,
        window_mode='past',
        n_splits=5,
        embargo_bars=1,
        model_type='ridge',
    )
    print(res.get('eval', {}).get('summary'))

if __name__ == '__main__':
    main()



