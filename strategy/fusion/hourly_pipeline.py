import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def _ensure_datetime(series: pd.Series) -> pd.Series:
    # 若已是datetime类型，直接返回
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    
    # 安全转换为数值类型（非数值转为NaN，避免astype直接报错）
    s = pd.to_numeric(series, errors='coerce')
    
    # 检查是否存在无法转换的非数值
    if s.isna().any():
        warnings.warn("序列中包含无法转换为数值的元素，已自动转为NaT")
    
    # 避免空序列导致的判断错误
    if s.empty:
        return pd.Series([], dtype='datetime64[ns]')
    
    # 基于2025年前后的合理时间戳范围设置阈值（单位：对应单位的数量）
    # 参考：2025年的时间戳约为 1.7e9秒 ≈ 1.7e12毫秒 ≈ 1.7e15微秒 ≈ 1.7e18纳秒
    ns_threshold = 1e17  # 纳秒级阈值（约317年，覆盖合理时间范围）
    us_threshold = 1e14  # 微秒级阈值（约317年）
    ms_threshold = 1e11  # 毫秒级阈值（约317年）
    s_abs = s.abs()  # 用绝对值判断量级，保留原始符号用于转换（支持负时间戳）
    
    # 按any()逻辑判断单位（只要有一个元素满足阈值就用对应单位）
    if (s_abs > ns_threshold).any():
        return pd.to_datetime(s, unit='ns', errors='coerce')
    elif (s_abs > us_threshold).any():
        return pd.to_datetime(s, unit='us', errors='coerce')
    elif (s_abs > ms_threshold).any():
        return pd.to_datetime(s, unit='ms', errors='coerce')
    else:
        return pd.to_datetime(s, unit='s', errors='coerce')

def generate_dollar_bars(trades_df, target_hour=1):
    """生成近似目标小时级的Dollar Bar"""
    # 计算小时级平均成交额作为阈值
    hourly_volume = trades_df.resample(f'{target_hour}H', on='trade_time')['dollar_amount'].sum()
    dollar_threshold = hourly_volume.mean()  # 阈值=目标小时级平均成交额
    
    # 生成Bar
    trades_df['cum_dollar'] = trades_df['dollar_amount'].cumsum()
    trades_df['bar_id'] = (trades_df['cum_dollar'] // dollar_threshold).astype(int)
    
    # 过滤不完整的最后一个Bar
    last_valid_id = trades_df['bar_id'].max() - 1
    trades_df = trades_df[trades_df['bar_id'] <= last_valid_id]
    
    # 提取每个Bar的时间区间和基础统计量
    bar_info = trades_df.groupby('bar_id').agg(
        start_time=('trade_time', 'min'),
        end_time=('trade_time', 'max'),
        total_dollar=('dollar_amount', 'sum'),
        price_open=('price', 'first'),
        price_close=('price', 'last'),
        price_high=('price', 'max'),
        price_low=('price', 'min'),
        trade_count=('price', 'count')
    ).reset_index()
    
    # 计算Bar的未来收益（预测目标：下一个Bar的涨跌幅）
    bar_info['future_return'] = (bar_info['price_close'].shift(-1) - bar_info['price_close']) / bar_info['price_close'] * 100
    
    return trades_df, bar_info, dollar_threshold

def build_dollar_bars(
    trades: pd.DataFrame,
    dollar_threshold: float,
) -> pd.DataFrame:
    """
    生成dollar bars，确保bar_id连续递增。
    
    改进点：
    1. 重构bar_id计算逻辑，通过跟踪累积成交额确保连续
    2. 避免因单笔大额交易导致的bar_id跳跃
    3. 仅过滤最后一个不完整的bar（若存在）
    """
    trades['time'] = _ensure_datetime(trades['time'])
    trades = trades.sort_values('time').reset_index(drop=True)
    df = trades.copy()
    # 处理时间列和排序
    # df['time'] = _ensure_datetime(df['time'])
    # df = df.sort_values('time').reset_index(drop=True)
    
    # 计算成交额（quote_qty）
    if 'quote_qty' not in df.columns or df['quote_qty'].isna().all():
        df['quote_qty'] = df['price'] * df['qty']
    
    # 标记交易方向
    df['trade_sign'] = np.where(df['is_buyer_maker'], -1, 1)
    df['buy_qty'] = df['qty'].where(df['trade_sign'] > 0, 0.0)
    df['sell_qty'] = df['qty'].where(df['trade_sign'] < 0, 0.0)
    
    # 核心改进：逐笔计算bar_id，确保连续递增
    cumulative = 0.0  # 累积成交额
    bar_id = 0        # 当前bar_id
    bar_ids = []      # 存储每个交易的bar_id
    
    for qty in df['quote_qty']:
        cumulative += qty
        # 当累积成交额达到阈值时，当前交易仍属于当前bar_id，随后bar_id递增并重置累积
        if cumulative >= dollar_threshold:
            bar_ids.append(bar_id)
            # 重置累积（保留超额部分，用于下一个bar的计算）
            cumulative -= dollar_threshold
            bar_id += 1
        else:
            bar_ids.append(bar_id)
    
    df['bar_id'] = bar_ids
    
    # 分组聚合
    agg = {
        'time': ['first', 'last'],
        'price': ['first', 'max', 'min', 'last'],
        'qty': 'sum',
        'quote_qty': 'sum',
        'buy_qty': 'sum',
        'sell_qty': 'sum',
    }
    g = df.groupby('bar_id', sort=True).agg(agg)
    
    # 展平列名
    g.columns = [
        'start_time', 'end_time',
        'open', 'high', 'low', 'close',
        'volume', 'dollar_value',
        'buy_volume', 'sell_volume'
    ]
    
    # 仅过滤最后一个可能不完整的bar（若其成交额不足阈值）
    if not g.empty and g.iloc[-1]['dollar_value'] < dollar_threshold:
        g = g.iloc[:-1]
    
    # 重置bar_id为连续整数（避免因过滤最后一个bar导致的断档）
    g = g.reset_index(drop=True)
    g['bar_id'] = g.index
    
    return g
 


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

        left_block  = (idx >= (test_start - embargo_td)) & (idx <  test_start)
        right_block = (idx >  test_end)                  & (idx <= (test_end + embargo_td))
        exclude = left_block | right_block | test_mask   # 再加上测试集本身
        train_idx = idx[~exclude]
        test_idx  = idx[test_mask]
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
    fee_rate: float = 1e-4,
    annualize: bool = True,
    period_seconds: Optional[float] = None,
    seconds_per_year: float = 365.0 * 24.0 * 3600.0,
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

        Xtr.drop(columns=['interval_start', 'interval_end'], inplace=True)
        Xte.drop(columns=['interval_start', 'interval_end'], inplace=True)
        ytr.drop(columns=['end_time'], inplace=True)
        yte.drop(columns=['end_time'], inplace=True)
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

        # 预测与误差指标
        pearson_ic = yhat.corr(yte)
        spearman_ic = yhat.corr(yte, method='spearman')
        rmse = mean_squared_error(yte, yhat) ** 0.5
        dir_acc = (np.sign(yhat) == np.sign(yte)).mean()

        # 简单交易指标（方向持仓，含手续费）
        pos = np.sign(yhat).fillna(0.0)
        ret_gross = (pos * yte).astype(float)
        turnover = pos.diff().abs().fillna(np.abs(pos.iloc[0]))
        ret_net = ret_gross - fee_rate * turnover
        sharpe_net = float(ret_net.mean() / ret_net.std()) if ret_net.std() > 0 else np.nan
        if annualize and pd.notna(sharpe_net):
            ps = float(period_seconds) if (period_seconds is not None and period_seconds > 0) else np.nan
            if np.isfinite(ps) and ps > 0:
                ann_factor = np.sqrt(seconds_per_year / ps)
                sharpe_net_ann = float(sharpe_net * ann_factor)
            else:
                sharpe_net_ann = np.nan
        else:
            sharpe_net_ann = np.nan

        plot_predictions_vs_truth(yhat, yte, save_path = '/Users/aming/project/python/crypto-trade/strategy/fusion/pic/')

        by_fold.append({
            'fold': fold_id,
            'pearson_ic': float(pearson_ic),
            'spearman_ic': float(spearman_ic),
            'rmse': float(rmse),
            'dir_acc': float(dir_acc),
            'ret_gross_mean': float(ret_gross.mean()),
            'ret_net_mean': float(ret_net.mean()),
            'ret_net_std': float(ret_net.std()) if ret_net.std() > 0 else np.nan,
            'sharpe_net': sharpe_net,
            'sharpe_net_ann': sharpe_net_ann,
            'fee_rate': float(fee_rate),
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
        'ret_gross_mean_mean': float(df_folds['ret_gross_mean'].mean()) if 'ret_gross_mean' in df_folds else np.nan,
        'ret_net_mean_mean': float(df_folds['ret_net_mean'].mean()) if 'ret_net_mean' in df_folds else np.nan,
        'sharpe_net_mean': float(df_folds['sharpe_net'].mean()) if 'sharpe_net' in df_folds else np.nan,
        'sharpe_net_ann_mean': float(df_folds['sharpe_net_ann'].mean()) if 'sharpe_net_ann' in df_folds else np.nan,
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
    bar_feat = []
    # bar_feat = aggregate_trade_features_on_bars(trades, bars)

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


def _factor_realized_variance(seg: pd.DataFrame) -> Dict[str, float]:
    if seg.empty:
        return {'rv': np.nan}
    p = seg.sort_values('time')['price'].astype(float).to_numpy()
    if len(p) < 2:
        return {'rv': np.nan}
    r = np.diff(np.log(p))
    rv = float(np.sum(r ** 2)) if len(r) > 0 else np.nan
    return {'rv': rv}


def _factor_bipower_variation(seg: pd.DataFrame) -> Dict[str, float]:
    if seg.empty:
        return {'bpv': np.nan}
    p = seg.sort_values('time')['price'].astype(float).to_numpy()
    if len(p) < 3:
        return {'bpv': np.nan}
    r = np.diff(np.log(p))
    if len(r) < 2:
        return {'bpv': np.nan}
    mu1 = np.sqrt(2.0 / np.pi)
    bpv = float(np.sum(np.abs(r[1:]) * np.abs(r[:-1])) / (mu1 ** 2))
    return {'bpv': bpv}


def _factor_jump_indicator(seg: pd.DataFrame) -> Dict[str, float]:
    if seg.empty:
        return {'jump_rv_bpv': np.nan}
    # 复用前两者逻辑，简单重算一次，便于独立使用
    out_rv = _factor_realized_variance(seg)
    out_bpv = _factor_bipower_variation(seg)
    rv = out_rv.get('rv')
    bpv = out_bpv.get('bpv')
    jump = float(max(rv - bpv, 0.0)) if (pd.notna(rv) and pd.notna(bpv)) else np.nan
    return {'jump_rv_bpv': jump}


def _factor_hf_flip_rate(seg: pd.DataFrame) -> Dict[str, float]:
    if seg.empty:
        return {'hf_flip_rate': np.nan}
    sgn = seg['trade_sign'].astype(int).to_numpy()
    if len(sgn) < 2:
        return {'hf_flip_rate': np.nan}
    flips = int((np.diff(sgn) != 0).sum())
    rate = flips / (len(sgn) - 1)
    return {'hf_flip_rate': float(rate)}


def _factor_mean_reversion_strength(seg: pd.DataFrame) -> Dict[str, float]:
    if seg.empty:
        return {'mr_rho1': np.nan, 'mr_strength': np.nan}
    p = seg.sort_values('time')['price'].astype(float).to_numpy()
    if len(p) < 3:
        return {'mr_rho1': np.nan, 'mr_strength': np.nan}
    r = np.diff(np.log(p))
    if len(r) < 2:
        return {'mr_rho1': np.nan, 'mr_strength': np.nan}
    x = r[:-1] - np.mean(r[:-1])
    y = r[1:] - np.mean(r[1:])
    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if denom == 0:
        return {'mr_rho1': np.nan, 'mr_strength': np.nan}
    rho1 = float(np.sum(x * y) / denom)
    mr_strength = -rho1 if pd.notna(rho1) else np.nan
    return {'mr_rho1': rho1, 'mr_strength': float(mr_strength) if pd.notna(mr_strength) else np.nan}


def _factor_highlow_amplitude_ratio(seg: pd.DataFrame) -> Dict[str, float]:
    if seg.empty:
        return {'hl_amplitude_ratio': np.nan}
    p = seg['price'].astype(float).to_numpy()
    if len(p) == 0:
        return {'hl_amplitude_ratio': np.nan}
    hi = float(np.max(p))
    lo = float(np.min(p))
    mid = (hi + lo) / 2.0
    hl_ratio = float((hi - lo) / mid) if mid != 0 else np.nan
    return {'hl_amplitude_ratio': hl_ratio}


def _factor_cum_signed_flow_price_corr(seg: pd.DataFrame) -> Dict[str, float]:
    """
    累计签名成交 与 价格路径（累积log价差）的相关性（qty口径/dollar口径）。
    """
    if seg.empty:
        return {
            'corr_cumsum_signed_qty_logp': np.nan,
            'corr_cumsum_signed_dollar_logp': np.nan,
        }
    seg = seg.sort_values('time').copy()
    p = seg['price'].astype(float).to_numpy()
    if len(p) < 3:
        return {
            'corr_cumsum_signed_qty_logp': np.nan,
            'corr_cumsum_signed_dollar_logp': np.nan,
        }
    logp = np.log(p)
    logp_rel = logp - logp[0]
    cs_qty = np.cumsum((seg['trade_sign'] * seg['qty']).astype(float).to_numpy())
    cs_dollar = np.cumsum((seg['trade_sign'] * seg['quote_qty']).astype(float).to_numpy())
    def _corr(a, b):
        if len(a) != len(b) or np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        c = np.corrcoef(a, b)[0, 1]
        return float(c) if np.isfinite(c) else np.nan
    return {
        'corr_cumsum_signed_qty_logp': _corr(cs_qty, logp_rel),
        'corr_cumsum_signed_dollar_logp': _corr(cs_dollar, logp_rel),
    }


def _factor_signed_vwap_deviation(seg: pd.DataFrame) -> Dict[str, float]:
    """
    带符号的VWAP偏离：((p_last - vwap)/vwap) * sign(∑signed_qty)
    同时输出未加符号的偏离。
    """
    if seg.empty:
        return {'signed_vwap_deviation': np.nan, 'vwap_deviation': np.nan}
    qty_sum = float(seg['qty'].sum())
    if qty_sum <= 0:
        return {'signed_vwap_deviation': np.nan, 'vwap_deviation': np.nan}
    vwap = float((seg['price'] * seg['qty']).sum() / qty_sum)
    p_last = float(seg.sort_values('time')['price'].iloc[-1])
    dev = (p_last - vwap) / vwap if vwap != 0 else np.nan
    net_signed_qty = float((seg['trade_sign'] * seg['qty']).sum())
    signed_dev = dev * (1.0 if net_signed_qty > 0 else (-1.0 if net_signed_qty < 0 else 0.0)) if pd.notna(dev) else np.nan
    return {'signed_vwap_deviation': signed_dev, 'vwap_deviation': dev}


def _factor_micro_price_momentum(seg: pd.DataFrame, window: int = 20) -> Dict[str, float]:
    """
    微价格动量：短窗Δlog价与价格Z-score（以log价格计算）。
    window 默认20笔，若样本不足则取最大小于等于长度。
    """
    if seg.empty:
        return {'micro_dp_short': np.nan, 'micro_dp_zscore': np.nan}
    seg = seg.sort_values('time').copy()
    p = seg['price'].astype(float).to_numpy()
    if len(p) < 2:
        return {'micro_dp_short': np.nan, 'micro_dp_zscore': np.nan}
    w = int(max(2, min(window, len(p))))
    logp = np.log(p)
    dp_short = float(logp[-1] - logp[-w]) if len(logp) >= w else np.nan
    win = logp[-w:]
    mu = float(np.mean(win))
    sd = float(np.std(win))
    z = (float(logp[-1]) - mu) / sd if sd > 0 else np.nan
    return {'micro_dp_short': dp_short, 'micro_dp_zscore': z}


def _factor_arrival_rate_metrics(seg: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Dict[str, float]:
    """
    到达率与间隔统计：成交笔数、平均到达率、间隔均值/方差/倒数均值。
    """
    duration = max(1.0, (end_ts - start_ts).total_seconds())
    if seg.empty:
        return {
            'arr_trade_count': 0.0,
            'arr_rate_per_sec': 0.0,
            'arr_interval_mean': np.nan,
            'arr_interval_var': np.nan,
            'arr_interval_inv_mean': np.nan,
        }
    t = seg['time'].view('int64') / 1e9
    t = np.sort(t)
    gaps = np.diff(t)
    arr_rate = len(seg) / duration
    interval_mean = float(np.mean(gaps)) if len(gaps) else np.nan
    interval_var = float(np.var(gaps)) if len(gaps) else np.nan
    inv_mean = float(np.mean(1.0 / gaps)) if len(gaps) and np.all(gaps > 0) else np.nan
    return {
        'arr_trade_count': float(len(seg)),
        'arr_rate_per_sec': float(arr_rate),
        'arr_interval_mean': interval_mean,
        'arr_interval_var': interval_var,
        'arr_interval_inv_mean': inv_mean,
    }


def _factor_hawkes_like_clustering(seg: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Dict[str, float]:
    """
    自激励近似：短窗内聚簇度。
    做法：将间隔小于阈值 tau 视为同一簇；输出簇数、平均簇大小、最大簇大小、聚簇度=平均簇大小/总数。
    tau 取区间内间隔中位数或固定阈值最小值。
    """
    duration = max(1.0, (end_ts - start_ts).total_seconds())
    if seg.empty:
        return {
            'hawkes_cluster_count': 0.0,
            'hawkes_cluster_size_mean': np.nan,
            'hawkes_cluster_size_max': 0.0,
            'hawkes_clustering_degree': np.nan,
        }
    t = np.sort(seg['time'].view('int64') / 1e9)
    gaps = np.diff(t)
    if len(gaps) == 0:
        return {
            'hawkes_cluster_count': 1.0,
            'hawkes_cluster_size_mean': float(len(seg)),
            'hawkes_cluster_size_max': float(len(seg)),
            'hawkes_clustering_degree': 1.0,
        }
    tau = float(np.nanmedian(gaps)) if np.isfinite(np.nanmedian(gaps)) else 0.0
    tau = max(tau, 0.001)
    clusters = []
    cur_size = 1
    for g in gaps:
        if g <= tau:
            cur_size += 1
        else:
            clusters.append(cur_size)
            cur_size = 1
    clusters.append(cur_size)
    cluster_count = len(clusters)
    size_mean = float(np.mean(clusters)) if clusters else np.nan
    size_max = float(np.max(clusters)) if clusters else 0.0
    clustering_degree = size_mean / float(len(seg)) if clusters else np.nan
    return {
        'hawkes_cluster_count': float(cluster_count),
        'hawkes_cluster_size_mean': size_mean,
        'hawkes_cluster_size_max': size_max,
        'hawkes_clustering_degree': float(clustering_degree) if pd.notna(clustering_degree) else np.nan,
    }


def _factor_large_trade_tail_share(seg: pd.DataFrame, q: float = 0.9) -> Dict[str, float]:
    """
    大单占比/尾部比例：成交额 > 分位阈值(如90%) 的金额占区间总成交额比例；
    同时输出大单笔数占比与均值（便于诊断）。
    """
    if seg.empty:
        return {
            'large_tail_dollar_share': np.nan,
            'large_tail_trade_share': np.nan,
            'large_tail_dollar_mean': np.nan,
        }
    dv = seg['quote_qty'].astype(float)
    thr = float(np.quantile(dv, q)) if len(dv) > 0 else np.nan
    if not np.isfinite(thr) or thr <= 0:
        return {
            'large_tail_dollar_share': np.nan,
            'large_tail_trade_share': np.nan,
            'large_tail_dollar_mean': np.nan,
        }
    mask = dv >= thr
    share_dollar = float(dv[mask].sum() / dv.sum()) if dv.sum() > 0 else np.nan
    share_trade = float(mask.mean())
    mean_large = float(dv[mask].mean()) if mask.any() else np.nan
    return {
        'large_tail_dollar_share': share_dollar,
        'large_tail_trade_share': share_trade,
        'large_tail_dollar_mean': mean_large,
    }

def _factor_large_trade_tail_directional(seg: pd.DataFrame, q_list: Optional[List[float]] = None) -> Dict[str, float]:
    """
    大单买卖方向分解（慢速）：对每个分位阈值 q，分别统计买/卖大单的金额占比、笔数占比与均额，
    并给出买卖金额占比差值（买-卖）。
    输出键示例：
      - large_buy_tail_dollar_share_q90, large_sell_tail_dollar_share_q90, large_tail_dir_dollar_diff_q90
      - large_buy_tail_trade_share_q95, large_sell_tail_trade_share_q95
      - large_buy_tail_dollar_mean_q90, large_sell_tail_dollar_mean_q90
    """
    q_list = q_list or [0.9, 0.95]
    out: Dict[str, float] = {}
    if seg.empty:
        for q in q_list:
            suffix = f"q{int(round(q*100))}"
            out[f'large_buy_tail_dollar_share_{suffix}'] = np.nan
            out[f'large_sell_tail_dollar_share_{suffix}'] = np.nan
            out[f'large_tail_dir_dollar_diff_{suffix}'] = np.nan
            out[f'large_buy_tail_trade_share_{suffix}'] = np.nan
            out[f'large_sell_tail_trade_share_{suffix}'] = np.nan
            out[f'large_buy_tail_dollar_mean_{suffix}'] = np.nan
            out[f'large_sell_tail_dollar_mean_{suffix}'] = np.nan
        return out

    dv = seg['quote_qty'].astype(float)
    sgn = seg['trade_sign'].astype(int)
    dv_sum = float(dv.sum())

    for q in q_list:
        suffix = f"q{int(round(q*100))}"
        thr = float(np.quantile(dv, q)) if len(dv) > 0 else np.nan
        if not np.isfinite(thr) or thr <= 0 or dv_sum <= 0:
            out[f'large_buy_tail_dollar_share_{suffix}'] = np.nan
            out[f'large_sell_tail_dollar_share_{suffix}'] = np.nan
            out[f'large_tail_dir_dollar_diff_{suffix}'] = np.nan
            out[f'large_buy_tail_trade_share_{suffix}'] = np.nan
            out[f'large_sell_tail_trade_share_{suffix}'] = np.nan
            out[f'large_buy_tail_dollar_mean_{suffix}'] = np.nan
            out[f'large_sell_tail_dollar_mean_{suffix}'] = np.nan
            continue

        mask_large = dv >= thr
        buy_large = mask_large & (sgn > 0)
        sell_large = mask_large & (sgn < 0)

        buy_dollar_share = float(dv[buy_large].sum() / dv_sum) if dv_sum > 0 else np.nan
        sell_dollar_share = float(dv[sell_large].sum() / dv_sum) if dv_sum > 0 else np.nan
        dir_dollar_diff = (buy_dollar_share - sell_dollar_share) if (pd.notna(buy_dollar_share) and pd.notna(sell_dollar_share)) else np.nan

        buy_trade_share = float(buy_large.mean())
        sell_trade_share = float(sell_large.mean())
        buy_mean = float(dv[buy_large].mean()) if buy_large.any() else np.nan
        sell_mean = float(dv[sell_large].mean()) if sell_large.any() else np.nan

        out[f'large_buy_tail_dollar_share_{suffix}'] = buy_dollar_share
        out[f'large_sell_tail_dollar_share_{suffix}'] = sell_dollar_share
        out[f'large_tail_dir_dollar_diff_{suffix}'] = dir_dollar_diff
        out[f'large_buy_tail_trade_share_{suffix}'] = buy_trade_share
        out[f'large_sell_tail_trade_share_{suffix}'] = sell_trade_share
        out[f'large_buy_tail_dollar_mean_{suffix}'] = buy_mean
        out[f'large_sell_tail_dollar_mean_{suffix}'] = sell_mean
    return out

def _factor_price_impact_kyle(seg: pd.DataFrame) -> Dict[str, float]:
    """
    Kyle λ：回归 Δlog(price) 对 签名成交额（quote_qty * sign）。
    以协方差/方差比值近似斜率。
    """
    if len(seg) < 3:
        return {'kyle_lambda': np.nan}
    seg = seg.sort_values('time').copy()
    seg['logp'] = np.log(seg['price'])
    seg['ret'] = seg['logp'].diff()
    seg['signed_dollar'] = seg['quote_qty'].astype(float) * seg['trade_sign'].astype(float)
    x = seg['signed_dollar'].to_numpy()[1:]
    y = seg['ret'].to_numpy()[1:]
    if len(x) < 2 or np.var(x) <= 0:
        return {'kyle_lambda': np.nan}
    covxy = np.cov(x, y, ddof=1)[0, 1]
    lam = covxy / np.var(x)
    return {'kyle_lambda': float(lam) if np.isfinite(lam) else np.nan}


def _factor_price_impact_amihud(seg: pd.DataFrame) -> Dict[str, float]:
    """
    Amihud λ：平均 |Δlog(price)| / 成交额（逐笔近似）。
    """
    if len(seg) < 2:
        return {'amihud_lambda': np.nan}
    seg = seg.sort_values('time').copy()
    seg['logp'] = np.log(seg['price'])
    ret_abs = seg['logp'].diff().abs()
    denom = seg['quote_qty'].replace(0, np.nan).astype(float)
    ratio = (ret_abs / denom).iloc[1:]
    val = ratio.mean()
    return {'amihud_lambda': float(val) if pd.notna(val) else np.nan}


def _factor_price_impact_hasbrouck(seg: pd.DataFrame) -> Dict[str, float]:
    """
    Hasbrouck（简化）：回归 Δlog(price) 对 trade_sign * sqrt(dollar)。
    用协方差/方差近似斜率。
    """
    if len(seg) < 3:
        return {'hasbrouck_lambda': np.nan}
    seg = seg.sort_values('time').copy()
    seg['logp'] = np.log(seg['price'])
    ret = seg['logp'].diff().to_numpy()[1:]
    x = (seg['trade_sign'].astype(float) * np.sqrt(seg['quote_qty'].astype(float))).to_numpy()[1:]
    if len(x) < 2 or np.var(x) <= 0:
        return {'hasbrouck_lambda': np.nan}
    covxy = np.cov(x, ret, ddof=1)[0, 1]
    lam = covxy / np.var(x)
    return {'hasbrouck_lambda': float(lam) if np.isfinite(lam) else np.nan}


def _factor_price_impact_halflife(seg: pd.DataFrame) -> Dict[str, float]:
    """
    冲击半衰期（近似）：基于区间内 Δlog(price) 的一阶自相关 ρ，
    t_half = ln(2) / -ln(ρ)，仅在 0<ρ<1 时定义。
    """
    if len(seg) < 3:
        return {'impact_half_life': np.nan}
    seg = seg.sort_values('time').copy()
    r = seg['price'].astype(float)
    r = np.log(r).diff().dropna().to_numpy()
    if len(r) < 2:
        return {'impact_half_life': np.nan}
    r0 = r[:-1] - np.mean(r[:-1])
    r1 = r[1:] - np.mean(r[1:])
    denom = np.sqrt(np.sum(r0**2) * np.sum(r1**2))
    if denom == 0:
        return {'impact_half_life': np.nan}
    rho = float(np.sum(r0 * r1) / denom)
    if not (0 < rho < 1):
        return {'impact_half_life': np.nan}
    t_half = np.log(2.0) / (-np.log(rho))
    return {'impact_half_life': float(t_half)}


def _factor_price_impact_decomposition(seg: pd.DataFrame) -> Dict[str, float]:
    """
    冲击占比（简化）：
    permanent_share ≈ |p_end - p_start| / (∑|Δp| + eps)
    transient_share = 1 - permanent_share
    """
    if len(seg) < 2:
        return {'impact_perm_share': np.nan, 'impact_transient_share': np.nan}
    p = seg.sort_values('time')['price'].astype(float).to_numpy()
    dp = np.diff(p)
    denom = float(np.sum(np.abs(dp)))
    if denom <= 0:
        return {'impact_perm_share': np.nan, 'impact_transient_share': np.nan}
    perm = float(np.abs(p[-1] - p[0]) / denom)
    perm = float(np.clip(perm, 0.0, 1.0))
    return {'impact_perm_share': perm, 'impact_transient_share': float(1.0 - perm)}


def _compute_interval_trade_features(trades: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Dict[str, float]:
    """
    在给定时间区间 [start_ts, end_ts) 内，逐笔计算区间特征。
    每个特征由独立函数实现并组合返回。
    """
    df = trades.copy()
    # if not pd.api.types.is_datetime64_any_dtype(df['time']):
    #     df['time'] = pd.to_datetime(df['time'])
    df['time'] = _ensure_datetime(df['time'])
    df = df.sort_values('time')
    if 'quote_qty' not in df.columns or df['quote_qty'].isna().all():
        df['quote_qty'] = df['price'] * df['qty']
    df['trade_sign'] = np.where(df['is_buyer_maker'], -1, 1)
    seg = df.loc[(df['time'] >= start_ts) & (df['time'] < end_ts)].copy()

    out = {}
    # 基础交易统计（可按需启用）
    out.update(_factor_int_trade_stats(seg, start_ts, end_ts))
    # 订单流与旺盛度（如需一并启用可取消注释）
    out.update(_factor_order_flow_imbalance(seg))
    out.update(_factor_garman_order_flow(seg))
    out.update(_factor_rolling_ofi(seg))
    out.update(_factor_activity_and_persistence(seg))
    # # 价格冲击/流动性代理
    out.update(_factor_price_impact_kyle(seg))
    out.update(_factor_price_impact_amihud(seg))
    out.update(_factor_price_impact_hasbrouck(seg))
    out.update(_factor_price_impact_halflife(seg))
    out.update(_factor_price_impact_decomposition(seg))
    # 波动与噪声（拆分为独立函数）
    out.update(_factor_realized_variance(seg))
    out.update(_factor_bipower_variation(seg))
    out.update(_factor_jump_indicator(seg))
    out.update(_factor_hf_flip_rate(seg))
    out.update(_factor_mean_reversion_strength(seg))
    out.update(_factor_highlow_amplitude_ratio(seg))
    # # 成交节奏与聚簇
    out.update(_factor_arrival_rate_metrics(seg, start_ts, end_ts))
    out.update(_factor_hawkes_like_clustering(seg, start_ts, end_ts))
    out.update(_factor_large_trade_tail_share(seg))
    # 大单买卖方向分解（慢速）
    out.update(_factor_large_trade_tail_directional(seg, q_list=[0.9, 0.95]))
    #     # 价格路径形状
    out.update(_factor_cum_signed_flow_price_corr(seg))
    out.update(_factor_signed_vwap_deviation(seg))
    out.update(_factor_micro_price_momentum(seg))
    return out


# ========= 高性能实现：一次预处理 + 前缀和O(1)聚合 =========
class TradesContext:
    def __init__(self, t_ns: np.ndarray, price: np.ndarray, qty: np.ndarray, quote: np.ndarray, sign: np.ndarray):
        self.t_ns = t_ns  # int64 ns 时间戳（已排序）
        self.price = price.astype(np.float64)
        self.qty = qty.astype(np.float64)
        self.quote = quote.astype(np.float64)
        self.sign = sign.astype(np.float64)

        # 衍生量
        self.logp = np.log(self.price)
        self.ret = np.diff(self.logp)
        self.ret2 = np.r_[0.0, self.ret ** 2]
        # |r_t||r_{t-1}| 对齐成与 price 同长（首位补0）
        abs_r = np.abs(self.ret)
        bp_core = np.r_[0.0, np.r_[0.0, abs_r[1:] * abs_r[:-1]]]  # 与 price 对齐

        # 前缀和（与 price 同长）
        self.csum_qty = np.cumsum(self.qty)
        self.csum_quote = np.cumsum(self.quote)
        self.csum_signed_qty = np.cumsum(self.sign * self.qty)
        self.csum_signed_quote = np.cumsum(self.sign * self.quote)
        self.csum_pxqty = np.cumsum(self.price * self.qty)
        self.csum_ret2 = np.cumsum(self.ret2)
        self.csum_bpv = np.cumsum(bp_core)

    def locate(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Tuple[int, int]:
        s = np.searchsorted(self.t_ns, int(np.int64(np.datetime64(start_ts, 'ns'))), side='left')
        e = np.searchsorted(self.t_ns, int(np.int64(np.datetime64(end_ts, 'ns'))), side='left')
        return s, e


def _build_trades_context(trades: pd.DataFrame) -> TradesContext:
    df = trades
    t = _ensure_datetime(df['time']).values.astype('datetime64[ns]').astype('int64')
    order = np.argsort(t)
    t_ns = t[order]
    price = df['price'].to_numpy(dtype=float)[order]
    qty = df['qty'].to_numpy(dtype=float)[order]
    quote = (df['quote_qty'] if 'quote_qty' in df.columns else df['price'] * df['qty']).to_numpy(dtype=float)[order]
    sign = np.where(df['is_buyer_maker'].to_numpy()[order], -1.0, 1.0)
    return TradesContext(t_ns, price, qty, quote, sign)


def _sum_range(prefix: np.ndarray, s: int, e: int) -> float:
    if e <= s:
        return 0.0
    return float(prefix[e - 1] - (prefix[s - 1] if s > 0 else 0.0))


def _default_features_config() -> Dict[str, bool]:
    return {
        'base': True,                   # 基础汇总/VWAP/强度/买量占比
        'order_flow': True,             # GOF/签名不平衡
        'price_impact': True,           # Kyle/Amihud/Hasbrouck/半衰期/占比
        'volatility_noise': True,       # RV/BPV/Jump/微动量/均值回复/高低幅比
        'arrival_stats': True,          # 到达间隔统计
        'run_markov': True,             # run-length/Markov/翻转率
        'rolling_ofi': True,            # 滚动OFI
        'hawkes': True,                 # 聚簇（Hawkes近似）
        'path_shape': True,             # 协动相关/VWAP偏离
        'tail': True,                   # 大单尾部比例
        'tail_directional': True,       # 大单买卖方向性（分位阈值法）
    }


def _compute_interval_trade_features_fast(
    ctx: TradesContext,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    features_config: Optional[Dict[str, bool]] = None,
    tail_q: float = 0.9,
    tail_q_list: Optional[List[float]] = None,
) -> Dict[str, float]:
    s, e = ctx.locate(start_ts, end_ts)
    if e - s <= 0:
        return {}

    cfg = _default_features_config()
    if features_config:
        cfg.update(features_config)

    # 基础聚合
    sum_qty = _sum_range(ctx.csum_qty, s, e)
    sum_quote = _sum_range(ctx.csum_quote, s, e)
    sum_signed_qty = _sum_range(ctx.csum_signed_qty, s, e)
    sum_signed_quote = _sum_range(ctx.csum_signed_quote, s, e)
    sum_pxqty = _sum_range(ctx.csum_pxqty, s, e)

    vwap = sum_pxqty / sum_qty if sum_qty > 0 else np.nan
    p_last = ctx.price[e - 1]
    duration = max(1.0, (end_ts - start_ts).total_seconds())
    intensity = (e - s) / duration

    # RV/BPV/Jump（ret2、bpv 与 price 对齐）
    rv = _sum_range(ctx.csum_ret2, s, e)
    bpv = _sum_range(ctx.csum_bpv, s, e)
    jump = max(rv - bpv, 0.0) if (np.isfinite(rv) and np.isfinite(bpv)) else np.nan

    # Garman OF (count/volume)
    n = float(e - s)
    # count 口径用 sign 的平均（与逐笔实现等价）
    gof_by_count = float(np.mean(np.sign(ctx.sign[s:e]))) if n > 0 else np.nan
    gof_by_volume = (sum_signed_qty / sum_qty) if sum_qty > 0 else np.nan

    # VWAP偏离（带符号）
    dev = (p_last - vwap) / vwap if vwap != 0 and np.isfinite(vwap) else np.nan
    signed_dev = dev * (1.0 if sum_signed_qty > 0 else (-1.0 if sum_signed_qty < 0 else 0.0)) if pd.notna(dev) else np.nan

    # 买量占比
    buy_mask = ctx.sign[s:e] > 0
    buy_qty = float(ctx.qty[s:e][buy_mask].sum()) if (e - s) > 0 else 0.0
    trade_buy_ratio = (buy_qty / sum_qty) if sum_qty > 0 else np.nan

    # 微动量（短窗，用末尾W笔）
    W = min(20, e - s)
    if W >= 2:
        lp = ctx.logp[max(s, e - W):e]
        dp_short = float(lp[-1] - lp[0])
        mu = float(np.mean(lp))
        sd = float(np.std(lp))
        z = (float(lp[-1]) - mu) / sd if sd > 0 else np.nan
    else:
        dp_short = np.nan
        z = np.nan

    # 到达间隔统计
    t_slice = ctx.t_ns[s:e].astype(np.float64) / 1e9
    if cfg['arrival_stats'] and t_slice.size >= 2:
        gaps = np.diff(t_slice)
        arr_interval_mean = float(np.mean(gaps))
        arr_interval_var = float(np.var(gaps))
        arr_interval_inv_mean = float(np.mean(1.0 / gaps)) if np.all(gaps > 0) else np.nan
    else:
        arr_interval_mean = np.nan
        arr_interval_var = np.nan
        arr_interval_inv_mean = np.nan

    # 均值回复强度（lag-1 自相关）与高低幅度占比
    r_slice = np.diff(ctx.logp[s:e])
    if cfg['volatility_noise'] and r_slice.size >= 2:
        x0 = r_slice[:-1] - np.mean(r_slice[:-1])
        x1 = r_slice[1:] - np.mean(r_slice[1:])
        denom = np.sqrt(np.sum(x0**2) * np.sum(x1**2))
        mr_rho1 = float(np.sum(x0 * x1) / denom) if denom > 0 else np.nan
        mr_strength = -mr_rho1 if pd.notna(mr_rho1) else np.nan
    else:
        mr_rho1 = np.nan
        mr_strength = np.nan

    if cfg['volatility_noise'] and (e - s) > 0:
        hi = float(np.max(ctx.price[s:e]))
        lo = float(np.min(ctx.price[s:e]))
        mid = (hi + lo) / 2.0
        hl_amplitude_ratio = float((hi - lo) / mid) if mid != 0 else np.nan
    else:
        hl_amplitude_ratio = np.nan

    # 价格冲击代理（Kyle/Amihud/Hasbrouck/半衰期/占比）
    out_impact = _fast_price_impact_metrics(ctx, s, e) if cfg['price_impact'] else {}
    
    # 价格路径形状：协动相关性
    out_path_corr = _fast_cum_signed_flow_price_corr(ctx, s, e) if cfg['path_shape'] else {}

    # run-length / Markov / 翻转率
    out_run = _fast_run_length_metrics(ctx, s, e) if cfg['run_markov'] else {}
    out_markov = _fast_markov_persistence(ctx, s, e) if cfg['run_markov'] else {}

    # 滚动OFI（区间内）
    out_ofi = _fast_rolling_ofi_stats(ctx, s, e, window=min(20, e - s)) if cfg['rolling_ofi'] else {}

    # Hawkes 近似聚簇
    out_hawkes = _fast_hawkes_clustering(ctx, s, e) if cfg['hawkes'] else {}

    base = {}
    if cfg['base']:
        base.update({
            'int_trade_vwap': vwap,
            'int_trade_volume_sum': sum_qty,
            'int_trade_dollar_sum': sum_quote,
            'int_trade_signed_volume': sum_signed_qty,
            'int_trade_buy_ratio': trade_buy_ratio,
            'int_trade_intensity': intensity,
            'int_trade_rv': rv,
        })
    if cfg['order_flow']:
        base.update({
            'ofi_signed_qty_sum': sum_signed_qty,
            'ofi_signed_quote_sum': sum_signed_quote,
            'gof_by_count': gof_by_count,
            'gof_by_volume': gof_by_volume,
        })
    if cfg['volatility_noise']:
        base.update({
            'rv': rv,
            'bpv': bpv,
            'jump_rv_bpv': jump,
            'micro_dp_short': dp_short,
            'micro_dp_zscore': z,
            'mr_rho1': mr_rho1,
            'mr_strength': mr_strength,
            'hl_amplitude_ratio': hl_amplitude_ratio,
        })
    if cfg['path_shape']:
        base.update({
            'signed_vwap_deviation': signed_dev,
            'vwap_deviation': dev,
        })

    # 大单尾部比例
    out_tail = {}
    if cfg['tail'] and (e - s) > 0:
        dv = ctx.quote[s:e]
        if dv.size > 0:
            thr = float(np.quantile(dv, tail_q))
            if np.isfinite(thr) and thr > 0:
                mask_tail = dv >= thr
                share_dollar = float(dv[mask_tail].sum() / dv.sum()) if dv.sum() > 0 else np.nan
                share_trade = float(mask_tail.mean())
                mean_large = float(dv[mask_tail].mean()) if mask_tail.any() else np.nan
                out_tail = {
                    'large_tail_dollar_share': share_dollar,
                    'large_tail_trade_share': share_trade,
                    'large_tail_dollar_mean': mean_large,
                }
    # 大单买卖方向性（分位阈值列表）
    out_tail_dir = {}
    if cfg.get('tail_directional', False) and (e - s) > 0:
        q_list = tail_q_list or [0.9, 0.95]
        out_tail_dir = _fast_large_trade_tail_directional(ctx, s, e, q_list)
    out = {}
    out.update(base)
    out.update(out_impact)
    out.update(out_path_corr)
    out.update(out_run)
    out.update(out_markov)
    out.update(out_ofi)
    out.update(out_hawkes)
    out.update(out_tail)
    out.update(out_tail_dir)
    return out


def _fast_price_impact_metrics(ctx: TradesContext, s: int, e: int) -> Dict[str, float]:
    """
    价格冲击与流动性代理（fast）：
    - Kyle λ：Δlogp ~ signed_dollar（协方差/方差比）
    - Amihud λ：mean(|Δlogp| / dollar)
    - Hasbrouck（简化）：Δlogp ~ sign(Δlogp)*sqrt(dollar)
    - impact_half_life：Δlogp 的 lag-1 自相关半衰期
    - impact_perm/transient_share：|p_end - p_start| / ∑|Δp|
    返回空dict表示样本不足。
    """
    if e - s < 3:
        return {}
    r = np.diff(ctx.logp[s:e])                # 长度 K-1
    sdollar = ctx.sign[s:e] * ctx.quote[s:e]  # 长度 K
    x = sdollar[1:]                           # 与 r 对齐
    y = r
    varx = float(np.var(x))
    kyle = float(np.cov(x, y, ddof=1)[0, 1] / varx) if varx > 0 else np.nan

    amihud = float((np.abs(r) / (ctx.quote[s+1:e])).mean()) if np.all(ctx.quote[s+1:e] > 0) else np.nan

    xh = np.sign(r) * np.sqrt(ctx.quote[s+1:e])
    varxh = float(np.var(xh))
    hasb = float(np.cov(xh, r, ddof=1)[0, 1] / varxh) if varxh > 0 and len(r) > 1 else np.nan

    # 半衰期
    r0 = r[:-1] - np.mean(r[:-1])
    r1 = r[1:] - np.mean(r[1:])
    denom = np.sqrt(np.sum(r0**2) * np.sum(r1**2))
    if denom > 0:
        rho = float(np.sum(r0 * r1) / denom)
        t_half = float(np.log(2.0) / (-np.log(rho))) if (0 < rho < 1) else np.nan
    else:
        t_half = np.nan

    # 冲击占比
    dp = np.diff(ctx.price[s:e])
    denom2 = float(np.sum(np.abs(dp)))
    if denom2 > 0:
        perm = float(np.abs(ctx.price[e-1] - ctx.price[s]) / denom2)
        perm = float(np.clip(perm, 0.0, 1.0))
        trans = float(1.0 - perm)
    else:
        perm = np.nan
        trans = np.nan

    return {
        'kyle_lambda': kyle,
        'amihud_lambda': amihud,
        'hasbrouck_lambda': hasb,
        'impact_half_life': t_half,
        'impact_perm_share': perm,
        'impact_transient_share': trans,
    }


def _fast_run_length_metrics(ctx: TradesContext, s: int, e: int) -> Dict[str, float]:
    if e - s <= 0:
        return {'runlen_buy_max': 0.0, 'runlen_sell_max': 0.0, 'runlen_buy_mean': np.nan, 'runlen_sell_mean': np.nan}
    sgn = np.sign(ctx.sign[s:e]).astype(np.int8)
    if sgn.size == 0:
        return {'runlen_buy_max': 0.0, 'runlen_sell_max': 0.0, 'runlen_buy_mean': np.nan, 'runlen_sell_mean': np.nan}
    runs = []
    cur = sgn[0]
    length = 1
    for val in sgn[1:]:
        if val == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur = val
            length = 1
    runs.append((cur, length))
    buy_runs = [l for s_, l in runs if s_ > 0]
    sell_runs = [l for s_, l in runs if s_ < 0]
    return {
        'runlen_buy_max': float(max(buy_runs)) if buy_runs else 0.0,
        'runlen_sell_max': float(max(sell_runs)) if sell_runs else 0.0,
        'runlen_buy_mean': float(np.mean(buy_runs)) if buy_runs else np.nan,
        'runlen_sell_mean': float(np.mean(sell_runs)) if sell_runs else np.nan,
    }


def _fast_markov_persistence(ctx: TradesContext, s: int, e: int) -> Dict[str, float]:
    if e - s <= 1:
        return {'alt_frequency': np.nan, 'p_pos_to_pos': np.nan, 'p_neg_to_neg': np.nan, 'hf_flip_rate': np.nan}
    sgn = np.sign(ctx.sign[s:e]).astype(np.int8)
    flips = int(np.count_nonzero(np.diff(sgn)))
    alt_frequency = flips / (len(sgn) - 1)
    from_pos = sgn[:-1] > 0
    to_pos = sgn[1:] > 0
    from_neg = sgn[:-1] < 0
    to_neg = sgn[1:] < 0
    pos_count = int(from_pos.sum())
    neg_count = int(from_neg.sum())
    p_pos_to_pos = float((from_pos & to_pos).sum() / pos_count) if pos_count > 0 else np.nan
    p_neg_to_neg = float((from_neg & to_neg).sum() / neg_count) if neg_count > 0 else np.nan
    return {'alt_frequency': float(alt_frequency), 'p_pos_to_pos': p_pos_to_pos, 'p_neg_to_neg': p_neg_to_neg, 'hf_flip_rate': float(alt_frequency)}


def _fast_rolling_ofi_stats(ctx: TradesContext, s: int, e: int, window: int = 20) -> Dict[str, float]:
    if e - s <= 0 or window <= 1:
        return {'ofi_roll_sum_max': 0.0, 'ofi_roll_sum_std': 0.0}
    arr = (ctx.sign[s:e] * ctx.qty[s:e]).astype(np.float64)
    if arr.size < window:
        return {'ofi_roll_sum_max': float(arr.sum()), 'ofi_roll_sum_std': 0.0}
    csum = np.cumsum(arr)
    roll = csum[window - 1:] - np.r_[0.0, csum[:-window]]
    return {'ofi_roll_sum_max': float(np.max(roll)), 'ofi_roll_sum_std': float(np.std(roll))}


def _fast_hawkes_clustering(ctx: TradesContext, s: int, e: int) -> Dict[str, float]:
    if e - s <= 0:
        return {
            'hawkes_cluster_count': 0.0,
            'hawkes_cluster_size_mean': np.nan,
            'hawkes_cluster_size_max': 0.0,
            'hawkes_clustering_degree': np.nan,
        }
    t = ctx.t_ns[s:e].astype(np.float64) / 1e9
    if t.size <= 1:
        return {
            'hawkes_cluster_count': 1.0,
            'hawkes_cluster_size_mean': float(t.size),
            'hawkes_cluster_size_max': float(t.size),
            'hawkes_clustering_degree': 1.0,
        }
    gaps = np.diff(t)
    tau = float(np.nanmedian(gaps)) if np.isfinite(np.nanmedian(gaps)) else 0.0
    tau = max(tau, 0.001)
    clusters = []
    cur = 1
    for g in gaps:
        if g <= tau:
            cur += 1
        else:
            clusters.append(cur)
            cur = 1
    clusters.append(cur)
    size_mean = float(np.mean(clusters)) if clusters else np.nan
    size_max = float(np.max(clusters)) if clusters else 0.0
    degree = size_mean / float(e - s) if clusters else np.nan
    return {
        'hawkes_cluster_count': float(len(clusters)),
        'hawkes_cluster_size_mean': size_mean,
        'hawkes_cluster_size_max': size_max,
        'hawkes_clustering_degree': float(degree) if pd.notna(degree) else np.nan,
    }


def _fast_cum_signed_flow_price_corr(ctx: TradesContext, s: int, e: int) -> Dict[str, float]:
    if e - s <= 2:
        return {'corr_cumsum_signed_qty_logp': np.nan, 'corr_cumsum_signed_dollar_logp': np.nan}
    # 累计相对序列
    logp_rel = ctx.logp[s:e] - ctx.logp[s]
    cs_signed_qty = (ctx.csum_signed_qty[s:e] - (ctx.csum_signed_qty[s] if s < ctx.csum_signed_qty.size else 0.0))
    cs_signed_dollar = (ctx.csum_signed_quote[s:e] - (ctx.csum_signed_quote[s] if s < ctx.csum_signed_quote.size else 0.0))
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size != b.size or a.size < 3:
            return np.nan
        sa = np.std(a)
        sb = np.std(b)
        if sa == 0 or sb == 0:
            return np.nan
        c = np.corrcoef(a, b)[0, 1]
        return float(c) if np.isfinite(c) else np.nan
    return {
        'corr_cumsum_signed_qty_logp': _corr(cs_signed_qty, logp_rel),
        'corr_cumsum_signed_dollar_logp': _corr(cs_signed_dollar, logp_rel),
    }


def make_interval_feature_dataset(
    trades: pd.DataFrame,
    dollar_threshold: float,
    feature_window_bars: int = 10,
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

    # 预构建高性能上下文（一次即可）
    ctx = _build_trades_context(trades)

    # 计算每个样本的区间
    features = []
    idx = 1
    for bar_id in close_s.index:
        if window_mode == 'past':
            start_idx = bar_id - feature_window_bars
            end_idx = bar_id - 1
            if start_idx < 0:
                features.append({'bar_id': bar_id, '_skip': True})
                continue
        else:  # future 区间（注意信息泄露，仅在需要时使用）
            start_idx = bar_id
            end_idx = bar_id + feature_window_bars - 1
            if end_idx >= len(bars):
                features.append({'bar_id': bar_id, '_skip': True})
                continue
                
        start_ts = bars.loc[start_idx, 'start_time']
        end_ts = bars.loc[end_idx, 'end_time']
        feat = _compute_interval_trade_features_fast(ctx, start_ts, end_ts)
        print(idx)
        idx = idx + 1
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
    feature_window_bars: int = 10,
    horizon_bars: int = 3,
    window_mode: str = 'past',
    n_splits: int = 5,
    embargo_bars: Optional[int] = None,
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
        feature_window_bars=feature_window_bars,
        horizon_bars=horizon_bars,
        window_mode=window_mode,
    )
    
    mask = y.notna() & np.isfinite(y.values)
    X = X.loc[mask].replace([np.inf, -np.inf], np.nan)
    y = y.loc[X.index]

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
    # 自动放大 embargo：至少覆盖 feature_window_bars 的时间长度；若用户给了 embargo_bars，则取两者较大
    auto_embargo_td = median_duration * int(max(1, feature_window_bars))
    if embargo_bars is not None:
        user_embargo_td = median_duration * int(max(0, embargo_bars))
        embargo_td = max(auto_embargo_td, user_embargo_td)
    else:
        embargo_td = auto_embargo_td

    # 传入期长：使用 dollar bar 的中位秒数，便于年化换算
    # 单期收益对应的是 horizon_bars 根 bar 的窗口
    period_seconds = (
        float(median_duration.total_seconds() * max(1, horizon_bars))
        if median_duration is not None
        else None
    )
    eval_result = purged_cv_evaluate(
        X=X2,
        y=y2,
        n_splits=n_splits,
        embargo=embargo_td,
        model_type=model_type,
        random_state=random_state,
        period_seconds=period_seconds,
    )

    return {'eval': eval_result, 'X': X, 'y': y, 'bars': bars}

 


def plot_predictions_vs_truth(
    preds: pd.Series,
    y: pd.Series,
    title: str = 'Pred vs True',
    save_path: Optional[str] = None,
) -> None:
    """
    画预测值与真实值曲线，并标注基于 sign(pred) 的交易变更时间点（纵线+箭头）。
    preds: 索引为时间（DatetimeIndex）的预测序列
    y:     同索引的真实标签序列（若不一致会自动对齐）
    """
    if preds is None or len(preds) == 0:
        return
    # 对齐索引
    idx = preds.dropna().index.intersection(y.dropna().index)
    if len(idx) == 0:
        return
    y_plot = y.loc[idx].astype(float)
    p_plot = preds.loc[idx].astype(float)

    # 基于预测生成持仓与换手点
    pos = np.sign(p_plot).fillna(0.0)
    change = pos.diff().fillna(pos.iloc[0])
    turnover = change.abs()
    trade_times = turnover[turnover > 0].index
    long_entries = change[change > 0].index
    short_entries = change[change < 0].index

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_plot.index, y_plot.values, label='y_true', color='#1f77b4', alpha=0.8)
    ax.plot(p_plot.index, p_plot.values, label='y_pred', color='#ff7f0e', alpha=0.8)

    # 标注交易变更时间点
    for t in trade_times:
        ax.axvline(t, color='gray', alpha=0.15, linewidth=1)
    ax.scatter(long_entries, np.zeros(len(long_entries)), marker='^', color='green', label='enter long', zorder=3)
    ax.scatter(short_entries, np.zeros(len(short_entries)), marker='v', color='red', label='enter short', zorder=3)

    ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.3)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

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
    date_list = generate_date_range('2025-01-01', '2025-01-01')
    for date in date_list:
        raw_df.append(pd.read_csv(f'/Volumes/Ext-Disk/data/futures/um/daily/trades/ETHUSDT/ETHUSDT-trades-{date}.zip'))
        
    # dollar_bar = build_dollar_bars(raw_df, 10000 * 2000)
    # print(dollar_bar)

    trades_df = pd.concat(raw_df, ignore_index=True)
    res = run_bar_interval_pipeline(
        trades=trades_df,
        dollar_threshold=10000 * 6000,
        feature_window_bars=10,
        horizon_bars=3,
        window_mode='past',
        n_splits=5,
        embargo_bars=None,
        model_type='ridge',
    )
    print(res.get('eval', {}).get('summary'))

if __name__ == '__main__':
    main()



