import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import os

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
    bar_trade_counts={}

    for qty in df['quote_qty']:
        cumulative += qty
        bar_trade_counts[bar_id] = bar_trade_counts.get(bar_id, 0) + 1
        # 当累积成交额达到阈值时，当前交易仍属于当前bar_id，随后bar_id递增并重置累积
        if cumulative >= dollar_threshold:
            bar_ids.append(bar_id)
            # 重置累积（保留超额部分，用于下一个bar的计算）
            cumulative -= dollar_threshold
            bar_id += 1
        else:
            bar_ids.append(bar_id)
    
    
    
    df['bar_id'] = bar_ids
    df['trades'] = df['bar_id'].map(bar_trade_counts)  # 关键修复：通过map对齐

    # 分组聚合
    agg = {
        'time': ['first', 'last'],
        'price': ['first', 'max', 'min', 'last'],
        'qty': 'sum',
        'quote_qty': 'sum',
        'buy_qty': 'sum',
        'sell_qty': 'sum',
        'trades': 'first' 
    }
    g = df.groupby('bar_id', sort=True).agg(agg)
    
    # 展平列名
    g.columns = [
        'start_time', 'end_time',
        'open', 'high', 'low', 'close',
        'volume', 'dollar_value',
        'buy_volume', 'sell_volume','trades'
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

        # 安全删除可能存在的时间列
        drop_cols = [c for c in ['feature_start', 'feature_end', 'prediction_time'] if c in Xtr.columns]
        if drop_cols:
            Xtr.drop(columns=drop_cols, inplace=True)
        drop_cols_te = [c for c in ['feature_start', 'feature_end', 'prediction_time'] if c in Xte.columns]
        if drop_cols_te:
            Xte.drop(columns=drop_cols_te, inplace=True)
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
        e = np.searchsorted(self.t_ns, int(np.int64(np.datetime64(end_ts, 'ns'))), side='right') - 1
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
    # 计算Kyle λ：价格冲击系数，衡量订单流对价格的影响程度
    # 公式：Kyleλ = cov(Δlogp, signed_dollar) / var(signed_dollar)
    # signed_dollar（带符号的交易金额）：反映订单流方向和规模（正数表示买入，负数表示卖出，绝对值为交易金额）。
    # Δlogp（对数价格变化）：反映价格的变动幅度（用对数差分可近似收益率，避免价格绝对值影响）。
    # 这个公式的本质是：通过计算 “订单流（signed_dollar）与价格变化（Δlogp）的相关性”，除以 “订单流自身的波动”，得到 “单位订单流对价格的边际影响”
    sdollar = ctx.sign[s:e] * ctx.quote[s:e]
    x = sdollar[1:] 
    # y = r
    varx = float(np.var(x))
    kyle = float(np.cov(x, r, ddof=1)[0, 1] / varx) if varx > 0 else np.nan
    
    # 计算Amihud λ：非流动性指标，值越大表示流动性越差
    # 公式：Amihudλ = mean(|Δlogp| / 交易金额)
    # 确保交易金额都为正数（避免除零错误）
    # 当 Amihud 值越大，说明相同规模的交易（如 100 万元的买入或卖出）会引发更大的价格波动，意味着市场 “吸收交易的能力弱”，流动性更差（非流动性更高）。
    amihud = float((np.abs(r) / (ctx.quote[s+1:e])).mean()) if np.all(ctx.quote[s+1:e] > 0) else np.nan
    # 计算Hasbrouck λ：价格冲击系数，衡量订单流对价格的影响程度
    # Hasbrouck 的研究核心是将价格冲击分解为 “永久冲击”（由信息驱动，影响长期价格）和 “暂时冲击”（由流动性需求驱动，短期会逆转）。简化版的 Hasbrouck λ 更侧重整体冲击强度，其设计有两个关键考虑：
    # 非线性关系：大额交易的边际价格冲击通常会递减（比如 1000 万交易的冲击可能远小于 10 个 100 万交易的总和），用平方根转换能更好地拟合这种特征。
    # 信息含量：该指标更敏感于交易中的 “信息成分”—— 如果一笔交易包含新信息（如基本面变化），Hasbrouck λ 会更显著地反映其对价格的影响。
    # 与 Kyleλ：Kyleλ 用原始交易金额衡量线性冲击，而 Hasbrouck λ 用平方根捕捉非线性冲击，更贴近实际市场中 “规模越大、边际冲击越弱” 的特征。
    # 与 Amihud：Amihud 是 “价格波动绝对值 / 交易金额” 的均值，侧重 “平均冲击成本”；而 Hasbrouck λ 通过协方差计算，更侧重 “系统性的冲击敏感度”，且包含了价格变动方向的信息。
    # 通过平方根转换捕捉交易规模与价格冲击之间的非线性关系（实证中发现，价格冲击随交易规模增长的速度往往慢于线性关系，用平方根更贴合实际）。 
    xh = np.sign(r) * np.sqrt(ctx.quote[s+1:e])
    varxh = float(np.var(xh))
    hasb = float(np.cov(xh, r, ddof=1)[0, 1] / varxh) if varxh > 0 and len(r) > 1 else np.nan

    # 半衰期
    # 这段代码用于计算价格冲击的半衰期（impact half-life），核心是通过分析价格冲击序列的一阶自相关性，衡量价格冲击的 “持续性”—— 即价格受到冲击后，需要多久才能衰减到初始强度的一半。以下是逐部分的详细解释：
    r0 = r[:-1] - np.mean(r[:-1])
    r1 = r[1:] - np.mean(r[1:])
    denom = np.sqrt(np.sum(r0**2) * np.sum(r1**2))
    if denom > 0:
        rho = float(np.sum(r0 * r1) / denom)
        t_half = float(np.log(2.0) / (-np.log(rho))) if (0 < rho < 1) else np.nan
    else:
        t_half = np.nan

    # 冲击占比
    # 这段代码的核心功能是计算价格冲击的 “永久成分” 与 “暂时成分” 占比—— 通过分析一段区间内的价格变动，区分 “长期留存的价格变化（永久冲击）” 和 “短期波动后逆转的价格变化（暂时冲击）”，是市场微观结构中判断价格变动驱动因素（信息 vs 流动性）的关键逻辑。
    dp = np.diff(ctx.price[s:e])
    # 总价格波动的绝对值之和
    denom2 = float(np.sum(np.abs(dp)))
    if denom2 > 0:
        # 计算永久冲击占比
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


def _fast_large_trade_tail_directional(
    ctx: TradesContext,
    s: int,
    e: int,
    q_list: List[float],
) -> Dict[str, float]:
    """
    方向性大单指标（按成交额分位阈值）：
      - 对每个 q ∈ q_list，取阈值 thr_q = quantile(quote, q)
      - 统计：大单买/卖成交额与笔数、占比、方向不平衡（LTI）、单位时间净额
    输出：包含每个 q 的后缀（如 _q95），并提供一个主口径（取最大 q）。
    """
    if e - s <= 0:
        return {}
    dv = ctx.quote[s:e]
    if dv.size == 0:
        return {}
    sign = ctx.sign[s:e]
    eps = 1e-12
    total_dollar = float(dv.sum()) if np.isfinite(dv.sum()) else 0.0
    # 区间时长（秒）
    duration_seconds = max(1.0, float((ctx.t_ns[e - 1] - ctx.t_ns[s]) / 1e9))

    # 统一将 q 转换为标签（如 0.95 → q95）
    def _qtag(q: float) -> str:
        return f"q{int(round(q * 100))}"

    out: Dict[str, float] = {}
    q_list_sorted = sorted([q for q in q_list if 0.0 < q < 1.0])
    if not q_list_sorted:
        q_list_sorted = [0.95]

    last_q = q_list_sorted[-1]
    thr_last = float(np.quantile(dv, last_q)) if np.isfinite(dv).all() else np.nan
    # 主口径使用最大分位
    for q in q_list_sorted:
        thr = float(np.quantile(dv, q)) if np.isfinite(dv).all() else np.nan
        if not np.isfinite(thr) or thr <= 0:
            continue
        mask = dv >= thr
        if not mask.any():
            # 输出0/NaN以保持列齐全
            tag = _qtag(q)
            out.update({
                f'large_{tag}_buy_dollar_sum': 0.0,
                f'large_{tag}_sell_dollar_sum': 0.0,
                f'large_{tag}_buy_count': 0.0,
                f'large_{tag}_sell_count': 0.0,
                f'large_{tag}_dollar_share': np.nan if total_dollar <= 0 else 0.0,
                f'large_{tag}_lti': np.nan,
                f'large_{tag}_signed_dollar_ps': 0.0,
            })
            continue
        sd = sign[mask] * dv[mask]
        buy_dollar = float(dv[mask][sign[mask] > 0].sum())
        sell_dollar = float(dv[mask][sign[mask] < 0].sum())
        buy_count = float(np.count_nonzero(sign[mask] > 0))
        sell_count = float(np.count_nonzero(sign[mask] < 0))
        large_dollar_sum = float(dv[mask].sum())
        large_signed_dollar_sum = float(sd.sum())
        lti = (buy_dollar - sell_dollar) / (buy_dollar + sell_dollar + eps)
        dollar_share = large_dollar_sum / (total_dollar + eps) if total_dollar > 0 else np.nan
        ps = large_signed_dollar_sum / duration_seconds

        tag = _qtag(q)
        out.update({
            f'large_{tag}_buy_dollar_sum': buy_dollar,
            f'large_{tag}_sell_dollar_sum': sell_dollar,
            f'large_{tag}_buy_count': buy_count,
            f'large_{tag}_sell_count': sell_count,
            f'large_{tag}_dollar_share': dollar_share,
            f'large_{tag}_lti': lti,
            f'large_{tag}_signed_dollar_ps': ps,
        })

    # 若主口径有效，复制到无后缀的简洁列，便于下游使用
    tag_last = _qtag(last_q)
    key_base = f'large_{tag_last}'
    for k in [
        'buy_dollar_sum', 'sell_dollar_sum', 'buy_count', 'sell_count',
        'dollar_share', 'lti', 'signed_dollar_ps'
    ]:
        src = f'{key_base}_{k}'
        if src in out:
            out[f'large_{k}'] = out[src]

    return out


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
    # 霍克斯过程（Hawkes Process）是一种 “自激发的点过程”，核心假设是：一个事件（如一笔交易）的发生，会提高短期内另一事件发生的概率（即 “交易引发更多交易”），最终导致事件在时间上呈现 “聚类特征”（密集时段与稀疏时段交替）。

    # 这段代码正是通过 “时间间隔阈值” 捕捉这种特征 —— 交易间隔小（触发后续交易）的归为同一聚类，间隔大（无触发）的划分为新聚类，完美契合霍克斯过程的 “自激发” 逻辑，因此命名为 “hawkes_clustering”。
    return {
        'hawkes_cluster_count': float(len(clusters)), #交易被划分为的 “密集时段数量”	
        'hawkes_cluster_size_mean': size_mean, #每个密集时段的平均交易数（越大，整体交易越密集）	
        'hawkes_cluster_size_max': size_max, #最密集时段的交易数（反映 “交易高峰” 的强度）	
        'hawkes_clustering_degree': float(degree) if pd.notna(degree) else np.nan, #平均聚类大小 / 总交易数（取值 [0,1]，越接近 1，交易越集中）	
    }

# 若两个相关性指标均为正且显著，说明当前价格上涨是 “真买盘推动”（而非虚涨），持续性可能更强；若相关性低，则需警惕价格波动的虚假性。
def _fast_cum_signed_flow_price_corr(ctx: TradesContext, s: int, e: int) -> Dict[str, float]:
    if e - s <= 2:
        return {'corr_cumsum_signed_qty_logp': np.nan, 'corr_cumsum_signed_dollar_logp': np.nan}
    # 累计相对序列
    logp_rel = ctx.logp[s:e] - ctx.logp[s]
    cs_signed_qty = (ctx.csum_signed_qty[s:e] - (ctx.csum_signed_qty[s] if s < ctx.csum_signed_qty.size else 0.0))
    # 累计有符号交易金额序列：反映区间内净买入/卖出的金额压力
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
    bar_zip_path :str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    标签：由 dollar bars 生成的未来 N-bar 对数收益，返回含时间信息的 DataFrame：
      - y: 标签值
      - horizon_bars: 预测的未来bar数
      - t0_time: 标签基准bar的 close 时间（end_time）
      - tH_time: 未来 horizon_bars 对应 bar 的 close 时间（end_time）
    因子：对齐到对应区间 [start_time, end_time)（过去或未来N个bar）上基于逐笔成交直接计算。
    返回：X_interval, y_df, bars
    """

    bars = None
    if bar_zip_path is not None and os.path.exists(bar_zip_path):
        bars = pd.read_csv(bar_zip_path)
    else:
        bars = build_dollar_bars(trades, dollar_threshold=dollar_threshold)
        os.makedirs(os.path.dirname(bar_zip_path), exist_ok=True)
        bars.to_csv(bar_zip_path, index=False,compression={'method': 'zip', 'archive_name': 'bars.csv'})


    if bars.empty:
        return pd.DataFrame(), pd.Series(dtype=float), bars

    bars = bars.reset_index(drop=True)
    bars['bar_id'] = bars.index
    bars['start_time'] = pd.to_datetime(bars['start_time'])
    bars['end_time'] = pd.to_datetime(bars['end_time'])
    close_s = bars.set_index('bar_id')['close']

    # 未来 N-bar 对数收益 + 时间信息
    y_series = np.log(close_s.shift(-horizon_bars) / close_s.shift(-1))
    end_time_s = bars.set_index('bar_id')['end_time']
    y = pd.DataFrame({
        'y': y_series,
        'horizon_bars': int(horizon_bars),
        't0_time': end_time_s,
        'tH_time': end_time_s.shift(-horizon_bars),
    })

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
                
        feature_start_ts = bars.loc[start_idx, 'start_time']
        feature_end_ts = bars.loc[end_idx, 'end_time']
        feat = _compute_interval_trade_features_fast(ctx, feature_start_ts, feature_end_ts)
        print(idx)
        idx = idx + 1
        feat['bar_id'] = bar_id
        feat['feature_start'] = feature_start_ts  # 特征计算区间的开始
        feat['feature_end'] = feature_end_ts      # 特征计算区间的结束
        feat['prediction_time'] = bars.loc[bar_id, 'end_time']  # 预测时间点
        if bar_id + horizon_bars < len(bars) :
            feat['settle_time'] = bars.loc[bar_id + horizon_bars, 'end_time']
        else:
             feat['settle_time'] = np.nan
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
    bar_zip_path: str = None,
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
        bar_zip_path = bar_zip_path
    )
    
    mask = y['y'].notna() & np.isfinite(y['y'].values)
    X = X.loc[mask].replace([np.inf, -np.inf], np.nan)
    y = y.loc[X.index]
    bars = bars[bars['end_time'].isin(X['feature_end'])]

    if X.empty or y.empty:
        return {'error': '数据不足或阈值设置过大，无法构造区间数据集', 'X': X, 'y': y, 'bars': bars}

    # 用对应样本的区间结束时间作为索引
    # 若为 past 模式：使用当前锚点 bar 的 end_time 代表预测时点
    # end_times = bars.set_index('bar_id')['feature_end']
    # idx_time = pd.to_datetime(end_times.loc[X.index])
    X2 = X.copy(); 
    # X2.index = idx_time
    y2 = y.copy(); 
    # y2.index = idx_time

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
        y=y2['y'],
        n_splits=n_splits,
        embargo=embargo_td,
        model_type=model_type,
        random_state=random_state,
        period_seconds=period_seconds,
    )
    
    # 基于区间特征识别“大单主动买入”事件
    # try:
    #     events = detect_large_aggressive_buy(trades=trades, X_interval=X)
    # except Exception:
    #     events = pd.DataFrame()

    return {'eval': eval_result, 'X': X, 'y': y, 'bars': bars}
    # return {'eval': eval_result, 'X': X, 'y': y, 'bars': bars, 'events_large_aggr_buy': events}

 


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
    'detect_large_aggressive_buy',
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

def _robust_zscore(series: pd.Series) -> pd.Series:
    """基于中位数与MAD的稳健z分数（避免极端值影响）。"""
    x = pd.to_numeric(series, errors='coerce')
    med = x.median()
    mad = (x - med).abs().median()
    eps = 1e-9
    return 0.6745 * (x - med) / (mad + eps)

def _forward_return_by_seconds(ctx: TradesContext, anchor_ts: pd.Timestamp, forward_seconds: float) -> float:
    """在成交序列上以锚点时间为起点，计算 forward_seconds 之后的对数收益（若数据不足返回NaN）。"""
    try:
        start_ts = anchor_ts
        end_ts = anchor_ts + pd.Timedelta(seconds=float(max(0.0, forward_seconds)))
        s, e = ctx.locate(start_ts, end_ts)
        if e - s <= 1:
            return np.nan
        p0 = ctx.price[s] if s < ctx.price.size else np.nan
        p1 = ctx.price[e - 1] if (e - 1) < ctx.price.size else np.nan
        if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0 or p1 <= 0:
            return np.nan
        return float(np.log(p1 / p0))
    except Exception:
        return np.nan

def detect_large_aggressive_buy(
    trades: pd.DataFrame,
    X_interval: pd.DataFrame,
    min_votes: int = 3,
    ofi_z_thresh: float = 2.0,
    tail_share_thresh: float = 0.3,
    gof_volume_thresh: float = 0.2,
    micro_z_thresh: float = 1.0,
    confirm_seconds: float = 2.0,
    confirm_ret_thresh: float = 0.0005,
) -> pd.DataFrame:
    """
    基于已计算的区间特征，识别“大单主动买入”事件。

    返回包含每个样本（bar_id）的事件表：
      - event_time: 使用 X_interval['prediction_time']（或 'feature_end'）
      - 条件布尔列与 votes 数
      - score: 归一化简单得分（votes / 条件数），便于排序
      - is_large_aggr_buy: 是否满足阈值（含价格事后确认）
    """
    if X_interval is None or X_interval.empty:
        return pd.DataFrame(columns=['bar_id', 'event_time', 'is_large_aggr_buy'])

    X = X_interval.copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # 需要的列若不存在则填充NaN，保持稳健
    for col in [
        'ofi_roll_sum_max', 'large_tail_dollar_share', 'gof_by_volume',
        'signed_vwap_deviation', 'vwap_deviation', 'micro_dp_zscore',
        'micro_dp_short', 'impact_perm_share'
    ]:
        if col not in X.columns:
            X[col] = np.nan

    # ofi z-score（稳健）
    ofi_z = _robust_zscore(X['ofi_roll_sum_max'])

    cond_ofi = ofi_z >= ofi_z_thresh
    cond_tail = X['large_tail_dollar_share'] >= tail_share_thresh
    cond_gof = X['gof_by_volume'] >= gof_volume_thresh
    cond_vwap = (X['signed_vwap_deviation'] > 0) | (X['vwap_deviation'] > 0)
    cond_micro = (X['micro_dp_zscore'] >= micro_z_thresh) | (X['micro_dp_short'] > 0)
    cond_perm = X['impact_perm_share'] > 0

    # 事后价格确认（以交易序列近似mid），在 prediction_time 或 feature_end 之后确认
    event_time_col = 'prediction_time' if 'prediction_time' in X.columns else ('feature_end' if 'feature_end' in X.columns else None)
    if event_time_col is None:
        # 若没有时间信息，只能返回基于特征的条件判断
        forward_confirm = pd.Series(index=X.index, data=np.nan)
        cond_confirm = pd.Series(index=X.index, data=False)
    else:
        ctx = _build_trades_context(trades)
        forward_ret = X[event_time_col].apply(lambda t: _forward_return_by_seconds(ctx, pd.to_datetime(t), confirm_seconds))
        forward_confirm = forward_ret
        cond_confirm = forward_ret >= np.log(1.0 + confirm_ret_thresh)

    # 计票（不含确认与含确认两套）
    conditions = pd.DataFrame({
        'cond_ofi': cond_ofi.fillna(False),
        'cond_tail': cond_tail.fillna(False),
        'cond_gof': cond_gof.fillna(False),
        'cond_vwap': cond_vwap.fillna(False),
        'cond_micro': cond_micro.fillna(False),
        'cond_perm': cond_perm.fillna(False),
    })
    votes = conditions.sum(axis=1)
    base_pass = votes >= int(min_votes)
    final_pass = base_pass & cond_confirm.fillna(False)

    out = pd.DataFrame({
        'bar_id': X.index,
        'event_time': X[event_time_col] if event_time_col else pd.NaT,
        'ofi_z': ofi_z,
        'cond_ofi': conditions['cond_ofi'],
        'cond_tail': conditions['cond_tail'],
        'cond_gof': conditions['cond_gof'],
        'cond_vwap': conditions['cond_vwap'],
        'cond_micro': conditions['cond_micro'],
        'cond_perm': conditions['cond_perm'],
        'votes': votes.astype(int),
        'forward_logret': forward_confirm,
        'is_large_aggr_buy': final_pass,
    })
    # 简单分数：票数占比 + 前瞻收益正向奖励
    total_conds = float(conditions.shape[1])
    score = votes.astype(float) / max(1.0, total_conds)
    bonus = np.clip(forward_confirm.fillna(0.0), 0.0, 0.005) * 50.0  # 最高+0.25
    out['score'] = (score + bonus).astype(float)
    return out.set_index('bar_id')

def main():    
    start_date = '2025-01-01'
    end_date = '2025-01-30'
    dollar_threshold=10000*6000
    dollar_threshold_str = str(dollar_threshold).replace("*", "_")

    date_list = generate_date_range(start_date, end_date)

    trades_zip_path = f'/Users/aming/project/python/crypto-trade/output/trades-{start_date}-{end_date}-{dollar_threshold_str}.zip'
    bar_zip_path = f'/Users/aming/project/python/crypto-trade/output/bars-{start_date}-{end_date}-{dollar_threshold_str}.zip'
    trades_df = None

    if trades_zip_path is not None and os.path.exists(trades_zip_path):
        trades_df = pd.read_csv(trades_zip_path)
    else:
        raw_df = []
        for date in date_list:
            raw_df.append(pd.read_csv(f'/Volumes/Ext-Disk/data/futures/um/daily/trades/ETHUSDT/ETHUSDT-trades-{date}.zip'))
        trades_df = pd.concat(raw_df, ignore_index=True)
        
        os.makedirs(os.path.dirname(trades_zip_path), exist_ok=True)
        
        trades_df.to_csv(
            trades_zip_path,
            index=False,
            compression={'method': 'zip', 'archive_name': 'trades_df.csv'}  # zip里csv的文件名
        )


        
    res = run_bar_interval_pipeline(
        trades=trades_df,
        dollar_threshold=dollar_threshold,
        feature_window_bars=10,
        horizon_bars=3,
        window_mode='past',
        n_splits=5,
        embargo_bars=None,
        model_type='ridge',
        bar_zip_path = bar_zip_path
    )
    # print(res.get('eval', {}).get('summary'))

if __name__ == '__main__':
    main()



