import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


def add_rolling_derivatives(
    X: pd.DataFrame,
    windows: List[int],
    stats: List[str],
) -> pd.DataFrame:
    """
    在 bar 级特征表上添加 N 根窗口的滚动派生特征。

    - windows: 例如 [3, 5, 10]
    - stats: 支持 'mean', 'sum', 'std', 'min', 'max', 'median', 'q90'
    注意：对以 "_lag" 结尾的列会跳过，避免重复放大维度。
    """
    X = X.copy()
    X = X.sort_index()

    base_cols = [c for c in X.columns if not c.endswith('_lag1') and not c.endswith('_lag2')]

    for w in windows:
        roll = X[base_cols].rolling(window=w, min_periods=w)
        for stat in stats:
            if stat == 'mean':
                tmp = roll.mean()
            elif stat == 'sum':
                tmp = roll.sum()
            elif stat == 'std':
                tmp = roll.std()
            elif stat == 'min':
                tmp = roll.min()
            elif stat == 'max':
                tmp = roll.max()
            elif stat == 'median':
                tmp = roll.median()
            elif stat == 'q90':
                tmp = roll.quantile(0.9)
            else:
                continue

            tmp.columns = [f'{c}_roll{w}_{stat}' for c in tmp.columns]
            X = X.join(tmp)

    return X


def _ensure_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    # 优先按纳秒/微秒/毫秒猜测
    s = series.astype(np.int64)
    # 避免负值或极端值导致 overflow
    s_abs = s.abs()
    # 简单阈值判断：> 1e14 视为纳秒，> 1e11 视为微秒，其它视为毫秒/秒
    if (s_abs > 1e14).any():
        return pd.to_datetime(series, unit='ns')
    if (s_abs > 1e11).any():
        return pd.to_datetime(series, unit='us')
    if (s_abs > 1e9).any():
        return pd.to_datetime(series, unit='ms')
    return pd.to_datetime(series, unit='s')


def build_dollar_bars(
    trades: pd.DataFrame,
    dollar_threshold: float,
) -> pd.DataFrame:
    """
    基于逐笔成交数据生成 dollar bars 的轴。

    trades 列要求：['time','id','price','qty','quote_qty','is_buyer_maker']
    - time: int|datetime
    - quote_qty: 成交额（价格×数量），若缺失则用 price*qty 代替
    - is_buyer_maker: 市价方向判定（True 表示成交对手为挂单买方 ⇒ 主动卖）

    返回：bar 级 DataFrame，包含每根 bar 的起止时间、OHLC、成交量/额等。
    """
    df = trades.copy()
    df['time'] = _ensure_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    if 'quote_qty' not in df.columns or df['quote_qty'].isna().all():
        df['quote_qty'] = df['price'] * df['qty']

    # 成交方向：taker 买为 +1，taker 卖为 -1
    # is_buyer_maker == True 表示成交对手是做市买方 ⇒ 主动方为卖
    df['trade_sign'] = np.where(df['is_buyer_maker'], -1, 1)

    # 向量化：使用“前缀和（移位前）”来决定 bar_id，确保阈值触发的那一笔属于上一根 bar
    cum_prev = df['quote_qty'].cumsum().shift(1).fillna(0.0)
    df['bar_id'] = (cum_prev // dollar_threshold).astype(int)

    # 预计算买/卖量
    df['buy_qty'] = df['qty'].where(df['trade_sign'] > 0, 0.0)
    df['sell_qty'] = df['qty'].where(df['trade_sign'] < 0, 0.0)

    # 分组一次性聚合 OHLC/成交量/金额/买卖量
    agg = {
        'time': ['first', 'last'],
        'price': ['first', 'max', 'min', 'last'],
        'qty': 'sum',
        'quote_qty': 'sum',
        'buy_qty': 'sum',
        'sell_qty': 'sum',
    }
    g = df.groupby('bar_id', sort=True).agg(agg)

    # 展平多级列
    g.columns = [
        'start_time', 'end_time',
        'open', 'high', 'low', 'close',
        'volume', 'dollar_value',
        'buy_volume', 'sell_volume'
    ]

    # 仅保留达到阈值的完整 bars（过滤最后一根不完整 bar）
    g = g[g['dollar_value'] >= dollar_threshold]

    # bar_id 作为列返回
    bars = g.reset_index()
    return bars


def aggregate_trade_features_on_bars(
    trades: pd.DataFrame,
    bars: pd.DataFrame,
) -> pd.DataFrame:
    """
    将逐笔交易因子聚合到 dollar bar 轴上。
    输入 trades 列：['time','id','price','qty','quote_qty','is_buyer_maker']
    输出以 bar_id 为索引的交易侧特征。
    """
    df = trades.copy()
    df['time'] = _ensure_datetime(df['time'])
    df = df.sort_values('time')
    if 'quote_qty' not in df.columns or df['quote_qty'].isna().all():
        df['quote_qty'] = df['price'] * df['qty']
    df['trade_sign'] = np.where(df['is_buyer_maker'], -1, 1)

    # 基于 bars 的时间边界做区间切分（左闭右开）
    features = []
    for _, b in bars.iterrows():
        mask = (df['time'] >= b['start_time']) & (df['time'] < b['end_time'])
        seg = df.loc[mask]
        if seg.empty:
            features.append({
                'bar_id': int(b['bar_id']),
                'trade_vwap': np.nan,
                'trade_volume_sum': 0.0,
                'trade_dollar_sum': 0.0,
                'trade_signed_volume': 0.0,
                'trade_buy_ratio': np.nan,
                'trade_intensity': 0.0,
                'trade_rv': np.nan,
            })
            continue

        # VWAP
        dollar = seg['quote_qty'].sum()
        vwap = (seg['price'] * seg['qty']).sum() / seg['qty'].sum() if seg['qty'].sum() > 0 else np.nan

        # 强度与方向
        signed_volume = (seg['qty'] * seg['trade_sign']).sum()
        buy_ratio = (seg.loc[seg['trade_sign']>0,'qty'].sum()) / seg['qty'].sum() if seg['qty'].sum() > 0 else np.nan

        # 实现波动率（对价格取对数差）
        seg = seg.copy()
        seg['logp'] = np.log(seg['price'])
        rv = (seg['logp'].diff().dropna() ** 2).sum()

        duration_seconds = max(1.0, (b['end_time'] - b['start_time']).total_seconds())
        intensity = len(seg) / duration_seconds

        features.append({
            'bar_id': int(b['bar_id']),
            'trade_vwap': vwap,
            'trade_volume_sum': seg['qty'].sum(),
            'trade_dollar_sum': dollar,
            'trade_signed_volume': signed_volume,
            'trade_buy_ratio': buy_ratio,
            'trade_intensity': intensity,
            'trade_rv': rv,
        })

    feat_df = pd.DataFrame(features).set_index('bar_id')
    return feat_df


def _twap(series: pd.Series, times: pd.Series) -> float:
    if series.empty:
        return np.nan
    # 基于停留时长的 TWAP：对相邻样本的持续时间加权
    t = times.view('int64')
    dt = np.diff(t, prepend=t.iloc[0])  # 第一个样本赋一个最小权重
    w = np.clip(dt.astype(float), 1.0, None)
    return float(np.average(series.astype(float), weights=w))


def aggregate_lob_features_on_bars(
    lob: pd.DataFrame,
    bars: pd.DataFrame,
) -> pd.DataFrame:
    """
    将五档订单簿在每根 dollar bar 的时间边界内聚合到低频。
    lob 列：['time','ap0','av0','bp0','bv0',...,'ap4','av4','bp4','bv4']
    输出以 bar_id 为索引的 LOB 特征。
    """
    book = lob.copy()
    book['time'] = _ensure_datetime(book['time'])
    book = book.sort_values('time')

    # 预计算常用序列
    mid = (book['ap0'] + book['bp0']) / 2.0
    spread = book['ap0'] - book['bp0']
    total_bid = book[[f'bv{i}' for i in range(5)]].sum(axis=1)
    total_ask = book[[f'av{i}' for i in range(5)]].sum(axis=1)
    depth_imb = (total_bid - total_ask) / (total_bid + total_ask + 1e-12)
    microprice = (book['ap0'] * total_bid + book['bp0'] * total_ask) / (total_bid + total_ask + 1e-12)
    micro_dev = (microprice - mid) / (spread.replace(0, np.nan))

    # OFI（order flow imbalance，基于 best 价与量的变动）
    d_ap0 = book['ap0'].diff()
    d_bp0 = book['bp0'].diff()
    d_av0 = book['av0'].diff()
    d_bv0 = book['bv0'].diff()
    ofi = (
        (d_bp0.gt(0).astype(int) - d_bp0.lt(0).astype(int)) * book['bv0'].shift(1).fillna(0)
        - (d_ap0.lt(0).astype(int) - d_ap0.gt(0).astype(int)) * book['av0'].shift(1).fillna(0)
        + d_bv0.where(d_bp0.eq(0), 0).fillna(0)
        - d_av0.where(d_ap0.eq(0), 0).fillna(0)
    )

    features = []
    for _, b in bars.iterrows():
        mask = (book['time'] >= b['start_time']) & (book['time'] < b['end_time'])
        seg = book.loc[mask]
        if seg.empty:
            features.append({
                'bar_id': int(b['bar_id']),
                'lob_spread_twap': np.nan,
                'lob_depth_twap': np.nan,
                'lob_imbalance_twap': np.nan,
                'lob_microprice_dev_twap': np.nan,
                'lob_ofi_sum': 0.0,
                'lob_cancel_rate': np.nan,
            })
            continue

        idx = seg.index
        sp_twap = _twap(spread.loc[idx], book.loc[idx, 'time'])
        depth_twap = _twap((total_bid + total_ask).loc[idx], book.loc[idx, 'time'])
        imb_twap = _twap(depth_imb.loc[idx], book.loc[idx, 'time'])
        micro_dev_twap = _twap(micro_dev.loc[idx], book.loc[idx, 'time'])

        ofi_sum = ofi.loc[idx].sum()

        # 撤单率粗略估计：深度减少的量 / 总深度
        tb = total_bid.loc[idx]
        ta = total_ask.loc[idx]
        cancel_bid = tb.diff().where(lambda x: x < 0, 0).abs().sum()
        cancel_ask = ta.diff().where(lambda x: x < 0, 0).abs().sum()
        cancel_rate = (cancel_bid + cancel_ask) / (tb.sum() + ta.sum() + 1e-12)

        features.append({
            'bar_id': int(b['bar_id']),
            'lob_spread_twap': sp_twap,
            'lob_depth_twap': depth_twap,
            'lob_imbalance_twap': imb_twap,
            'lob_microprice_dev_twap': micro_dev_twap,
            'lob_ofi_sum': ofi_sum,
            'lob_cancel_rate': cancel_rate,
        })

    feat_df = pd.DataFrame(features).set_index('bar_id')
    return feat_df


def make_dataset(
    trades: pd.DataFrame,
    lob: pd.DataFrame,
    dollar_threshold: float,
    horizon_n: int,
    add_lags: int = 2,
    use_interactions: bool = True,
    rolling_windows: Optional[List[int]] = None,
    rolling_stats: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    生成训练集：在 dollar bar 轴上融合交易与订单簿特征，并构造 N-bar 标签。

    返回：(X, y, bars)
    - X: 以 bar_id 为索引的特征表
    - y: 对应的 N-bar 未来收益（对数收益）
    - bars: bar 基础信息（含 close 等），便于回测与对齐
    """
    bars = build_dollar_bars(trades, dollar_threshold=dollar_threshold)
    if bars.empty:
        return pd.DataFrame(), pd.Series(dtype=float), bars

    trade_feat = aggregate_trade_features_on_bars(trades, bars)
    lob_feat = aggregate_lob_features_on_bars(lob, bars)

    # 融合
    X = trade_feat.join(lob_feat, how='inner')

    # 交互项（可选）
    if use_interactions:
        if {'lob_spread_twap','lob_ofi_sum'}.issubset(X.columns):
            X['ofi_x_spread'] = X['lob_ofi_sum'] * X['lob_spread_twap']
        if {'trade_intensity','lob_depth_twap'}.issubset(X.columns):
            X['intensity_x_depth'] = X['trade_intensity'] * X['lob_depth_twap']
        if {'trade_buy_ratio','lob_imbalance_twap'}.issubset(X.columns):
            X['buyratio_x_imb'] = X['trade_buy_ratio'] * X['lob_imbalance_twap']

    # 滞后特征
    for k in range(1, add_lags + 1):
        for col in list(X.columns):
            X[f'{col}_lag{k}'] = X[col].shift(k)

    # 滚动派生特征
    if rolling_windows:
        X = add_rolling_derivatives(
            X,
            windows=rolling_windows,
            stats=rolling_stats or ['mean', 'sum'],
        )

    # 标签：未来 N 根的对数收益，基于 bar 收盘价
    close = bars.set_index('bar_id')['close']
    y = np.log(close.shift(-horizon_n) / close)

    # 对齐：仅保留特征和标签同时非缺失的 bar
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    # 训练可用区间：去掉因滞后产生的前 add_lags 根和因标签产生的末尾 horizon_n 根
    X = X.dropna()
    y = y.loc[X.index]

    return X, y, bars


__all__ = [
    'build_dollar_bars',
    'aggregate_trade_features_on_bars',
    'aggregate_lob_features_on_bars',
    'make_dataset',
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



if __name__ == '__main__':
    # 最小用法示例（需用户提供 trades_df 和 lob_df DataFrame）
    # 假设两者 time 均为整型时间戳，可被 _ensure_datetime 自动识别
    # 这里放一个简单的占位示例，确保模块可直接运行不报错
    # print('dollarbar_fusion module ready.')
    date_list = generate_date_range('2025-01-01', '2025-01-02')
    raw_df = []
    for date in date_list:
        raw_df.append(pd.read_csv(f'/Volumes/Ext-Disk/data/futures/um/daily/trades/ETHUSDT/ETHUSDT-trades-{date}.zip'))
        
    dollar_bar = build_dollar_bars(raw_df, 10000 * 3000)
    print(dollar_bar)


