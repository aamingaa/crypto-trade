"""
Enhanced dollar bar builder.

This module provides `build_dollar_bar_enhanced`, which constructs dollar-value
bars from a stream/table of trades and augments each bar with advanced
per-bar statistics and cumulative (cs_*) columns suitable for fast interval
aggregation.

Input trades are expected to contain at least the following columns:
  - timestamp: pandas.Timestamp or datetime64[ns]
  - price: float
  - qty (or size/amount/volume): float (trade size in base units)

Optional columns that will be used if present:
  - dollar: float (if not provided, computed as price * qty)
  - sign: int in {+1, -1} denoting aggressive trade direction
  - side: str in {"buy", "sell"} or bool `is_buyer_maker` (Binance style)

Key outputs per bar include:
  - start_time, end_time
  - open, high, low, close
  - volume (base units), dollar_value (quote units)
  - buy_volume, sell_volume (if a sign can be inferred)
  - trades (count)
  - bar-level realized variation components: rv (sum of squared log returns),
    bpv (bi-power variation proxy), abs_r_sum, and higher moments
  - per-bar sums that are cumulative-additive across bars
  - cs_* cumulative columns computed via cumsum across bars
  - first_ticks, last_ticks: edge buffers storing the first/last m trades of a bar
  - start_trade_idx, end_trade_idx: absolute indices into the input trades
  - t_ns_sum, t_ns_sum_sq: per-bar sums of trade timestamps (ns) and their squares
    enabling prefix-sum style aggregation of mean/variance of timestamps

Notes on design:
  - The function is streaming-friendly and handles bars by accumulating quote
    value up to a threshold. It does not split single trades across bars; the
    last trade that crosses the threshold is included in the closing bar.
  - For applications requiring exact thresholding via partial fills, a
    trade-splitting variant can be implemented in the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from collections import deque


# ------------------------------
# Helper utilities
# ------------------------------


def _infer_sign_column(trades: pd.DataFrame) -> np.ndarray:
    """Infer aggressive side sign (+1 buy, -1 sell) from common columns.

    Priority order:
      1) `sign` if present
      2) `side` in {"buy","sell"}
      3) `is_buyer_maker` (Binance): True means seller-initiated => sign = -1
    Fallback: zeros (treated as unknown) if none present.
    """
    if "sign" in trades.columns:
        sign = trades["sign"].to_numpy()
        # Normalize to {-1, 0, +1}
        sign = np.where(sign > 0, 1, np.where(sign < 0, -1, 0))
        return sign
    if "side" in trades.columns:
        side = trades["side"].astype(str).str.lower().to_numpy()
        return np.where(side == "buy", 1, np.where(side == "sell", -1, 0))
    if "is_buyer_maker" in trades.columns:
        maker = trades["is_buyer_maker"].astype(bool).to_numpy()
        # True => buyer is maker => sell aggression
        return np.where(maker, -1, 1)
    return np.zeros(len(trades), dtype=int)


def _ensure_columns(trades: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist and have the right dtypes.

    - Ensures `timestamp` is datetime64[ns]
    - Ensures `price` and `qty`
    - Adds `dollar = price * qty` if missing
    - Adds `sign` if inferrable; otherwise zeros
    - Sorts by timestamp if not already sorted
    """
    required_price = "price"
    required_qty_candidates = ("qty", "size", "amount", "volume")

    if "timestamp" not in trades.columns:
        raise ValueError("trades must contain a 'timestamp' column")
    if required_price not in trades.columns:
        raise ValueError("trades must contain a 'price' column")

    qty_col: Optional[str] = None
    for c in required_qty_candidates:
        if c in trades.columns:
            qty_col = c
            break
    if qty_col is None:
        raise ValueError("trades must contain one of qty/size/amount/volume")

    df = trades.copy()
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)

    # Normalize names for local use
    df.rename(columns={qty_col: "qty"}, inplace=True)
    df["price"] = df["price"].astype(float)
    df["qty"] = df["qty"].astype(float)

    if "dollar" not in df.columns:
        df["dollar"] = df["price"] * df["qty"]

    if "sign" not in df.columns and ("side" in df.columns or "is_buyer_maker" in df.columns):
        df["sign"] = _infer_sign_column(df)
    elif "sign" in df.columns:
        df["sign"] = _infer_sign_column(df)
    else:
        df["sign"] = 0

    return df


@dataclass
class EnhancedBarConfig:
    dollar_threshold: float
    edge_tick_buffer: int = 3
    compute_higher_moments: bool = True
    compute_bpv: bool = True
    compute_tripower: bool = False
    micro_segments: int = 0  # 0 disables micro-segmentation


def _finalize_bar_record(
    bar_records: List[Dict[str, object]],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    first_time_ns: int,
    last_time_ns: int,
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    sum_qty: float,
    sum_dollar: float,
    sum_signed_qty: float,
    sum_signed_quote: float,
    num_trades: int,
    sum_pxqty: float,
    rv_sum: float,
    abs_r_sum: float,
    r2_sum: float,
    r3_sum: float,
    r4_sum: float,
    bpv_sum: float,
    dt_sum: float,
    dt_sum_sq: float,
    dt_min: float,
    dt_max: float,
    first_ticks: List[Tuple[pd.Timestamp, float, int, float]],
    last_ticks: List[Tuple[pd.Timestamp, float, int, float]],
    start_trade_idx: int,
    end_trade_idx: int,
    t_ns_sum: int,
    t_ns_sum_sq: int,
    micro_payload: Optional[Dict[str, float]] = None,
) -> None:
    buy_volume = 0.5 * (sum_qty + sum_signed_qty)
    sell_volume = 0.5 * (sum_qty - sum_signed_qty)

    rec: Dict[str, object] = {
        "start_time": start_time,
        "end_time": end_time,
        "open": float(open_price),
        "high": float(high_price),
        "low": float(low_price),
        "close": float(close_price),
        "volume": float(sum_qty),
        "dollar_value": float(sum_dollar),
        "buy_volume": float(buy_volume),
        "sell_volume": float(sell_volume),
        "trades": int(num_trades),
        "start_trade_idx": int(start_trade_idx),
        "end_trade_idx": int(end_trade_idx),
        # additive sums for cumulative columns
        "sum_signed_qty": float(sum_signed_qty),
        "sum_signed_quote": float(sum_signed_quote),
        "sum_pxqty": float(sum_pxqty),
        # realized variation components
        "rv": float(rv_sum),
        "abs_r_sum": float(abs_r_sum),
        "r2_sum": float(r2_sum),
        "r3_sum": float(r3_sum),
        "r4_sum": float(r4_sum),
        "bpv": float(bpv_sum),
        # inter-arrival stats (seconds)
        "dt_sum": float(dt_sum),
        "dt_sum_sq": float(dt_sum_sq),
        "dt_min": float(dt_min) if np.isfinite(dt_min) else np.nan,
        "dt_max": float(dt_max) if np.isfinite(dt_max) else np.nan,
        # timestamp sums (nanoseconds) for prefix aggregates
        "first_time_ns": int(first_time_ns),
        "last_time_ns": int(last_time_ns),
        "t_ns_sum": int(t_ns_sum),
        "t_ns_sum_sq": int(t_ns_sum_sq),
        # edge buffers
        "first_ticks": list(first_ticks),
        "last_ticks": list(last_ticks),
    }
    if micro_payload:
        rec.update(micro_payload)

    bar_records.append(rec)


def build_dollar_bar_enhanced(
    trades: pd.DataFrame,
    dollar_threshold: float,
    *,
    edge_tick_buffer: int = 3,
    micro_segments: int = 0,
    compute_higher_moments: bool = True,
    compute_bpv: bool = True,
    compute_tripower: bool = False,
) -> pd.DataFrame:
    """Build dollar-value bars with advanced statistics and cumulative columns.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade table sorted by time, with columns described in the module doc.
    dollar_threshold : float
        Target dollar value per bar (accumulated quote volume per bar).
    edge_tick_buffer : int
        Number of first/last ticks to store for each bar.
    micro_segments : int
        If > 0, compute equal-size micro-segment aggregates per bar. For each
        k in [0, K-1], adds `micro_k_qty`, `micro_k_signed_qty`, `micro_k_dollar`,
        and `micro_k_price_mean`.
    compute_higher_moments : bool
        If True, compute r^3, r^4 aggregates per bar.
    compute_bpv : bool
        If True, compute bipower variation proxy per bar.
    compute_tripower : bool
        If True, compute a simple tripower variation proxy per bar.

    Returns
    -------
    pd.DataFrame
        Bars with enriched columns and cs_* cumulative columns.
    """
    if dollar_threshold <= 0:
        raise ValueError("dollar_threshold must be positive")

    df = _ensure_columns(trades)
    ts = df["timestamp"].to_numpy()
    price = df["price"].to_numpy(dtype=float)
    qty = df["qty"].to_numpy(dtype=float)
    dollar = df["dollar"].to_numpy(dtype=float)
    sign = df["sign"].to_numpy(dtype=int)

    bar_records: List[Dict[str, object]] = []

    # State for the current bar
    bar_sum_dollar = 0.0
    bar_sum_qty = 0.0
    bar_sum_signed_qty = 0.0
    bar_sum_signed_quote = 0.0
    bar_sum_pxqty = 0.0
    bar_num_trades = 0

    bar_open: float = np.nan
    bar_high: float = -np.inf
    bar_low: float = np.inf
    bar_close: float = np.nan
    bar_start_time: Optional[pd.Timestamp] = None
    bar_end_time: Optional[pd.Timestamp] = None
    bar_start_trade_idx: Optional[int] = None
    bar_end_trade_idx: Optional[int] = None
    bar_first_time_ns: Optional[int] = None
    bar_last_time_ns: Optional[int] = None

    prev_trade_price: Optional[float] = None
    prev_trade_ts: Optional[pd.Timestamp] = None

    # realized variation components
    rv_sum = 0.0
    abs_r_sum = 0.0
    r2_sum = 0.0
    r3_sum = 0.0
    r4_sum = 0.0
    bpv_sum = 0.0
    tripower_sum = 0.0

    # keep last abs return for bpv
    prev_abs_r: Optional[float] = None

    # dt stats (seconds)
    dt_sum = 0.0
    dt_sum_sq = 0.0
    dt_min = np.inf
    dt_max = -np.inf

    # timestamp sums (nanoseconds) per bar
    t_ns_sum = 0
    t_ns_sum_sq = 0

    # edge tick buffers
    first_ticks: List[Tuple[pd.Timestamp, float, int, float]] = []
    last_ticks_deque: Deque[Tuple[pd.Timestamp, float, int, float]] = deque(maxlen=edge_tick_buffer)

    # micro segments accumulation
    micro_records: List[Tuple[float, float, float, float]] = []  # qty, signed_qty, dollar, price_sum

    def flush_bar() -> None:
        nonlocal bar_records
        nonlocal bar_sum_dollar, bar_sum_qty, bar_sum_signed_qty, bar_sum_signed_quote, bar_sum_pxqty
        nonlocal bar_num_trades, bar_open, bar_high, bar_low, bar_close, bar_start_time, bar_end_time
        nonlocal bar_start_trade_idx, bar_end_trade_idx, bar_first_time_ns, bar_last_time_ns
        nonlocal prev_trade_price, prev_trade_ts
        nonlocal rv_sum, abs_r_sum, r2_sum, r3_sum, r4_sum, bpv_sum, tripower_sum
        nonlocal prev_abs_r
        nonlocal dt_sum, dt_sum_sq, dt_min, dt_max
        nonlocal first_ticks, last_ticks_deque
        nonlocal micro_records
        nonlocal t_ns_sum, t_ns_sum_sq

        if bar_num_trades == 0:
            return

        micro_payload: Optional[Dict[str, float]] = None
        if micro_segments and bar_num_trades > 0:
            # Split the trades within the bar into K chunks of nearly equal size
            # using accumulated micro_records which we filled per trade.
            # micro_records holds (qty, signed_qty, dollar, price_sum) per trade.
            K = int(micro_segments)
            counts = [0] * K
            m_qty = [0.0] * K
            m_sqty = [0.0] * K
            m_dollar = [0.0] * K
            m_price_sum = [0.0] * K

            # Assign by index proportional mapping
            for idx, rec in enumerate(micro_records):
                k = int(np.floor(idx * K / max(1, len(micro_records))))
                if k >= K:
                    k = K - 1
                q, sq, d, ps = rec
                counts[k] += 1
                m_qty[k] += q
                m_sqty[k] += sq
                m_dollar[k] += d
                m_price_sum[k] += ps

            micro_payload = {}
            for k in range(K):
                micro_payload[f"micro_{k}_qty"] = float(m_qty[k])
                micro_payload[f"micro_{k}_signed_qty"] = float(m_sqty[k])
                micro_payload[f"micro_{k}_dollar"] = float(m_dollar[k])
                micro_payload[f"micro_{k}_price_mean"] = float(m_price_sum[k] / counts[k]) if counts[k] > 0 else np.nan

        _finalize_bar_record(
            bar_records=bar_records,
            start_time=bar_start_time if isinstance(bar_start_time, pd.Timestamp) else pd.Timestamp(bar_start_time),
            end_time=bar_end_time if isinstance(bar_end_time, pd.Timestamp) else pd.Timestamp(bar_end_time),
            first_time_ns=int(bar_first_time_ns) if bar_first_time_ns is not None else int(pd.Timestamp(bar_start_time).value),
            last_time_ns=int(bar_last_time_ns) if bar_last_time_ns is not None else int(pd.Timestamp(bar_end_time).value),
            open_price=bar_open,
            high_price=bar_high,
            low_price=bar_low,
            close_price=bar_close,
            sum_qty=bar_sum_qty,
            sum_dollar=bar_sum_dollar,
            sum_signed_qty=bar_sum_signed_qty,
            sum_signed_quote=bar_sum_signed_quote,
            num_trades=bar_num_trades,
            sum_pxqty=bar_sum_pxqty,
            rv_sum=rv_sum,
            abs_r_sum=abs_r_sum,
            r2_sum=r2_sum,
            r3_sum=r3_sum if compute_higher_moments else np.nan,
            r4_sum=r4_sum if compute_higher_moments else np.nan,
            bpv_sum=bpv_sum if compute_bpv else np.nan,
            dt_sum=dt_sum,
            dt_sum_sq=dt_sum_sq,
            dt_min=dt_min,
            dt_max=dt_max,
            first_ticks=first_ticks,
            last_ticks=list(last_ticks_deque),
            start_trade_idx=int(bar_start_trade_idx) if bar_start_trade_idx is not None else -1,
            end_trade_idx=int(bar_end_trade_idx) if bar_end_trade_idx is not None else -1,
            t_ns_sum=int(t_ns_sum),
            t_ns_sum_sq=int(t_ns_sum_sq),
            micro_payload=micro_payload,
        )

        # reset bar state
        bar_sum_dollar = 0.0
        bar_sum_qty = 0.0
        bar_sum_signed_qty = 0.0
        bar_sum_signed_quote = 0.0
        bar_sum_pxqty = 0.0
        bar_num_trades = 0
        bar_open = np.nan
        bar_high = -np.inf
        bar_low = np.inf
        bar_close = np.nan
        bar_start_time = None
        bar_end_time = None
        bar_start_trade_idx = None
        bar_end_trade_idx = None
        bar_first_time_ns = None
        bar_last_time_ns = None
        prev_trade_price = None
        prev_trade_ts = None
        rv_sum = 0.0
        abs_r_sum = 0.0
        r2_sum = 0.0
        r3_sum = 0.0
        r4_sum = 0.0
        bpv_sum = 0.0
        tripower_sum = 0.0
        prev_abs_r = None
        dt_sum = 0.0
        dt_sum_sq = 0.0
        dt_min = np.inf
        dt_max = -np.inf
        first_ticks = []
        last_ticks_deque.clear()
        micro_records = []
        t_ns_sum = 0
        t_ns_sum_sq = 0

    for i in range(len(df)):
        t = ts[i]
        p = price[i]
        q = qty[i]
        d = dollar[i]
        s = sign[i]
        tns = int(pd.Timestamp(t).value)

        if bar_num_trades == 0:
            bar_open = p
            bar_high = p
            bar_low = p
            bar_start_time = pd.Timestamp(t)
            bar_start_trade_idx = i
            bar_first_time_ns = tns
            # initialize prev trade trackers for this bar
            prev_trade_price = p
            prev_trade_ts = pd.Timestamp(t)
            # seed edge buffers
            first_ticks = [(pd.Timestamp(t), float(p), int(s), float(q))]
            last_ticks_deque.clear()
            last_ticks_deque.append((pd.Timestamp(t), float(p), int(s), float(q)))
        else:
            # update highs/lows
            if p > bar_high:
                bar_high = p
            if p < bar_low:
                bar_low = p

        # per-trade return (log)
        if prev_trade_price is not None and p > 0 and prev_trade_price > 0:
            r = float(np.log(p) - np.log(prev_trade_price))
            r_abs = abs(r)
            rv_sum += r * r
            abs_r_sum += r_abs
            r2_sum += r * r
            if compute_higher_moments:
                r3_sum += r * r * r
                r4_sum += r * r * r * r
            if compute_bpv and prev_abs_r is not None:
                bpv_sum += r_abs * prev_abs_r
            prev_abs_r = r_abs
        else:
            prev_abs_r = None

        # inter-arrival times within bar
        if prev_trade_ts is not None:
            dt = (pd.Timestamp(t) - prev_trade_ts).total_seconds()
            dt_sum += dt
            dt_sum_sq += dt * dt
            dt_min = min(dt_min, dt)
            dt_max = max(dt_max, dt)
        prev_trade_ts = pd.Timestamp(t)

        # accumulations
        bar_sum_dollar += d
        bar_sum_qty += q
        bar_sum_signed_qty += s * q
        bar_sum_signed_quote += s * d
        bar_sum_pxqty += p * q
        bar_num_trades += 1
        bar_close = p
        bar_end_time = pd.Timestamp(t)
        bar_end_trade_idx = i
        bar_last_time_ns = tns

        # micro per-trade record
        if micro_segments:
            micro_records.append((q, s * q, d, p))

        # update edge buffers
        if len(first_ticks) < edge_tick_buffer:
            first_ticks.append((pd.Timestamp(t), float(p), int(s), float(q)))
        last_ticks_deque.append((pd.Timestamp(t), float(p), int(s), float(q)))

        # timestamp prefixable sums
        t_ns_sum += tns
        t_ns_sum_sq += tns * tns

        # bar close condition
        if bar_sum_dollar >= dollar_threshold:
            flush_bar()

        prev_trade_price = p

    # flush the last (possibly partial) bar
    flush_bar()

    if not bar_records:
        # Return an empty DataFrame with expected columns
        empty = pd.DataFrame(columns=[
            "start_time", "end_time", "open", "high", "low", "close",
            "volume", "dollar_value", "buy_volume", "sell_volume", "trades",
            "sum_signed_qty", "sum_signed_quote", "sum_pxqty",
            "rv", "abs_r_sum", "r2_sum", "r3_sum", "r4_sum", "bpv",
            "dt_sum", "dt_sum_sq", "dt_min", "dt_max",
            "first_ticks", "last_ticks",
        ])
        return empty

    bars = pd.DataFrame.from_records(bar_records)

    # cumulative columns (monotone over bars, suitable for fast interval diffs)
    # Use names consistent with `_compute_interval_bar_features_fast` expectations
    bars = bars.copy()
    bars["cs_qty"] = bars["volume"].cumsum()
    bars["cs_quote"] = bars["dollar_value"].cumsum()
    bars["cs_signed_qty"] = bars["sum_signed_qty"].cumsum()
    bars["cs_signed_quote"] = bars["sum_signed_quote"].cumsum()
    bars["cs_pxqty"] = bars["sum_pxqty"].cumsum()
    bars["cs_ret2"] = bars["rv"].cumsum()
    bars["cs_abs_r"] = bars["abs_r_sum"].cumsum()
    bars["cs_bpv"] = bars["bpv"].cumsum() if "bpv" in bars.columns else np.nan
    # cumulative timestamp-related sums and dt sums for prefix aggregation
    bars["cs_trades"] = bars["trades"].cumsum()
    bars["cs_dt_sum"] = bars["dt_sum"].cumsum()
    bars["cs_dt_sum_sq"] = bars["dt_sum_sq"].cumsum()
    bars["cs_t_ns_sum"] = bars["t_ns_sum"].cumsum()
    bars["cs_t_ns_sum_sq"] = bars["t_ns_sum_sq"].cumsum()

    # Ensure time columns have tz-aware dtype consistently
    if not np.issubdtype(bars["start_time"].dtype, np.datetime64):
        bars["start_time"] = pd.to_datetime(bars["start_time"], utc=True)
    if not np.issubdtype(bars["end_time"].dtype, np.datetime64):
        bars["end_time"] = pd.to_datetime(bars["end_time"], utc=True)

    return bars


__all__ = ["build_dollar_bar_enhanced", "EnhancedBarConfig"]


