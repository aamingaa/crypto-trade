import pandas as pd

def non_negative_series(series):
    series = series.copy(deep=True)
    series['Returns'] = series['Close'].diff() / series['Close'].shift(1)
    series['rPrices'] = (1 + series['Returns']).cumprod()
    return series

def daily_bars(series):
    series = series.copy(deep=True)
    return group_bars(series, series.index.date)

def volume_bars(series, bar_size=10000):
    series = series.copy(deep=True)
    series['Cum Volume'] = series['Volume'].cumsum()
    bar_idx = (series['Cum Volume'] / bar_size).round(0).astype(int).values
    return group_bars(series, bar_idx)

def dollar_bars(series, bar_size=10000 * 3000):
    series = series.copy(deep=True)
    series['Dollar Volume'] = (series['Volume'] * series['Close'])
    series['Cum Dollar Volume'] = series['Dollar Volume'].cumsum()
    bar_idx = (series['Cum Dollar Volume'] / bar_size).round(0).astype(int).values
    return group_bars(series, bar_idx)

def dollar_bars_v2(series, bar_size=10000 * 3000):
    series = series.copy(deep=True)
    # series['Dollar Volume'] = (series['Volume'] * series['Close'])
    series['cum_quote_qty'] = series['quote_qty'].cumsum()
    bar_idx = (series['cum_quote_qty'] / bar_size).round(0).astype(int).values
    return group_bars_v2(series, bar_idx)

def group_bars_v2(series, bar_idx):
    gg = series.groupby(bar_idx)
    df = pd.DataFrame()
    df['qty'] = gg['qty'].sum()
    if 'quote_qty' in series.columns:
        df['quote_qty_sum'] = gg['quote_qty'].sum()
    df['open'] = gg['price'].first()
    df['close'] = gg['price'].last()
    df['high'] = gg['price'].max()
    df['low'] = gg['price'].min()

    if 'is_buyer_maker' in series.columns:
        def safe_calculate(group):
            maker_arr = group[group['is_buyer_maker'] == True]
            taker_arr = group[group['is_buyer_maker'] == False]
            
            maker_qty = maker_arr['quote_qty'].sum()
            taker_qty = taker_arr['quote_qty'].sum()
            maker_ticks = len(maker_arr)
            taker_ticks = len(taker_arr)
            
            total_qty = maker_qty + taker_qty
            total_ticks = maker_ticks + taker_ticks
            
            return pd.Series({
                'maker_qty': maker_qty,
                'taker_qty': taker_qty,
                'maker_ticks': maker_ticks,
                'seller_maker_ticks': taker_ticks,
                'maker_ratio': maker_ticks / total_ticks if total_ticks > 0 else 0,
                'taker_ratio': taker_ticks / total_ticks if total_ticks > 0 else 0,
                'maker_qty_ratio': maker_qty / total_qty if total_qty > 0 else 0,
                'taker_qty_ratio': taker_qty / total_qty if total_qty > 0 else 0
            })
        
        result = gg.apply(safe_calculate)
        df_maker_taker = pd.DataFrame(result)
        df = pd.concat([df, df_maker_taker], axis=1)


    # df['Instrument'] = gg['Instrument'].first()
    df['time'] = gg['time'].first()
    df['num_tickers'] = gg.size()
    
    # df = pd.concat([df, df_maker_taker], axis=1)

    
    df = df.set_index('time')
    return df





def group_bars(series, bar_idx):
    gg = series.groupby(bar_idx)
    df = pd.DataFrame()
    df['Volume'] = gg['Volume'].sum()
    if 'Dollar Volume' in series.columns:
        df['Dollar Volume'] = gg['Dollar Volume'].sum()
    df['Open'] = gg['Open'].first()
    df['Close'] = gg['Close'].last()
    if 'rPrices' in series.columns:
        df['rPrices'] = gg['rPrices'].last()
    df['Instrument'] = gg['Instrument'].first()
    df['Time'] = gg.apply(lambda x:x.index[0])
    df['Num Ticks'] = gg.size()
    df = df.set_index(gg.apply(lambda x:x.index[0]))
    return df


# def group_bars(series, bar_idx):
#     gg = series.groupby(bar_idx)
#     df = pd.DataFrame()
#     df['Volume'] = gg['Volume'].sum()
#     if 'Dollar Volume' in series.columns:
#         df['Dollar Volume'] = gg['Dollar Volume'].sum()
#     df['Open'] = gg['Open'].first()
#     df['Close'] = gg['Close'].last()
#     if 'rPrices' in series.columns:
#         df['rPrices'] = gg['rPrices'].last()
#     df['Instrument'] = gg['Instrument'].first()
#     df['Time'] = gg.apply(lambda x:x.index[0])
#     df['Num Ticks'] = gg.size()
#     df = df.set_index(gg.apply(lambda x:x.index[0]))
#     return df
