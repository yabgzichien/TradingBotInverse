import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz
import os

def initialize_mt5():
    if not mt5.initialize():
        print(f"initialize() failed, error code = {mt5.last_error()}")
        return False
    print(f"MetaTrader5 version {mt5.version()} connected.")
    return True

def _fetch_from_mt5(symbol, mt5_tf, start_date, end_date):
    timezone = pytz.timezone("UTC")
    utc_from = timezone.localize(start_date) if start_date.tzinfo is None else start_date
    utc_to = timezone.localize(end_date) if end_date.tzinfo is None else end_date
    
    rates = mt5.copy_rates_range(symbol, mt5_tf, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def get_data(symbol, timeframe, start_date, end_date, force_refresh=False):
    tf_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }
    
    mt5_tf = tf_map.get(timeframe)
    if mt5_tf is None:
        raise ValueError(f"Invalid timeframe: {timeframe}")

    cache_dir = "data"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{symbol}_{timeframe}.csv")

    req_start = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
    req_end = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date

    cached_df = pd.DataFrame()
    if not force_refresh and os.path.exists(cache_file):
        try:
            cached_df = pd.read_csv(cache_file, index_col='time', parse_dates=True)
        except Exception as e:
            print(f"Warning: Failed to read cache {cache_file} ({e}). Refetching all.")

    if not cached_df.empty:
        cache_min = cached_df.index.min()
        cache_max = cached_df.index.max()

        chunks = [cached_df]
        
        if req_start < cache_min:
            print(f"[{symbol} {timeframe}] Fetching older missing data: {req_start.date()} to {cache_min.date()}")
            older_df = _fetch_from_mt5(symbol, mt5_tf, req_start, cache_min)
            if not older_df.empty:
                chunks.append(older_df)
                
        if req_end > cache_max:
            print(f"[{symbol} {timeframe}] Fetching newer missing data: {cache_max.date()} to {req_end.date()}")
            newer_df = _fetch_from_mt5(symbol, mt5_tf, cache_max, req_end)
            if not newer_df.empty:
                chunks.append(newer_df)
                
        if len(chunks) > 1:
            cached_df = pd.concat(chunks)
            cached_df = cached_df[~cached_df.index.duplicated(keep='last')]
            cached_df.sort_index(inplace=True)
            cached_df.to_csv(cache_file)
            
    else:
        print(f"[{symbol} {timeframe}] No cache found or forced refresh. Fetching full range from MT5...")
        cached_df = _fetch_from_mt5(symbol, mt5_tf, start_date, end_date)
        if not cached_df.empty:
            cached_df.to_csv(cache_file)

    if cached_df.empty:
        print(f"Failed to fetch or load any data for {symbol} on {timeframe}.")
        return pd.DataFrame()

    mask = (cached_df.index >= req_start) & (cached_df.index <= req_end)
    return cached_df.loc[mask]


def get_symbol_specs(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    return {
        "point": float(getattr(info, "point", 0.0) or 0.0),
        "spread": float(getattr(info, "spread", 0.0) or 0.0),
        "trade_contract_size": float(getattr(info, "trade_contract_size", 0.0) or 0.0),
        "trade_tick_size": float(getattr(info, "trade_tick_size", 0.0) or 0.0),
        "trade_tick_value": float(getattr(info, "trade_tick_value", 0.0) or 0.0),
    }
