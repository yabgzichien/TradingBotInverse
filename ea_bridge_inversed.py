import os
import time
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from strategy import generate_signals_refined


@dataclass(frozen=True)
class BridgeConfig:
    symbols: tuple[str, ...] = ("BTCUSD", "XAUUSD")
    anchor_tf: str = "H4"
    execution_tf: str = "M15"
    poll_interval_sec: int = 15
    bars_limit: int = 1200
    risk_base_balance_usd: float = 10000.0
    risk_per_trade_pct: float = 0.01
    magic: int = 1001001
    replace_tol_points: float = 10.0
    max_pending_bars: int = 96
    anchor_swing_window: int = 7
    execution_swing_window: int = 1
    entry_retracement: float = 0.618
    sweep_mode: str = "prev_bar"
    internal_structure_lookback_bars: int = 1
    max_bos_wait_bars: int = 8
    invert_signals: bool = True


LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ea_bridge.log")

def _log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"{ts} [ea_bridge] {msg}"
    print(log_line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
    except Exception as e:
        print(f"[Logging Error] {e}", flush=True)


def _common_files_dir() -> str:
    appdata = os.environ.get("APPDATA", "")
    if not appdata:
        return ""
    return os.path.join(appdata, "MetaQuotes", "Terminal", "Common", "Files")


def _rates_path(symbol: str, timeframe: str) -> str:
    base = _common_files_dir()
    return os.path.join(base, f"ag_rates_{symbol}_{timeframe}.csv")


def _commands_path() -> str:
    base = _common_files_dir()
    return os.path.join(base, "ag_commands.csv")


def _ensure_commands_file():
    path = _commands_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        _log(f"Commands file exists: {path}")
        return
    fieldnames = [
        "ts",
        "symbol",
        "cmd",
        "dir",
        "entry",
        "sl",
        "tp",
        "risk_usd",
        "magic",
        "replace_tol_points",
        "max_pending_bars",
        "client_tag",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')
        w.writeheader()
    _log(f"Commands file created: {path}")



def _read_rates_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    
    # Try common encodings for MT5/CSV files
    encodings = ["utf-16", "utf-8-sig", "utf-8", "cp1252"]
    
    # Peek at the file to prioritize UTF-16 if BOM is present
    try:
        with open(path, "rb") as f:
            head = f.read(4)
        if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
            # Move utf-16 to front if detected
            encodings.insert(0, encodings.pop(encodings.index("utf-16")))
    except Exception:
        pass

    df = None
    last_err = None
    for attempt in range(5):
        for enc in encodings:
            try:
                # Use sep=None and engine='python' to auto-detect tab vs comma
                df = pd.read_csv(path, encoding=enc, sep=None, engine='python', on_bad_lines='skip')
                last_err = None
                break
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                last_err = e
                continue
            except (PermissionError, IOError) as e:
                last_err = e
                break # Wait for retry loop to sleep
            except pd.errors.EmptyDataError:
                return pd.DataFrame()
        
        if df is not None and last_err is None:
            break
        
        _log(f"File access delay for {os.path.basename(path)} (attempt {attempt+1}/5): {last_err}")
        time.sleep(0.5) 
    
    if df is None:
        if last_err is not None:
            _log(f"Warning: Could not read {path} after 5 attempts: {last_err}")
        return pd.DataFrame()
    
    if df.empty:
        return df
        
    # Standardize column names (MT5 sometimes exports with capitalized headers)
    df.columns = [c.lower() for c in df.columns]

    if "time" not in df.columns:
        _log(f"Warning: 'time' column missing in {os.path.basename(path)}. Columns found: {list(df.columns)}")
        return pd.DataFrame()
        
    if pd.api.types.is_numeric_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True, errors="coerce")
    else:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    cols = ["open", "high", "low", "close"]
    for c in cols:
        if c not in df.columns:
            _log(f"Warning: column '{c}' missing in {os.path.basename(path)}")
            return pd.DataFrame()
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    df = df.dropna(subset=cols)
    return df


def _append_command(row: dict):
    path = _commands_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    fieldnames = [
        "ts",
        "symbol",
        "cmd",
        "dir",
        "entry",
        "sl",
        "tp",
        "risk_usd",
        "magic",
        "replace_tol_points",
        "max_pending_bars",
        "client_tag",
    ]
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')
        if not exists:
            w.writeheader()
        w.writerow(row)
    _log(
        f"Command written: symbol={row.get('symbol')} cmd={row.get('cmd')} dir={row.get('dir')} "
        f"entry={row.get('entry')} sl={row.get('sl')} tp={row.get('tp')} tag={row.get('client_tag')}"
    )


def _latest_setup(df: pd.DataFrame):
    if df is None or df.empty or len(df) < 2:
        return None
    row = df.iloc[-2]
    setup_id = row.get("setup_id", None)
    entry_level = row.get("entry_level", None)
    setup_dir = row.get("setup_dir", None)
    sl_level = row.get("sl_level", None)
    tp_level = row.get("tp_level", None)
    bos_time = row.get("bos_time", pd.NaT)
    if pd.isna(setup_id) or pd.isna(entry_level) or pd.isna(setup_dir) or pd.isna(sl_level) or pd.isna(tp_level) or pd.isna(bos_time):
        return None
    return int(setup_id), int(setup_dir), float(entry_level), float(sl_level), float(tp_level), bos_time


def run_bridge(config: BridgeConfig):
    common_dir = _common_files_dir()
    if not common_dir or not os.path.isdir(common_dir):
        raise RuntimeError(f"MT5 Common Files folder not found: {common_dir}")

    _log(f"Common Files: {common_dir}")
    _log(
        "Config: "
        f"symbols={','.join(config.symbols)} anchor_tf={config.anchor_tf} exec_tf={config.execution_tf} "
        f"poll={config.poll_interval_sec}s bars_limit={config.bars_limit} magic={config.magic}"
    )
    _ensure_commands_file()
    last_sent_setup = {}
    start_time = datetime.now(timezone.utc)
    _log(f"Bridge started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    while True:
        now = datetime.now(timezone.utc)
        cycle_ts = now.isoformat()
        
        uptime = now - start_time
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        uptime_str = f"{days}d {hours}h {minutes}m"
        
        _log(f"Cycle start | Uptime: {uptime_str}")
        for symbol in config.symbols:
            anchor_path = _rates_path(symbol, config.anchor_tf)
            exec_path = _rates_path(symbol, config.execution_tf)
            rates_anchor = _read_rates_csv(anchor_path)
            rates_exec = _read_rates_csv(exec_path)
            if rates_anchor.empty or rates_exec.empty:
                _log(
                    f"{symbol}: missing/empty rates "
                    f"anchor_rows={len(rates_anchor)} exec_rows={len(rates_exec)} "
                    f"anchor_file={os.path.basename(anchor_path)} exec_file={os.path.basename(exec_path)}"
                )
                continue

            if config.bars_limit is not None and config.bars_limit > 0:
                rates_anchor = rates_anchor.iloc[-config.bars_limit :]
                rates_exec = rates_exec.iloc[-config.bars_limit :]

            df = generate_signals_refined(
                rates_anchor,
                rates_exec,
                anchor_swing_window=config.anchor_swing_window,
                execution_swing_window=config.execution_swing_window,
                entry_retracement=config.entry_retracement,
                sweep_mode=config.sweep_mode,
                internal_structure_lookback_bars=config.internal_structure_lookback_bars,
                max_bos_wait_bars=config.max_bos_wait_bars,
                max_pending_bars=config.max_pending_bars,
            )

            setup = _latest_setup(df)
            if setup is None:
                if last_sent_setup.get(symbol) == "CANCEL":
                    continue
                _log(
                    f"{symbol}: no setup "
                    f"anchor_last={rates_anchor.index[-1].isoformat()} exec_last={rates_exec.index[-1].isoformat()}"
                )
                cmd = {
                    "ts": cycle_ts,
                    "symbol": symbol,
                    "cmd": "CANCEL_ALL",
                    "dir": "",
                    "entry": "",
                    "sl": "",
                    "tp": "",
                    "risk_usd": "",
                    "magic": str(config.magic),
                    "replace_tol_points": str(config.replace_tol_points),
                    "max_pending_bars": str(config.max_pending_bars),
                    "client_tag": "py",
                }
                _append_command(cmd)
                last_sent_setup[symbol] = "CANCEL"
                continue

            setup_id, setup_dir, entry, sl, tp, bos_time = setup

            def get_last_bos(sym):
                path = os.path.join(common_dir, f"last_bos_{sym}.txt")
                if os.path.exists(path):
                    with open(path, "r") as f:
                        content = f.read().strip()
                        if content:
                            try:
                                return pd.Timestamp(content)
                            except Exception:
                                pass
                return pd.NaT

            def set_last_bos(sym, t):
                path = os.path.join(common_dir, f"last_bos_{sym}.txt")
                with open(path, "w") as f:
                    f.write(str(t))

            last_bos = get_last_bos(symbol)
            if not pd.isna(last_bos) and bos_time == last_bos:
                # We have already traded this exact historical setup.
                continue

            # Check for Exhaustion (Late Boot Syndrome):
            # If the bot is started late, verify the setup hasn't ALREADY triggered in the past.
            exhausted = False
            if not pd.isna(bos_time):
                check_df = rates_exec[rates_exec.index > bos_time]
                for _, r in check_df.iterrows():
                    if setup_dir == 1 and r['low'] <= entry:
                        exhausted = True
                        break
                    elif setup_dir == -1 and r['high'] >= entry:
                        exhausted = True
                        break
            
            if exhausted:
                _log(f"{symbol}: IGNORING setup_id={setup_id} - Already triggered/exhausted historically at bos_time={bos_time}")
                set_last_bos(symbol, bos_time)
                last_sent_setup[symbol] = setup_id
                continue
                
            _log(
                f"{symbol}: NEW SETUP setup_id={setup_id} dir={setup_dir} entry={entry:.5f} sl={sl:.5f} tp={tp:.5f} "
                f"bos_time={bos_time} anchor_last={rates_anchor.index[-1].isoformat()} exec_last={rates_exec.index[-1].isoformat()}"
            )
            
            # Persist to disk so that EA Bridge restarts don't re-trigger it
            set_last_bos(symbol, bos_time)
            last_sent_setup[symbol] = setup_id

            opened_dir = setup_dir
            sl_level_target = sl
            tp_level_target = tp
            
            if config.invert_signals:
                opened_dir = -setup_dir
                sl_level_target = tp
                tp_level_target = sl

            risk_usd = float(config.risk_base_balance_usd) * float(config.risk_per_trade_pct)
            cmd = {
                "ts": cycle_ts,
                "symbol": symbol,
                "cmd": "PLACE_LIMIT",
                "dir": str(1 if opened_dir == 1 else -1),
                "entry": f"{entry:.10f}",
                "sl": f"{sl_level_target:.10f}",
                "tp": f"{tp_level_target:.10f}",
                "risk_usd": f"{risk_usd:.2f}",
                "magic": str(config.magic),
                "replace_tol_points": str(config.replace_tol_points),
                "max_pending_bars": str(config.max_pending_bars),
                "client_tag": f"py_setup_{setup_id}",
            }
            _append_command(cmd)

        _log("Cycle complete. Sleeping...")
        time.sleep(int(config.poll_interval_sec))


if __name__ == "__main__":
    run_bridge(BridgeConfig())
