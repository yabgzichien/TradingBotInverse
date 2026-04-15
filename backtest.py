import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_results(df, trades_df, symbol):
    """
    Generates an interactive HTML chart using TradingView's Lightweight Charts.
    Shows candlesticks with trade entry/exit markers and connecting lines.
    """
    if trades_df.empty:
        print("No trades to plot.")
        return

    # Prepare candlestick data as list of dicts
    candles = []
    for ts, row in df.iterrows():
        candles.append({
            'time': int(ts.timestamp()),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
        })

    # Prepare markers for entries and exits
    markers = []
    for _, trade in trades_df.iterrows():
        entry_ts = int(trade['entry_time'].timestamp())
        exit_ts = int(trade['exit_time'].timestamp())
        is_long = trade['type'] == 'Long'
        is_win = trade['result'] == 'Win'

        # Entry marker
        markers.append({
            'time': entry_ts,
            'position': 'belowBar' if is_long else 'aboveBar',
            'color': '#26a69a' if is_long else '#ef5350',
            'shape': 'arrowUp' if is_long else 'arrowDown',
            'text': f"{'BUY' if is_long else 'SELL'} @ {trade['entry']:.2f}",
        })

        # Exit marker
        exit_color = '#4caf50' if is_win else '#f44336'
        markers.append({
            'time': exit_ts,
            'position': 'aboveBar' if is_long else 'belowBar',
            'color': exit_color,
            'shape': 'circle',
            'text': f"{'TP' if is_win else 'SL'} @ {trade['exit']:.2f}",
        })

    # Sort markers by time (required by Lightweight Charts)
    markers.sort(key=lambda m: m['time'])

    # Prepare trade lines data (entry-to-exit connections)
    trade_lines = []
    for _, trade in trades_df.iterrows():
        is_win = trade['result'] == 'Win'
        trade_lines.append({
            'entry_time': int(trade['entry_time'].timestamp()),
            'entry_price': float(trade['entry']),
            'exit_time': int(trade['exit_time'].timestamp()),
            'exit_price': float(trade['exit']),
            'color': '#4caf50' if is_win else '#f44336',
            'type': trade['type'],
            'result': trade['result'],
            'pnl': float(trade['pnl']),
        })

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{symbol} Backtest Results</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #131722;
            color: #d1d4dc;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            overflow: hidden;
        }}
        #header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 20px;
            background: #1e222d;
            border-bottom: 1px solid #2a2e39;
        }}
        #header h1 {{
            font-size: 18px;
            font-weight: 600;
            color: #e0e3eb;
        }}
        #header h1 span {{
            color: #2962ff;
        }}
        .stats {{
            display: flex;
            gap: 20px;
            font-size: 12px;
        }}
        .stat {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .stat-label {{
            color: #787b86;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 10px;
        }}
        .stat-value {{
            font-weight: 600;
            font-size: 14px;
        }}
        .stat-value.green {{ color: #26a69a; }}
        .stat-value.red {{ color: #ef5350; }}
        .stat-value.blue {{ color: #2962ff; }}
        #legend {{
            display: flex;
            gap: 16px;
            padding: 8px 20px;
            background: #1e222d;
            border-bottom: 1px solid #2a2e39;
            font-size: 11px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}
        #chart-container {{
            width: 100%;
            height: calc(100vh - 90px);
        }}
        #tooltip {{
            position: absolute;
            display: none;
            background: #1e222d;
            border: 1px solid #2a2e39;
            border-radius: 6px;
            padding: 10px 14px;
            font-size: 12px;
            color: #d1d4dc;
            z-index: 1000;
            pointer-events: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1><span>&#9679;</span> {symbol} — Backtest Results</h1>
        <div class="stats">
            <div class="stat">
                <span class="stat-label">Trades</span>
                <span class="stat-value blue">{len(trades_df)}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Win Rate</span>
                <span class="stat-value green">{len(trades_df[trades_df['result']=='Win'])}/{len(trades_df)} ({len(trades_df[trades_df['result']=='Win'])/len(trades_df)*100:.1f}%)</span>
            </div>
            <div class="stat">
                <span class="stat-label">Total PnL</span>
                <span class="stat-value {'green' if trades_df['pnl'].sum() >= 0 else 'red'}">${trades_df['pnl'].sum():,.2f}</span>
            </div>
        </div>
    </div>
    <div id="legend">
        <div class="legend-item"><div class="legend-dot" style="background:#26a69a"></div> Long Entry</div>
        <div class="legend-item"><div class="legend-dot" style="background:#ef5350"></div> Short Entry</div>
        <div class="legend-item"><div class="legend-dot" style="background:#4caf50"></div> Take Profit</div>
        <div class="legend-item"><div class="legend-dot" style="background:#f44336"></div> Stop Loss</div>
        <div class="legend-item"><div class="legend-dot" style="background:#4caf50; border-radius:0"></div> Win Line</div>
        <div class="legend-item"><div class="legend-dot" style="background:#f44336; border-radius:0"></div> Loss Line</div>
    </div>
    <div id="chart-container"></div>
    <div id="tooltip"></div>

    <script>
        const candleData = {json.dumps(candles)};
        const markerData = {json.dumps(markers)};
        const tradeLines = {json.dumps(trade_lines)};

        const container = document.getElementById('chart-container');
        const chart = LightweightCharts.createChart(container, {{
            layout: {{
                background: {{ type: 'solid', color: '#131722' }},
                textColor: '#d1d4dc',
            }},
            grid: {{
                vertLines: {{ color: '#1e222d' }},
                horzLines: {{ color: '#1e222d' }},
            }},
            crosshair: {{
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: {{ color: '#2962ff33', width: 1, style: 0 }},
                horzLine: {{ color: '#2962ff33', width: 1, style: 0 }},
            }},
            rightPriceScale: {{
                borderColor: '#2a2e39',
            }},
            timeScale: {{
                borderColor: '#2a2e39',
                timeVisible: true,
                secondsVisible: false,
            }},
        }});

        // Candlestick series
        const candleSeries = chart.addCandlestickSeries({{
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderUpColor: '#26a69a',
            borderDownColor: '#ef5350',
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        }});
        candleSeries.setData(candleData);
        candleSeries.setMarkers(markerData);

        // Draw trade connection lines using line series
        tradeLines.forEach(function(trade) {{
            const lineSeries = chart.addLineSeries({{
                color: trade.color,
                lineWidth: 2,
                lineStyle: trade.result === 'Win' ? 0 : 2,
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            }});
            lineSeries.setData([
                {{ time: trade.entry_time, value: trade.entry_price }},
                {{ time: trade.exit_time, value: trade.exit_price }},
            ]);
        }});

        chart.timeScale().fitContent();

        // Resize handler
        window.addEventListener('resize', () => {{
            chart.applyOptions({{
                width: container.clientWidth,
                height: container.clientHeight,
            }});
        }});
    </script>
</body>
</html>"""

    os.makedirs("backtest_results", exist_ok=True)
    output_file = f"backtest_results/backtest_plot_{symbol}.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Chart saved to {output_file}")


def _bar_sequence(o, h, l):
    """
    Returns a 3-step price path for a bar using an OHLC heuristic to break the
    intra-bar ambiguity (i.e. did the bar hit the High or the Low first?).

    Heuristic: whichever extreme is closer to the bar's Open price is assumed
    to be visited first.  This is conservative and directionally neutral.

    Returns: (open, first_extreme, second_extreme)
    """
    dist_to_high = abs(o - h)
    dist_to_low  = abs(o - l)
    if dist_to_high <= dist_to_low:
        return (o, h, l)   # Open → High → Low
    else:
        return (o, l, h)   # Open → Low → High


def run_backtest(
    df,
    initial_balance=10000.0,
    risk_per_trade=0.01,
    fib_level=0.618,
    spread_points=0,
    slippage_points=0,
    commission_per_unit=0,
    point_value=0.01,
    use_refined_setups=True,
    max_pending_bars=96,
    replace_pending_on_new_setup=True,
    verbose=False,
    risk_base="initial",
    min_sl_points=0.0,
    both_hit_policy="sl",
    allow_entry_bar_tp=True,
    symbol="UNKNOWN",
    return_events=False,
    save_csv=True,
    invert_signals=False
):
    """
    Runs an iterrows backtest with realistic costs (spread, slippage, commission).

    Intra-bar illusion is handled via _bar_sequence(), which guesses the H/L
    visit order from the bar's Open price.  Limit orders that are gapped-through
    on the open are filled at the Open price (not the limit) to avoid an
    unrealistic fill-at-limit advantage.
    """
    balance = initial_balance
    trades = []
    events = []
    
    # State tracking
    in_position = False
    entry_price = 0
    sl_price = 0
    tp_price = 0
    position_type = 0 # 1 for Long, -1 for Short
    entry_time = None
    position_size = 0
    equity_curve = []
    pending = False
    pending_setup_id = None
    pending_dir = 0
    pending_entry = np.nan
    pending_sl = np.nan
    pending_tp = np.nan
    pending_since_i = None
    pending_active_from_i = None
    used_setup_ids = set()
    
    # Realistic costs in price terms
    entry_cost_points = (spread_points + slippage_points) * point_value
    exit_cost_points = slippage_points * point_value
    
    if verbose:
        print(f"Starting backtest with {len(df)} candles. Costs: Spread={spread_points}, Slippage={slippage_points}, Commission=${commission_per_unit}/unit")
    
    for i, (index, row) in enumerate(df.iterrows()):
        
        # Check if we need to close the current position
        if in_position:
            bar_o = row['open']
            bar_h = row['high']
            bar_l = row['low']
            path_o, path_first, path_second = _bar_sequence(bar_o, bar_h, bar_l)
            
            if position_type == 1: # Long
                sl_hit = bar_l <= sl_price
                tp_hit = bar_h >= tp_price
                
                if sl_hit and tp_hit:
                    # Chronological check
                    if path_first == bar_l:
                        exit_at_sl = True # Hit SL bottom first
                    else:
                        exit_at_sl = False # Hit TP top first
                else:
                    exit_at_sl = sl_hit
                    
                if exit_at_sl:
                    execution_exit = sl_price - exit_cost_points
                    pnl = (execution_exit - entry_price) * position_size - (commission_per_unit * position_size)
                    balance += pnl
                    trades.append({'entry_time': entry_time, 'exit_time': index, 'type': 'Long', 'entry': entry_price, 'tp_price': tp_price, 'sl_price': sl_price, 'exit': execution_exit, 'pnl': pnl, 'result': 'Loss'})
                    in_position = False
                elif tp_hit:
                    execution_exit = tp_price - exit_cost_points
                    pnl = (execution_exit - entry_price) * position_size - (commission_per_unit * position_size)
                    balance += pnl
                    trades.append({'entry_time': entry_time, 'exit_time': index, 'type': 'Long', 'entry': entry_price, 'tp_price': tp_price, 'sl_price': sl_price, 'exit': execution_exit, 'pnl': pnl, 'result': 'Win'})
                    in_position = False
                    
            elif position_type == -1: # Short
                sl_hit = bar_h >= sl_price
                tp_hit = bar_l <= tp_price
                
                if sl_hit and tp_hit:
                    # Chronological check
                    if path_first == bar_h:
                        exit_at_sl = True # Hit SL top first
                    else:
                        exit_at_sl = False # Hit TP bottom first
                else:
                    exit_at_sl = sl_hit
                    
                if exit_at_sl:
                    execution_exit = sl_price + exit_cost_points
                    pnl = (entry_price - execution_exit) * position_size - (commission_per_unit * position_size)
                    balance += pnl
                    trades.append({'entry_time': entry_time, 'exit_time': index, 'type': 'Short', 'entry': entry_price, 'tp_price': tp_price, 'sl_price': sl_price, 'exit': execution_exit, 'pnl': pnl, 'result': 'Loss'})
                    in_position = False
                elif tp_hit:
                    execution_exit = tp_price + exit_cost_points
                    pnl = (entry_price - execution_exit) * position_size - (commission_per_unit * position_size)
                    balance += pnl
                    trades.append({'entry_time': entry_time, 'exit_time': index, 'type': 'Short', 'entry': entry_price, 'tp_price': tp_price, 'sl_price': sl_price, 'exit': execution_exit, 'pnl': pnl, 'result': 'Win'})
                    in_position = False

        if not in_position:
            if use_refined_setups and ('entry_level' in df.columns) and ('setup_id' in df.columns) and ('setup_dir' in df.columns):
                sid = row.get('setup_id', np.nan)
                sdir = row.get('setup_dir', np.nan)
                entry_level = row.get('entry_level', np.nan)
                sl_level = row.get('sl_level', np.nan)
                tp_level = row.get('tp_level', np.nan)

                if (
                    pd.notna(sid)
                    and pd.notna(sdir)
                    and pd.notna(entry_level)
                    and pd.notna(sl_level)
                    and pd.notna(tp_level)
                    and (sid not in used_setup_ids)
                    and (replace_pending_on_new_setup or not pending)
                    and ((not pending) or (pending_setup_id != sid))
                ):
                    pending = True
                    pending_setup_id = sid
                    pending_dir = int(sdir)
                    pending_entry = float(entry_level)
                    pending_sl = float(sl_level)
                    pending_tp = float(tp_level)
                    pending_since_i = i
                    pending_active_from_i = i + 1
                    used_setup_ids.add(sid)

            if pending and (pending_since_i is not None) and (max_pending_bars is not None) and ((i - pending_since_i) > max_pending_bars):
                pending = False
                pending_setup_id = None
                pending_since_i = None
                pending_active_from_i = None

            if pending and not in_position and (pending_active_from_i is None or i >= pending_active_from_i):
                bar_o = row['open']
                bar_h = row['high']
                bar_l = row['low']
                path_o, path_first, path_second = _bar_sequence(bar_o, bar_h, bar_l)

                # Evaluator for trigger condition purely based on original pending_dir
                is_triggered = False
                trigger_path_first = None
                actual_fill = pending_entry

                if pending_dir == 1:  # Original Long
                    if bar_l <= pending_entry:
                        is_triggered = True
                        actual_fill = bar_o if bar_o <= pending_entry else pending_entry
                        trigger_path_first = bar_l
                else:  # Original Short
                    if bar_h >= pending_entry:
                        is_triggered = True
                        actual_fill = bar_o if bar_o >= pending_entry else pending_entry
                        trigger_path_first = bar_h

                if is_triggered:
                    opened_dir = pending_dir
                    sl_level_target = pending_sl
                    tp_level_target = pending_tp
                    
                    if invert_signals:
                        opened_dir = -pending_dir
                        sl_level_target = pending_tp
                        tp_level_target = pending_sl

                    original_execution_entry = actual_fill + entry_cost_points if pending_dir == 1 else actual_fill - entry_cost_points

                    if opened_dir == 1:
                        execution_entry = original_execution_entry
                        sl_dist = execution_entry - sl_level_target
                        if sl_dist > 0:
                            base_balance = initial_balance if risk_base == "initial" else balance
                            min_sl_dist = float(min_sl_points) * point_value
                            safe_sl_dist = max(sl_dist, min_sl_dist) if min_sl_dist > 0 else sl_dist
                            position_size = (base_balance * risk_per_trade) / safe_sl_dist
                            in_position = True
                            position_type = 1
                            entry_price = execution_entry
                            sl_price = sl_level_target
                            tp_price = tp_level_target
                            entry_time = index
                    else:
                        execution_entry = original_execution_entry
                        sl_dist = sl_level_target - execution_entry
                        if sl_dist > 0:
                            base_balance = initial_balance if risk_base == "initial" else balance
                            min_sl_dist = float(min_sl_points) * point_value
                            safe_sl_dist = max(sl_dist, min_sl_dist) if min_sl_dist > 0 else sl_dist
                            position_size = (base_balance * risk_per_trade) / safe_sl_dist
                            in_position = True
                            position_type = -1
                            entry_price = execution_entry
                            sl_price = sl_level_target
                            tp_price = tp_level_target
                            entry_time = index
                            
                    pending = False
                    pending_setup_id = None
                    pending_since_i = None
                    pending_active_from_i = None

                    if in_position:
                        # ── Chronological Intra-bar check ─────────────
                        sl_reachable = False
                        tp_reachable = False
                        
                        bar_c = row['close']
                        if position_type == 1:
                            sl_hit = bar_l <= sl_price
                            tp_hit = bar_h >= tp_price
                            if trigger_path_first == bar_l:
                                sl_reachable = sl_hit
                                tp_reachable = tp_hit and (path_first == bar_l or bar_o <= pending_entry or bar_c >= tp_price)
                            else:
                                sl_reachable = sl_hit and (path_first == bar_h or bar_o >= pending_entry or bar_c <= sl_price)
                                tp_reachable = tp_hit
                        else:  # Short
                            sl_hit = bar_h >= sl_price
                            tp_hit = bar_l <= tp_price
                            if trigger_path_first == bar_l:
                                sl_reachable = sl_hit and (path_first == bar_l or bar_o <= pending_entry or bar_c >= sl_price)
                                tp_reachable = tp_hit
                            else:
                                sl_reachable = sl_hit
                                tp_reachable = tp_hit and (path_first == bar_h or bar_o >= pending_entry or bar_c <= tp_price)

                        if not allow_entry_bar_tp:
                            tp_reachable = False
                            
                        # If both hit in the entry bar, the chronologically FIRST target wins!
                        if sl_reachable and tp_reachable:
                            # Both are reachable! We determine which came first via path sequence.
                            if position_type == 1:
                                # Long: SL is downside, TP is upside
                                if path_first == bar_l:
                                    # Hit low first, so SL hit first
                                    sl_hit_first = True
                                else:
                                    sl_hit_first = False
                            else:
                                # Short: SL is upside, TP is downside
                                if path_first == bar_h:
                                    # Hit high first, so SL hit first
                                    sl_hit_first = True
                                else:
                                    sl_hit_first = False
                                    
                            if sl_hit_first:
                                forced_result = 'Loss'
                            else:
                                forced_result = 'Win'
                        elif sl_reachable:
                            forced_result = 'Loss'
                        elif tp_reachable:
                            forced_result = 'Win'
                        else:
                            forced_result = None

                        if forced_result == 'Loss':
                            execution_exit = (sl_price - exit_cost_points) if position_type == 1 else (sl_price + exit_cost_points)
                            pnl = (execution_exit - entry_price) * position_size - (commission_per_unit * position_size) if position_type == 1 else (entry_price - execution_exit) * position_size - (commission_per_unit * position_size)
                            balance += pnl
                            trades.append({'entry_time': entry_time, 'exit_time': index, 'type': 'Long' if position_type == 1 else 'Short', 'entry': entry_price, 'tp_price': tp_price, 'sl_price': sl_price, 'exit': execution_exit, 'pnl': pnl, 'result': 'Loss'})
                            in_position = False
                        elif forced_result == 'Win':
                            execution_exit = (tp_price - exit_cost_points) if position_type == 1 else (tp_price + exit_cost_points)
                            pnl = (execution_exit - entry_price) * position_size - (commission_per_unit * position_size) if position_type == 1 else (entry_price - execution_exit) * position_size - (commission_per_unit * position_size)
                            balance += pnl
                            trades.append({'entry_time': entry_time, 'exit_time': index, 'type': 'Long' if position_type == 1 else 'Short', 'entry': entry_price, 'tp_price': tp_price, 'sl_price': sl_price, 'exit': execution_exit, 'pnl': pnl, 'result': 'Win'})
                            in_position = False
            elif (not pending) and (not in_position):
                htf_trend = row.get('htf_trend', 0)
                if pd.isna(row.get('last_swing_high', np.nan)) or pd.isna(row.get('last_swing_low', np.nan)):
                    pass
                else:
                    sh = row['last_swing_high']
                    sl = row['last_swing_low']
                    swing_range = sh - sl
                    if swing_range > 0:
                        if htf_trend == 1:
                            entry_level = sh - (swing_range * fib_level)
                            if row['low'] <= entry_level and row['open'] > entry_level:
                                execution_entry = entry_level + entry_cost_points
                                sl_dist = execution_entry - sl
                                if sl_dist > 0:
                                    base_balance = initial_balance if risk_base == "initial" else balance
                                    min_sl_dist = float(min_sl_points) * point_value
                                    safe_sl_dist = max(sl_dist, min_sl_dist) if min_sl_dist > 0 else sl_dist
                                    position_size = (base_balance * risk_per_trade) / safe_sl_dist
                                    in_position = True
                                    position_type = 1
                                    entry_price = execution_entry
                                    sl_price = sl
                                    tp_price = sh
                                    entry_time = index
                        elif htf_trend == -1:
                            entry_level = sl + (swing_range * fib_level)
                            if row['high'] >= entry_level and row['open'] < entry_level:
                                execution_entry = entry_level - entry_cost_points
                                sl_dist = sh - execution_entry
                                if sl_dist > 0:
                                    base_balance = initial_balance if risk_base == "initial" else balance
                                    min_sl_dist = float(min_sl_points) * point_value
                                    safe_sl_dist = max(sl_dist, min_sl_dist) if min_sl_dist > 0 else sl_dist
                                    position_size = (base_balance * risk_per_trade) / safe_sl_dist
                                    in_position = True
                                    position_type = -1
                                    entry_price = execution_entry
                                    sl_price = sh
                                    tp_price = sl
                                    entry_time = index

        # Track floating equity
        floating_pnl = 0
        if in_position:
            if position_type == 1:
                floating_pnl = (row['close'] - entry_price) * position_size
            elif position_type == -1:
                floating_pnl = (entry_price - row['close']) * position_size
        equity_curve.append({'time': index, 'equity': balance + floating_pnl})

        if return_events:
            events.append({
                'time': int(index.timestamp()),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'equity': float(balance + floating_pnl),
                'balance': float(balance),
                'initial_balance': float(initial_balance),
                'pending': {
                    'active': bool(pending),
                    'dir': int(pending_dir) if pending else 0,
                    'entry': float(pending_entry) if pending else None,
                    'sl': float(pending_sl) if pending else None,
                    'tp': float(pending_tp) if pending else None
                },
                'position': {
                    'active': bool(in_position),
                    'type': int(position_type) if in_position else 0,
                    'entry': float(entry_price) if in_position else None,
                    'sl': float(sl_price) if in_position else None,
                    'tp': float(tp_price) if in_position else None
                }
            })

    # Close any open position at the end
    if in_position:
        last_price = df.iloc[-1]['close']
        if position_type == 1:
            execution_exit = float(last_price) - exit_cost_points
            pnl = (execution_exit - entry_price) * position_size - (commission_per_unit * position_size)
        else:
            execution_exit = float(last_price) + exit_cost_points
            pnl = (entry_price - execution_exit) * position_size - (commission_per_unit * position_size)
        balance += pnl
        trades.append({
            'entry_time': entry_time,
            'exit_time': df.index[-1],
            'type': 'Long' if position_type == 1 else 'Short',
            'entry': entry_price,
            'tp_price': np.nan,
            'sl_price': np.nan,
            'exit': execution_exit,
            'pnl': pnl,
            'result': 'Open/Closed at End'
        })

    trades_df = pd.DataFrame(trades)

    # Compute trade durations and exit reasons
    if not trades_df.empty:
        trades_df['duration'] = trades_df['exit_time'] - trades_df['entry_time']
        trades_df['duration_minutes'] = trades_df['duration'].dt.total_seconds() / 60.0
        trades_df['exit_reason'] = trades_df['result'].map({
            'Win': 'TP',
            'Loss': 'SL'
        }).fillna('End')
    
    # Calculate metrics
    if not trades_df.empty:
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['result'] == 'Win'])
        long_trades = len(trades_df[trades_df['type'] == 'Long'])
        short_trades = len(trades_df[trades_df['type'] == 'Short'])
        win_rate = wins / total_trades if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum()
        avg_trade_duration = trades_df['duration'].mean()
        
        equity_df = pd.DataFrame(equity_curve).set_index('time')
        peak = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Identify distinct drawdown episodes > 10%
        in_dd = (drawdown < -0.10)
        dd_starts = in_dd & ~in_dd.shift(1, fill_value=False)
        dd_ends = ~in_dd & in_dd.shift(1, fill_value=False)
        
        start_times = drawdown.index[dd_starts]
        end_times = drawdown.index[dd_ends]
        
        # If still in a drawdown at the end, count it
        if in_dd.iloc[-1]:
            end_times = end_times.append(pd.DatetimeIndex([drawdown.index[-1]]))
        
        num_dd_episodes = len(start_times)
        dd_episodes = []
        for s, e in zip(start_times, end_times):
            episode = drawdown.loc[s:e]
            worst = episode.min()
            duration = e - s
            dd_episodes.append({'start': s, 'end': e, 'worst': worst, 'duration': duration})
        
        daily_equity = equity_df['equity'].resample('D').last().dropna()
        daily_returns = daily_equity.pct_change().dropna()
        
        daily_std = daily_returns.std()
        annualized_std = daily_std * np.sqrt(252) if daily_std != 0 else 0.0
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_std) if daily_std != 0 else 0.0
        
        # New Metrics: Total Return
        total_return_pct = (balance - initial_balance) / initial_balance

        gross_profit = trades_df.loc[trades_df['pnl'] > 0, 'pnl'].sum()
        gross_loss = -trades_df.loc[trades_df['pnl'] < 0, 'pnl'].sum()
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        # Max consecutive take profit wins in a row
        max_consecutive_tp_wins = 0
        current_streak = 0
        for result in trades_df['result']:
            if result == 'Win':
                current_streak += 1
                max_consecutive_tp_wins = max(max_consecutive_tp_wins, current_streak)
            else:
                current_streak = 0
        
        # New Metrics: Prop Firm Style (+15% of initial cap, without -8% of initial cap loss)
        # e.g. for $10k initial: win condition is gaining $1500, lose condition is losing $800.
        profit_target_amount = initial_balance * 0.15
        drawdown_limit_amount = initial_balance * 0.08
        
        prop_firm_passes = 0
        prop_firm_fails = 0
        current_baseline = initial_balance
        
        for eq in equity_curve:
            val = eq['equity']
            
            # Did we hit the profit target relative to current baseline?
            if val >= current_baseline + profit_target_amount:
                prop_firm_passes += 1
                current_baseline = val  # Reset baseline after target hit
                
            # Did we hit the drawdown limit relative to current baseline?
            elif val <= current_baseline - drawdown_limit_amount:
                prop_firm_fails += 1
                current_baseline = val  # Reset baseline after failure
        
        # Benchmark: Buy and Hold (Always Long from first close to last close)
        first_price = df.iloc[0]['close']
        last_price_close = df.iloc[-1]['close']
        bh_return_pct = (last_price_close - first_price) / first_price
        bot_vs_bh = total_return_pct - bh_return_pct

        print("\n--- Backtest Results ---")
        print(f"Total Trades: {total_trades}")
        print(f"Long Trades: {long_trades}")
        print(f"Short Trades: {short_trades}")
        print(f"Wins: {wins}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"Average Trade Duration: {avg_trade_duration}")
        print(f"Total Return: {total_return_pct:.2%}")
        print(f"Buy and Hold Return: {bh_return_pct:.2%}")
        print(f"Bot vs Buy & Hold: {bot_vs_bh:+.2%}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Ann. Std Dev: {annualized_std:.2%}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Num Trades / DD > 10% Episodes: {total_trades / num_dd_episodes if num_dd_episodes > 0 else float('inf'):.2f}")
        print(f"Drawdown >10% Episodes: {num_dd_episodes}")
        print(f"Prop Firm Challenge (+15% before -8%): {prop_firm_passes} Passes / {prop_firm_fails} Fails")
        print(f"Max Consecutive Take Profit Wins: {max_consecutive_tp_wins}")
        if dd_episodes:
            print("\n--- Drawdown >10% Details ---")
            for i, ep in enumerate(dd_episodes, 1):
                print(f"  Episode {i}: {ep['start']} to {ep['end']} | Worst: {ep['worst']:.2%} | Duration: {ep['duration']}")
        print(f"\nFinal Balance: ${balance:.2f}")
    else:
        print("No trades taken during the period.")

    # Save trade-by-trade details to CSV
    if save_csv and not trades_df.empty:
        export_cols = [
            'entry_time',
            'exit_time',
            'type',
            'entry',
            'tp_price',
            'sl_price',
            'exit',
            'result',
            'exit_reason',
            'duration',
            'duration_minutes',
            'pnl'
        ]
        os.makedirs("backtest_results", exist_ok=True)
        out_path = f"backtest_results/backtest_trades_{symbol}.csv"
        try:
            trades_df.to_csv(out_path, columns=export_cols, index=False)
            print(f"Trade details saved to {out_path}")
        except PermissionError:
            # Windows commonly locks CSVs when opened in Excel.
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            alt_path = f"backtest_results/backtest_trades_{symbol}_{ts}.csv"
            trades_df.to_csv(alt_path, columns=export_cols, index=False)
            print(
                f"Could not write to {out_path} (file may be open). "
                f"Trade details saved to {alt_path} instead."
            )


    if return_events:
        return trades_df, events
    return trades_df


def monte_carlo_simulation(trades_df, initial_balance=10000.0, n_sims=1000):
    """
    Runs a simple Monte Carlo simulation by randomly shuffling trade PnLs
    (with replacement) to generate a distribution of final returns.
    Returns an array of final balances.
    """
    if trades_df.empty:
        return np.array([])

    pnls = trades_df['pnl'].values
    n_trades = len(pnls)
    final_balances = []

    for _ in range(n_sims):
        balance = initial_balance
        sampled_pnls = np.random.choice(pnls, size=n_trades, replace=True)
        for pnl in sampled_pnls:
            balance += pnl
        final_balances.append(balance)

    return np.array(final_balances)


def plot_return_distribution(final_balances, initial_balance=10000.0, output_file="monte_carlo_returns.png"):
    """
    Plots and saves a histogram of Monte Carlo final returns.
    """
    if final_balances.size == 0:
        print("No Monte Carlo results to plot.")
        return

    returns = (final_balances - initial_balance) / initial_balance

    plt.figure(figsize=(8, 5))
    plt.hist(returns, bins=40, color="#2962ff", alpha=0.8, edgecolor="black")
    plt.title("Monte Carlo Final Return Distribution")
    plt.xlabel("Total Return")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    
    os.makedirs("backtest_results", exist_ok=True)
    full_output_path = os.path.join("backtest_results", output_file)
    plt.savefig(full_output_path, dpi=120)
    plt.close()
    print(f"Monte Carlo return distribution saved to {full_output_path}")
