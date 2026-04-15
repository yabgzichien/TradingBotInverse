from data_loader import initialize_mt5, get_data, get_symbol_specs
from strategy import generate_signals_refined
from backtest import run_backtest, plot_results, monte_carlo_simulation, plot_return_distribution
from datetime import datetime, timedelta
import pandas as pd
import pytz
import MetaTrader5 as mt5
import json
import os

def main():
    if not initialize_mt5():
        return

    print("Select pairs to backtest:")
    print("1. XAUUSD")
    print("2. BTCUSD")
    print("3. XAUUSD + BTCUSD")
    print("4. Others")
    choice = input("Enter choice (1-4) [default 3]: ").strip() or "3"
    
    target_symbols = []
    if choice == "1":
        target_symbols = ["XAUUSD"]
    elif choice == "2":
        target_symbols = ["BTCUSD"]
    elif choice == "3":
        target_symbols = ["XAUUSD", "BTCUSD"]
    elif choice == "4":
        custom = input("Enter symbols separated by commas: ")
        target_symbols = [s.strip().upper() for s in custom.split(",") if s.strip()]
    else:
        print("Invalid choice. Using default (XAUUSD + BTCUSD).")
        target_symbols = ["XAUUSD", "BTCUSD"]

    if not target_symbols:
        print("No valid symbols selected.")
        return

    print("\nSelect Date Range type:")
    print("1. Specific number of days back from today")
    print("2. Specific start and end dates (YYYY-MM-DD)")
    date_choice = input("Enter choice (1-2) [default 1]: ").strip() or "1"
    
    tick = mt5.symbol_info_tick(target_symbols[0])
    if tick:
        end_date = datetime.utcfromtimestamp(tick.time)
    else:
        end_date = datetime.utcnow() + timedelta(hours=3) # Fallback to GMT+3
        
    etf_start = end_date - timedelta(days=3)  # Default
    
    if date_choice == "1":
        try:
            days = int(input("Enter number of days backward (default 3): ") or 3)
            etf_start = end_date - timedelta(days=days)
        except ValueError:
            print("Invalid input, using default 3 days.")
    elif date_choice == "2":
        start_str = input("Enter Start Date (YYYY-MM-DD): ").strip()
        end_str = input("Enter End Date (YYYY-MM-DD) [Leave blank for today]: ").strip()
        
        try:
            etf_start = datetime.strptime(start_str, "%Y-%m-%d")
        except ValueError:
            print("Invalid start date format. Using default 3 days ago.")
            
        if end_str:
            try:
                end_date_parsed = datetime.strptime(end_str, "%Y-%m-%d")
                # Move to end of the specified day
                end_date = end_date_parsed.replace(hour=23, minute=59, second=59)
            except ValueError:
                print("Invalid end date format. Using today's current time.")
                
    want_replay = input("Generate Animated Visual Replay (Interactive Charting)? (y/n) [default n]: ").strip().lower() == 'y'
    
    # Load strategy configuration
    config_path = "strategy_config.json"
    strat_cfg = {"fib_level": 0.618, "htf_swing_window": 7, "lookback_bars": 1, "bos_wait_bars": 8}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                strat_cfg = json.load(f)
            print(f"\nLoaded strategy config: {strat_cfg}")
        except Exception as e:
            print(f"\nError loading {config_path}: {e}")
    else:
        print("\nstrategy_config.json not found, using default parameters.")

    anchor_tf = "H4"
    etf = "M15"
    htf_warmup_days = 60  # Extra lookback so H4 swings/trend are fully initialised
    etf_warmup_candles = 360  # 15M warmup candles (360 × 15min = 3.75 days)

    htf_start = etf_start - timedelta(days=htf_warmup_days)  # H4 fetches further back
    etf_warmup_td = timedelta(minutes=etf_warmup_candles * 15)
    etf_fetch_start = etf_start - etf_warmup_td  # Fetch extra M15 history for warmup

    all_trades_dfs = []

    for symbol in target_symbols:
        print(f"\n{'='*50}\nStarting backtest for {symbol}\n{'='*50}")

        print(f"Fetching {anchor_tf} data for {symbol} (with {htf_warmup_days}-day warm-up buffer)...")
        htf_data = get_data(symbol, anchor_tf, htf_start, end_date)

        print(f"Fetching {etf} data for {symbol} (with {etf_warmup_candles}-candle warm-up buffer)...")
        etf_data = get_data(symbol, etf, etf_fetch_start, end_date)

        if htf_data.empty or etf_data.empty:
            print(f"Error fetching data for {symbol}. Skipping.")
            continue

        specs = get_symbol_specs(symbol) or {}
        point_value = float(specs.get("point") or 0.01)
        spread_points = float(specs.get("spread") or 0.0)
    
        print("Generating Signals...")
        strategy_df = generate_signals_refined(
            htf_data,
            etf_data,
            anchor_swing_window=strat_cfg.get("htf_swing_window", 7),
            execution_swing_window=1,
            entry_retracement=strat_cfg.get("fib_level", 0.618),
            sweep_mode="prev_bar",
            internal_structure_lookback_bars=strat_cfg.get("lookback_bars", 1),
            max_bos_wait_bars=strat_cfg.get("bos_wait_bars", 8)
        )

        # Trim warmup: only keep rows from the actual backtest start onwards
        strategy_df = strategy_df[strategy_df.index >= pd.Timestamp(etf_start)]
        
        # Save a sample of the data to verify
        os.makedirs("backtest_results", exist_ok=True)
        strategy_df.tail(100).to_csv(f"backtest_results/strategy_data_tail_{symbol}.csv")
        
        print(f"\n--- Diagnostic: Anchor Bias Distribution in ETF data ({symbol}) ---")
        print(strategy_df['bias'].value_counts(dropna=False))
        print("----------------------------------------------------\n")

        print("Running Backtest...")
        result = run_backtest(
            strategy_df, 
            initial_balance=10000.0, 
            risk_per_trade=0.01, 
            spread_points=spread_points,
            slippage_points=5,
            commission_per_unit=0.07,
            point_value=point_value,
            symbol=symbol,
            return_events=want_replay,
            invert_signals=True
        )

        if want_replay:
            trades_df, events = result
            from visual_replay import generate_visual_replay
            print("Generating Visual Replay HTML...")
            generate_visual_replay(events, symbol)
        else:
            trades_df = result

        if not trades_df.empty:
            trades_df['symbol'] = symbol
            all_trades_dfs.append(trades_df)

        print("Creating Trade Plot...")
        plot_results(strategy_df, trades_df, symbol)

        # Monte Carlo simulation on trade PnLs
        print(f"Running Monte Carlo simulation for {symbol}...")
        final_balances = monte_carlo_simulation(trades_df, initial_balance=10000.0, n_sims=1000)
        # Note: If running many pairs, overriding monte carlo plot might hide earlier ones.
        # Plot distribution uses default name, which might overwrite. We will just plot it.
        try:
            plot_return_distribution(final_balances, initial_balance=10000.0, output_file=f"monte_carlo_returns_{symbol}.png")
        except Exception as e:
            print(f"Monte Carlo plot skipped: {e}")

    # Final combined statistics
    if len(target_symbols) > 1 and all_trades_dfs:
        print(f"\n\n{'*'*50}\nCOMBINED PORTFOLIO SUMMARY\n{'*'*50}")
        combined_df = pd.concat(all_trades_dfs, ignore_index=True)
        combined_df.sort_values('exit_time', inplace=True)
        
        total_pnl = combined_df['pnl'].sum()
        total_trades = len(combined_df)
        wins = len(combined_df[combined_df['result'] == 'Win'])
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        print(f"Total Combined Trades: {total_trades}")
        print(f"Total Combined Wins: {wins}")
        print(f"Combined Win Rate: {win_rate:.2%}")
        print(f"Total Gross PnL: ${total_pnl:.2f}")
        print(f"{'*'*50}\n")

if __name__ == "__main__":
    main()
