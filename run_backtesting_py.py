import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from backtesting import Backtest, Strategy

from data_loader import get_data, initialize_mt5
from strategy import generate_signals_refined

# Global variable to hold user configuration for inverse mode
INVERT_SIGNALS = True
MAX_PENDING_BARS = 96

class FibInvertedStrategy(Strategy):
    """
    Backtesting.py Strategy subclass that replicates the EA Bridge logic.
    It reads predefined setup signals from the DataFrame columns and correctly
    places Stop or Limit orders based on the Inverse Geometry.
    """
    
    def init(self):
        # We don't need any complex indicators here because all signals
        # and levels are pre-calculated and embedded directly into self.data.
        # We track state manually to avoid duplicates.
        self.used_setup_ids = set()
        
        # Track our manually placed pending order to enforce the max pending bars limit
        self.pending_setup_id = None
        self.pending_bar_count = 0
        
    def next(self):
        # 1. Manage Active Pending Orders (Cancel if expired)
        if self.pending_setup_id is not None:
            self.pending_bar_count += 1
            if self.pending_bar_count > MAX_PENDING_BARS:
                # Cancel all pending orders in Backtesting.py
                self.orders.cancel()
                self.pending_setup_id = None
                self.pending_bar_count = 0
                
        # 2. Check if a position is open; if so, we ignore new setups just like main backtest
        if len(self.trades) > 0:
            return
            
        # 3. Read current row's setup signals
        # self.data variables are accessed by their current index [-1]
        sid = self.data.setup_id[-1]
        
        if pd.notna(sid) and sid not in self.used_setup_ids:
            # We have a brand new setup
            sdir = int(self.data.setup_dir[-1])
            entry_level = float(self.data.entry_level[-1])
            sl_level = float(self.data.sl_level[-1])
            tp_level = float(self.data.tp_level[-1])
            
            # Apply Inverse Logic identically to ea_bridge_inversed.py
            if INVERT_SIGNALS:
                opened_dir = -sdir
                sl_target = tp_level
                tp_target = sl_level
            else:
                opened_dir = sdir
                sl_target = sl_level
                tp_target = tp_level
                
            current_price = self.data.Close[-1]
            
            # Cancel any existing pending order to replace with the new one
            if len(self.orders) > 0:
                self.orders.cancel()
            
            # Risk Sizing (Simplified constant size for Backtesting.py)
            size = 0.1 # Example constraint
            
            # Place Order matching MT5 logic (Limit vs Stop)
            if opened_dir == 1: # LONG
                if entry_level > current_price:
                    self.buy(stop=entry_level, sl=sl_target, tp=tp_target)
                else:
                    self.buy(limit=entry_level, sl=sl_target, tp=tp_target)
            else: # SHORT
                if entry_level < current_price:
                    self.sell(stop=entry_level, sl=sl_target, tp=tp_target)
                else:
                    self.sell(limit=entry_level, sl=sl_target, tp=tp_target)
                    
            # Mark setup as processed
            self.used_setup_ids.add(sid)
            self.pending_setup_id = sid
            self.pending_bar_count = 0

def run_backtesting_engine(symbol="BTCUSD", anchor_tf="H4", exec_tf="M5"):
    print(f"--- Running Backtesting.py Engine: {symbol} ---")
    
    if not initialize_mt5():
        return
        
    # 1. Fetch Data
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=60) # 60 days backtest
    
    htf_data = get_data(symbol, anchor_tf, start_date, end_date)
    sig_data = get_data(symbol, "M15", start_date, end_date)
    xtf_data = get_data(symbol, exec_tf, start_date, end_date)
    
    if htf_data.empty or sig_data.empty or xtf_data.empty:
        print("Data missing. Exiting.")
        return

    # 2. Generate Signals on M15
    print("Generating M15 Setup Signals...")
    strategy_df = generate_signals_refined(
        htf_data,
        sig_data,
        anchor_swing_window=7,
        execution_swing_window=1,
        entry_retracement=0.618,
        sweep_mode="prev_bar",
        internal_structure_lookback_bars=1,
        max_bos_wait_bars=8
    )
    
    # 3. Timeframe Decoupling - Map M15 setups to M5 execution dataset
    # We must shift the M15 setup dataset forward by exactly 15 minutes,
    # because the signal generated on the 10:00 M15 bar (which closes at 10:15)
    # is only ACTUALLY actionable AT 10:15.
    strategy_df['actionable_time'] = strategy_df.index + pd.Timedelta(minutes=15)
    setup_cols = ['actionable_time', 'setup_id', 'setup_dir', 'entry_level', 'sl_level', 'tp_level']
    setup_df = strategy_df.dropna(subset=['setup_id'])[setup_cols]
    
    # Capitalize execution dataframe columns for Backtesting.py
    xtf_data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    }, inplace=True)
    
    # Merge signals onto the M5 execution tape
    print("Merging Signals onto Execution timeframe...")
    merged_df = pd.merge_asof(
        xtf_data.reset_index(),
        setup_df,
        left_on='time',
        right_on='actionable_time',
        direction='backward',
        tolerance=pd.Timedelta(minutes=0) # Only exact matches get the signal flag
    )
    merged_df.set_index('time', inplace=True)
    
    # Forward-fill NA for setups? No, we only want the trigger to fire ONCE on the precise execution candle.
    # The 'tolerance=0' guarantees it only appears exactly at 10:15 M5 candle.
    
    # 4. Run Backtesting.py
    print("Running Simulator...")
    # Increase cash heavily because Backtesting.py default doesn't support fractional units 
    # of BTC out of the box (requires > price cash for 1 whole BTC).
    bt = Backtest(merged_df, FibInvertedStrategy, cash=1000000, commission=.0002, margin=1.0)
    
    stats = bt.run()
    
    print("\n--- Backtest Results ---")
    print(stats)
    print("\n-------------------------")
    
    # Check if trades happened
    if stats['_trades'].empty:
        print("No trades found!")
    else:
        print(f"Total Trades: {len(stats['_trades'])}")
        
    try:
        # Automatically generate interactive HTML chart
        output_file = os.path.abspath(f"backtesting_py_{symbol}_inversed.html")
        print(f"Generating optimized chart at {output_file}...")
        
        # Override bokeh formatter issue
        from backtesting import set_bokeh_output
        set_bokeh_output(notebook=False)
        
        bt.plot(filename=output_file, open_browser=False, show_legend=True)
        print("Chart generation complete!")
    except Exception as e:
        print(f"Warning: Failed to plot chart: {e}")


if __name__ == "__main__":
    # You can easily change symbols here
    run_backtesting_engine("BTCUSD", "H4", "M5")
