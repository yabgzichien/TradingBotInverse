"""
mc_optimizer.py — Monte Carlo + Walk-Forward Parameter Optimizer
=================================================================
For each WFO segment:
  1. Grid-search 108 strategy param combos on In-Sample data.
  2. Select the best combo by composite (Sharpe + MC + Return + DD + PropFirm).
  3. Run the best combo on Out-of-Sample data (blind test).
  4. Run 500 Monte Carlo sims on OOS trades to measure robustness.

Outputs:
  - mc_optimization/all_is_results_{symbol}.csv  (all 108 combos per segment)
  - mc_optimization/wfo_summary.csv              (per-segment WFO results)
  - mc_optimization/rankings_{symbol}.csv        (top-10 per metric)
  - mc_optimization/mc_distribution_{symbol}.png
  - mc_optimization/wfo_equity_curve.png
  - mc_optimization/param_heatmap_{symbol}.png
  - mc_optimization/rankings_chart_{symbol}.png
"""

import os
import sys
import io
import itertools
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from datetime import datetime, timedelta

from data_loader import initialize_mt5, get_data, get_symbol_specs
from strategy import generate_signals_refined
from backtest import run_backtest

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
SYMBOLS           = ["BTCUSD", "XAUUSD"]
HTF               = "H4"
ETF               = "M15"
INITIAL_BALANCE   = 10_000.0
RISK_PER_TRADE    = 0.01          # 1% fixed risk
LOOKBACK_DAYS     = 2 * 365
MC_SIMS           = 300

# Walk-Forward windows
TRAIN_MONTHS      = 5
TEST_MONTHS       = 2

# HTF warm-up so H4 swings are fully initialised
HTF_WARMUP_DAYS   = 60
ETF_WARMUP_DAYS   = 4

# Strategy parameter search space  (4 × 3 × 3 × 3 = 108 combos)
FIB_LEVELS            = [0.5, 0.618, 0.7, 0.786]
HTF_SWING_WINDOWS     = [5, 7, 10, 13]
LOOKBACK_BARS_LIST    = [1, 3, 5, 7]
BOS_WAIT_BARS_LIST    = [4, 8, 12, 16]

OUTPUT_DIR = os.path.join("backtest_results", "mc_optimization")


# ── Logging ────────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Silent backtest ────────────────────────────────────────────────────────────
def _silent_backtest(strategy_df, symbol: str) -> pd.DataFrame:
    """Run run_backtest with all prints suppressed."""
    specs = get_symbol_specs(symbol) or {}
    point_value    = float(specs.get("point")  or 0.01)
    spread_points  = float(specs.get("spread") or 0.0)

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        trades_df = run_backtest(
            strategy_df,
            initial_balance=INITIAL_BALANCE,
            risk_per_trade=RISK_PER_TRADE,
            spread_points=spread_points,
            slippage_points=5,
            commission_per_unit=0.07,
            point_value=point_value,
            symbol=symbol,
            save_csv=False,
            invert_signals=True,
        )
    finally:
        sys.stdout = old_stdout
    return trades_df


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(trades_df: pd.DataFrame) -> dict | None:
    """
    Reconstruct equity curve from trades and compute all key metrics.
    Returns None if no trades.
    """
    if trades_df is None or trades_df.empty:
        return None

    balance = INITIAL_BALANCE
    equity_points = [{"time": trades_df.iloc[0]["entry_time"], "equity": INITIAL_BALANCE}]
    for _, t in trades_df.iterrows():
        balance += t["pnl"]
        equity_points.append({"time": t["exit_time"], "equity": balance})

    eq_df = (
        pd.DataFrame(equity_points)
        .set_index("time")
        .sort_index()
    )
    eq_df = eq_df[~eq_df.index.duplicated(keep="last")]

    total_trades = len(trades_df)
    wins         = (trades_df["result"] == "Win").sum()
    win_rate     = wins / total_trades

    total_return_pct = (balance - INITIAL_BALANCE) / INITIAL_BALANCE

    # Annualised return
    # Guard against negative balance: a negative base raised to a fractional
    # exponent produces a complex number in NumPy, which poisons downstream
    # DataFrame operations (nlargest / nsmallest raise TypeError: complex128).
    span_days = max((eq_df.index[-1] - eq_df.index[0]).days, 1)
    balance_ratio = balance / INITIAL_BALANCE
    if balance_ratio <= 0:
        ann_return = -1.0   # treat as total loss
    else:
        ann_return = float(balance_ratio ** (365 / span_days) - 1)

    # Sharpe
    daily = eq_df["equity"].resample("D").last().dropna()
    dr    = daily.pct_change().dropna()
    std   = dr.std()
    sharpe = float(np.sqrt(252) * dr.mean() / std) if std > 0 else 0.0

    # Max Drawdown
    peak   = eq_df["equity"].cummax()
    dd     = (eq_df["equity"] - peak) / peak
    max_dd = float(dd.min())

    # Prop Firm Challenge
    profit_target = INITIAL_BALANCE * 0.15
    dd_limit      = INITIAL_BALANCE * 0.08
    prop_passes = prop_fails = 0
    baseline = INITIAL_BALANCE
    for ep in equity_points:
        v = ep["equity"]
        if v >= baseline + profit_target:
            prop_passes += 1
            baseline = v
        elif v <= baseline - dd_limit:
            prop_fails += 1
            baseline = v

    # Profit Factor
    gp = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    gl = -trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()
    pf = (gp / gl) if gl > 0 else float("inf")

    return {
        "total_trades":     total_trades,
        "win_rate":         win_rate,
        "total_pnl":        trades_df["pnl"].sum(),
        "total_return_pct": total_return_pct,
        "annual_return_pct": ann_return,
        "sharpe_ratio":     sharpe,
        "max_drawdown_pct": max_dd,
        "prop_firm_passes": prop_passes,
        "prop_firm_fails":  prop_fails,
        "profit_factor":    pf,
        "final_balance":    balance,
    }


# ── Monte Carlo ────────────────────────────────────────────────────────────────
def run_monte_carlo(trades_df: pd.DataFrame, n_sims: int = MC_SIMS) -> dict | None:
    """Shuffle-with-replacement MC simulation on trade PnLs."""
    if trades_df is None or trades_df.empty:
        return None

    pnls = trades_df["pnl"].values
    n    = len(pnls)

    finals = np.empty(n_sims)
    for i in range(n_sims):
        finals[i] = INITIAL_BALANCE + np.random.choice(pnls, size=n, replace=True).sum()

    returns = (finals - INITIAL_BALANCE) / INITIAL_BALANCE
    return {
        "final_balances": finals,
        "returns":        returns,
        "median_return":  float(np.median(returns)),
        "p5_return":      float(np.percentile(returns, 5)),
        "p95_return":     float(np.percentile(returns, 95)),
        "prob_profit":    float(np.mean(returns > 0)),
        "mean_return":    float(np.mean(returns)),
    }


# ── Composite Score ────────────────────────────────────────────────────────────
def composite_score(row: dict) -> float:
    def _real(v, default=0):
        """Return the real part of v (guards against accidental complex values)."""
        val = v if v is not None else default
        val = val or default
        return float(np.real(val))

    sharpe      = _real(row.get("sharpe_ratio"))
    prob_profit = _real(row.get("mc_prob_profit"))
    ann_ret     = _real(row.get("annual_return_pct"))
    p5_ret      = _real(row.get("mc_p5_return"))
    prop_passes = _real(row.get("prop_firm_passes"))
    max_dd      = abs(_real(row.get("max_drawdown_pct")))

    return float(
        sharpe      * 0.30
        + prob_profit * 0.25
        + ann_ret     * 0.20
        + p5_ret      * 0.15
        + (prop_passes * 0.01) * 0.10
        - max_dd      * 0.10
    )


# ── Strategy runner ────────────────────────────────────────────────────────────
def run_strategy(htf_df, etf_df, fib, htfw, lookback, bos_wait):
    return generate_signals_refined(
        htf_df,
        etf_df,
        anchor_swing_window=htfw,
        execution_swing_window=1,
        entry_retracement=fib,
        sweep_mode="prev_bar",
        internal_structure_lookback_bars=lookback,
        max_bos_wait_bars=bos_wait,
        max_pending_bars=96,
    )


# ── Visualisations ─────────────────────────────────────────────────────────────
def plot_mc_distribution(mc_result: dict, symbol: str, param_label: str):
    """Histogram of MC final returns with key percentiles."""
    if mc_result is None:
        return
    returns = mc_result["returns"] * 100   # percent
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(returns, bins=50, color="#2962FF", alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axvline(mc_result["p5_return"]   * 100, color="#FF5252", lw=2, ls="--", label=f"5th Pct: {mc_result['p5_return']*100:.1f}%")
    ax.axvline(mc_result["median_return"]* 100, color="#00BCD4", lw=2, ls="--", label=f"Median: {mc_result['median_return']*100:.1f}%")
    ax.axvline(mc_result["p95_return"]  * 100, color="#4CAF50", lw=2, ls="--", label=f"95th Pct: {mc_result['p95_return']*100:.1f}%")
    ax.axvline(0, color="white", lw=1.2, alpha=0.5)
    ax.set_facecolor("#1E222D")
    fig.patch.set_facecolor("#131722")
    ax.tick_params(colors="#D1D4DC")
    ax.spines[["top","right","bottom","left"]].set_color("#2A2E39")
    ax.set_title(f"Monte Carlo Return Distribution — {symbol}\n{param_label} | N={MC_SIMS} sims",
                 color="#D1D4DC", fontsize=12, pad=12)
    ax.set_xlabel("Total Return (%)", color="#D1D4DC")
    ax.set_ylabel("Frequency",        color="#D1D4DC")
    ax.legend(facecolor="#1E222D", labelcolor="#D1D4DC", framealpha=0.8)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"mc_distribution_{symbol}.png")
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    log(f"  Saved {path}")


def plot_wfo_equity_curve(oos_trades_by_symbol: dict):
    """Cumulative OOS equity curve per symbol."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor("#1E222D")
    fig.patch.set_facecolor("#131722")

    colours = {"BTCUSD": "#26A69A", "XAUUSD": "#FFD700"}

    for sym, trades_list in oos_trades_by_symbol.items():
        if not trades_list:
            continue
        all_trades = pd.concat(trades_list, ignore_index=True).sort_values("exit_time")
        balance = INITIAL_BALANCE
        times, balances = [all_trades.iloc[0]["entry_time"]], [balance]
        for _, t in all_trades.iterrows():
            balance += t["pnl"]
            times.append(t["exit_time"])
            balances.append(balance)
        ax.plot(times, balances, color=colours.get(sym, "cyan"), lw=1.8, label=f"{sym} OOS Equity")

    ax.axhline(INITIAL_BALANCE, color="white", lw=0.8, alpha=0.4, ls="--")
    ax.tick_params(colors="#D1D4DC")
    ax.spines[["top","right","bottom","left"]].set_color("#2A2E39")
    ax.set_title("Walk-Forward Out-of-Sample Equity Curve", color="#D1D4DC", fontsize=13, pad=12)
    ax.set_xlabel("Date",    color="#D1D4DC")
    ax.set_ylabel("Balance", color="#D1D4DC")
    ax.legend(facecolor="#1E222D", labelcolor="#D1D4DC", framealpha=0.8)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "wfo_equity_curve.png")
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    log(f"  Saved {path}")


def plot_param_heatmap(results_df: pd.DataFrame, symbol: str):
    """2-D heatmap: Fib × HTF window coloured by Sharpe ratio."""
    if results_df.empty:
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pivot = results_df.pivot_table(values="sharpe_ratio", index="fib", columns="htf_swing_window", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#131722")
    ax.set_facecolor("#1E222D")

    cmap = cm.get_cmap("RdYlGn")
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="#D1D4DC")
    cbar.set_label("Mean Sharpe Ratio (IS)", color="#D1D4DC")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([str(c) for c in pivot.columns], color="#D1D4DC")
    ax.set_yticklabels([f"{r:.3f}" for r in pivot.index], color="#D1D4DC")
    ax.set_xlabel("HTF Swing Window", color="#D1D4DC")
    ax.set_ylabel("Fib Entry Level",  color="#D1D4DC")
    ax.set_title(f"Parameter Heatmap — {symbol} (Sharpe Ratio)", color="#D1D4DC", fontsize=12, pad=10)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i][j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="black")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"param_heatmap_{symbol}.png")
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    log(f"  Saved {path}")


def plot_rankings_chart(results_df: pd.DataFrame, symbol: str):
    """Bar charts showing top-10 combos by each key metric."""
    if results_df.empty:
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics = [
        ("composite_score",    "Composite Score",    "#2962FF"),
        ("sharpe_ratio",       "Sharpe Ratio",       "#00BCD4"),
        ("annual_return_pct",  "Annual Return",      "#4CAF50"),
        ("max_drawdown_pct",   "Max Drawdown (worst→best)", "#FF5252"),
        ("prop_firm_passes",   "Prop Firm Passes",   "#FFD700"),
        ("mc_prob_profit",     "MC Prob of Profit",  "#7E57C2"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("#131722")
    fig.suptitle(f"Parameter Rankings — {symbol}", color="#D1D4DC", fontsize=14, y=1.01)
    axes = axes.flatten()

    for ax, (col, title, colour) in zip(axes, metrics):
        if col not in results_df.columns:
            ax.set_visible(False)
            continue
        ascending = (col == "max_drawdown_pct")   # lower drawdown is better
        top10 = results_df.nsmallest(10, col) if ascending else results_df.nlargest(10, col)
        labels = [
            f"F{r['fib']:.2f} H{r['htf_swing_window']} L{r['lookback_bars']} B{r['bos_wait_bars']}"
            for _, r in top10.iterrows()
        ]
        values = top10[col].values
        if ascending:
            values = values   # show as-is; more negative = worse drawn
        bars = ax.barh(labels[::-1], values[::-1], color=colour, alpha=0.85)
        ax.set_facecolor("#1E222D")
        ax.tick_params(colors="#D1D4DC", labelsize=8)
        ax.spines[["top","right","bottom","left"]].set_color("#2A2E39")
        ax.set_title(title, color="#D1D4DC", fontsize=10, pad=8)
        ax.set_xlabel(col, color="#D1D4DC", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"rankings_chart_{symbol}.png")
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved {path}")


# ── Ranking CSV ────────────────────────────────────────────────────────────────
def save_ranking_tables(results_df: pd.DataFrame, symbol: str):
    """Save top-10 tables by each metric as one multi-sheet-style CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ranking_specs = [
        ("composite_score",   "by_composite",   False),
        ("sharpe_ratio",      "by_sharpe",       False),
        ("annual_return_pct", "by_annual_return",False),
        ("max_drawdown_pct",  "by_drawdown",     True),   # ascending: least bad
        ("prop_firm_passes",  "by_propfirm",     False),
        ("mc_prob_profit",    "by_mc_prob",      False),
        ("mc_p5_return",      "by_mc_worst_case",False),
    ]
    frames = []
    for col, label, asc in ranking_specs:
        if col not in results_df.columns:
            continue
        top = (results_df.nsmallest(10, col) if asc else results_df.nlargest(10, col)).copy()
        top.insert(0, "ranking_by", label)
        top.insert(1, "rank", range(1, len(top) + 1))
        frames.append(top)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        path = os.path.join(OUTPUT_DIR, f"rankings_{symbol}.csv")
        combined.to_csv(path, index=False, float_format="%.5f")
        log(f"  Rankings saved to {path}")


# ── WFO + MC Main Loop ────────────────────────────────────────────────────────
def run_optimization(symbol: str, htf_data: pd.DataFrame, etf_data: pd.DataFrame):
    """
    Full WFO + MC pipeline for one symbol.
    Returns:
        all_is_rows    — every IS combo result across all segments
        wfo_rows       — per-segment WFO summary
        oos_trades_list— OOS trades_df per segment (for equity curve)
    """
    combos = list(itertools.product(FIB_LEVELS, HTF_SWING_WINDOWS, LOOKBACK_BARS_LIST, BOS_WAIT_BARS_LIST))
    total_combos = len(combos)

    all_is_rows     = []
    wfo_rows        = []
    oos_trades_list = []

    segment = 1
    train_start = etf_data.index.min()
    data_end    = etf_data.index.max()

    while True:
        train_end = train_start + pd.DateOffset(months=TRAIN_MONTHS)
        test_end  = train_end  + pd.DateOffset(months=TEST_MONTHS)

        if train_end >= data_end:
            break

        log(f"[{symbol}] Segment {segment} — IS: {train_start.date()} → {train_end.date()} | OOS: → {min(test_end, data_end).date()}")

        # ── Slice IS data ──────────────────────────────────────────────────────
        htf_is = htf_data[(htf_data.index >= train_start - pd.Timedelta(days=HTF_WARMUP_DAYS)) & (htf_data.index < train_end)]
        etf_is = etf_data[(etf_data.index >= train_start - pd.Timedelta(days=ETF_WARMUP_DAYS)) & (etf_data.index < train_end)]

        best_combo   = None
        best_score   = -np.inf
        best_is_row  = {}

        # ── Grid search on IS ──────────────────────────────────────────────────
        for idx, (fib, htfw, lookback, bos_wait) in enumerate(combos, 1):
            sys.stdout.write(f"\r  [{symbol}] IS Search: {idx}/{total_combos} combos")
            sys.stdout.flush()

            try:
                strat_df = run_strategy(htf_is, etf_is, fib, htfw, lookback, bos_wait)
                strat_df = strat_df[strat_df.index >= train_start]

                trades_df = _silent_backtest(strat_df, symbol)
                metrics   = compute_metrics(trades_df)
                if metrics is None:
                    continue
                mc_result = run_monte_carlo(trades_df)
                mc_fields = {} if mc_result is None else {
                    "mc_prob_profit":  mc_result["prob_profit"],
                    "mc_median_return":mc_result["median_return"],
                    "mc_p5_return":    mc_result["p5_return"],
                    "mc_p95_return":   mc_result["p95_return"],
                    "mc_mean_return":  mc_result["mean_return"],
                }

                row = {
                    "segment":            segment,
                    "symbol":             symbol,
                    "fib":                fib,
                    "htf_swing_window":   htfw,
                    "lookback_bars":      lookback,
                    "bos_wait_bars":      bos_wait,
                    **metrics,
                    **mc_fields,
                }
                row["composite_score"] = composite_score(row)
                all_is_rows.append(row)

                if row["composite_score"] > best_score and metrics["total_trades"] > 0:
                    best_score = row["composite_score"]
                    best_combo = (fib, htfw, lookback, bos_wait)
                    best_is_row = row

            except Exception:
                pass

        print()  # newline after progress bar

        if best_combo is None:
            log(f"  [!] No profitable IS combos found. Skipping segment.")
            wfo_rows.append({
                "segment": segment, "symbol": symbol,
                "train_start": train_start, "train_end": train_end,
                "test_end": min(test_end, data_end),
                "best_fib": None,
            })
            train_start += pd.DateOffset(months=TEST_MONTHS)
            segment += 1
            continue

        log(f"  Best IS: Fib={best_combo[0]} HTF={best_combo[1]} LB={best_combo[2]} BOS={best_combo[3]} | Score={best_score:.4f} | Sharpe={best_is_row.get('sharpe_ratio',0):.2f}")

        # ── OOS blind test ─────────────────────────────────────────────────────
        real_test_end = min(test_end, data_end)
        htf_oos = htf_data[(htf_data.index >= train_end - pd.Timedelta(days=HTF_WARMUP_DAYS)) & (htf_data.index <= real_test_end)]
        etf_oos = etf_data[(etf_data.index >= train_end - pd.Timedelta(days=ETF_WARMUP_DAYS)) & (etf_data.index <= real_test_end)]

        try:
            fib, htfw, lookback, bos_wait = best_combo
            strat_oos = run_strategy(htf_oos, etf_oos, fib, htfw, lookback, bos_wait)
            strat_oos = strat_oos[strat_oos.index >= train_end]
            oos_trades = _silent_backtest(strat_oos, symbol)
            oos_metrics = compute_metrics(oos_trades)
            oos_mc      = run_monte_carlo(oos_trades)
        except Exception as e:
            log(f"  [!] OOS failed: {e}")
            oos_trades = pd.DataFrame()
            oos_metrics = {}
            oos_mc = None

        if not (oos_trades is None or oos_trades.empty):
            oos_trades_list.append(oos_trades)

        wfo_rows.append({
            "segment":       segment,
            "symbol":        symbol,
            "train_start":   train_start,
            "train_end":     train_end,
            "test_end":      real_test_end,
            "best_fib":      best_combo[0],
            "best_htfw":     best_combo[1],
            "best_lookback": best_combo[2],
            "best_bos_wait": best_combo[3],
            "is_composite":  best_score,
            "is_sharpe":     best_is_row.get("sharpe_ratio", 0),
            "is_trades":     best_is_row.get("total_trades", 0),
            "is_pnl":        best_is_row.get("total_pnl", 0),
            "oos_trades":    (oos_metrics or {}).get("total_trades", 0),
            "oos_pnl":       (oos_metrics or {}).get("total_pnl", 0),
            "oos_sharpe":    (oos_metrics or {}).get("sharpe_ratio", 0),
            "oos_ann_ret":   (oos_metrics or {}).get("annual_return_pct", 0),
            "oos_max_dd":    (oos_metrics or {}).get("max_drawdown_pct", 0),
            "oos_prop_passes":(oos_metrics or {}).get("prop_firm_passes", 0),
            "oos_mc_prob":   (oos_mc or {}).get("prob_profit", None),
            "oos_mc_p5":     (oos_mc or {}).get("p5_return", None),
        })

        log(f"  OOS Result: {(oos_metrics or {}).get('total_trades',0)} trades | PnL ${(oos_metrics or {}).get('total_pnl',0):.2f} | Sharpe {(oos_metrics or {}).get('sharpe_ratio',0):.2f}")

        # MC plot uses OOS trades of best combo for this symbol
        if oos_mc and segment == 1:
            label = f"Fib={fib:.3f} HTF={htfw} LB={lookback} BOS={bos_wait}"
            plot_mc_distribution(oos_mc, symbol, label)

        train_start += pd.DateOffset(months=TEST_MONTHS)
        segment += 1

    return all_is_rows, wfo_rows, oos_trades_list


# ── Console Report ─────────────────────────────────────────────────────────────
def print_wfo_report(wfo_df: pd.DataFrame):
    print("\n" + "=" * 100)
    print("  WALK-FORWARD OPTIMIZATION RESULTS")
    print("=" * 100)
    for _, row in wfo_df.iterrows():
        best = row.get("best_fib")
        sym  = row.get("symbol", "")
        seg  = int(row.get("segment", 0))
        ts   = row.get("train_start"); te = row.get("train_end"); oe = row.get("test_end")
        if best is None:
            print(f"  [{sym}] Seg {seg:02d} | IS: {ts} → {te} | OOS: → {oe}  → NO profitable combo found.")
        else:
            print(
                f"  [{sym}] Seg {seg:02d} | IS: {ts} → {te} | OOS: → {oe}\n"
                f"          Best: Fib={row['best_fib']} HTF={row['best_htfw']} LB={row['best_lookback']} BOS={row['best_bos_wait']}\n"
                f"          IS  → PnL ${row['is_pnl']:>8.2f} ({int(row['is_trades'])} trades) | Sharpe {row['is_sharpe']:.2f}\n"
                f"          OOS → PnL ${row['oos_pnl']:>8.2f} ({int(row['oos_trades'])} trades) | Sharpe {row.get('oos_sharpe',0):.2f} | MC Prob {row.get('oos_mc_prob',0) or 0:.0%}"
            )
        print("-" * 100)


def print_top_rankings(results_df: pd.DataFrame, symbol: str):
    metrics_cfg = [
        ("composite_score",   "COMPOSITE SCORE",     False),
        ("sharpe_ratio",      "SHARPE RATIO",        False),
        ("annual_return_pct", "ANNUAL RETURN",       False),
        ("max_drawdown_pct",  "MAX DRAWDOWN (least worst)", True),
        ("prop_firm_passes",  "PROP FIRM PASSES",   False),
        ("mc_prob_profit",    "MC PROB OF PROFIT",  False),
    ]
    print(f"\n{'='*90}")
    print(f"  PARAMETER RANKINGS — {symbol}")
    print(f"{'='*90}")
    for col, title, asc in metrics_cfg:
        if col not in results_df.columns:
            continue
        top5 = results_df.nsmallest(5, col) if asc else results_df.nlargest(5, col)
        print(f"\n  📊 Top-5 by {title}")
        print(f"  {'#':>3}  {'Fib':>6}  {'HTF':>4}  {'LB':>3}  {'BOS':>4}  {col:>22}")
        for rank, (_, r) in enumerate(top5.iterrows(), 1):
            v = r[col]
            fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"  {rank:>3}  {r['fib']:>6.3f}  {r['htf_swing_window']:>4}  {r['lookback_bars']:>3}  {r['bos_wait_bars']:>4}  {fmt:>22}")


# ── Entry Point ────────────────────────────────────────────────────────────────
def main():
    if not initialize_mt5():
        return

    import MetaTrader5 as mt5
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_wfo_rows = []
    all_oos_trades_by_symbol = {}

    for symbol in SYMBOLS:
        log(f"─── Fetching data for {symbol} (last {LOOKBACK_DAYS} days) ───")

        # Use MT5 broker time as end reference
        tick = mt5.symbol_info_tick(symbol)
        end_date   = datetime.utcfromtimestamp(tick.time) if tick else datetime.utcnow()
        start_date = end_date - timedelta(days=LOOKBACK_DAYS)

        htf_start = start_date - timedelta(days=HTF_WARMUP_DAYS)
        etf_start = start_date - timedelta(days=ETF_WARMUP_DAYS)

        htf_data = get_data(symbol, HTF, htf_start, end_date)
        etf_data = get_data(symbol, ETF, etf_start, end_date)

        if htf_data.empty or etf_data.empty:
            log(f"[ERROR] Could not fetch data for {symbol}. Skipping.")
            continue

        log(f"  HTF rows: {len(htf_data)} | ETF rows: {len(etf_data)}")

        # Run WFO + MC
        is_rows, wfo_rows, oos_trades_list = run_optimization(symbol, htf_data, etf_data)

        all_oos_trades_by_symbol[symbol] = oos_trades_list
        all_wfo_rows.extend(wfo_rows)

        if not is_rows:
            log(f"[!] No IS results for {symbol}.")
            continue

        # Build IS results dataframe for this symbol
        is_df = pd.DataFrame(is_rows)
        is_df["composite_score"] = is_df.apply(composite_score, axis=1)

        # Save all IS results
        all_is_path = os.path.join(OUTPUT_DIR, f"all_is_results_{symbol}.csv")
        is_df.to_csv(all_is_path, index=False, float_format="%.5f")
        log(f"All IS results saved → {all_is_path}")

        # Save ranking tables
        save_ranking_tables(is_df, symbol)

        # Charts
        log(f"Generating charts for {symbol}...")
        plot_param_heatmap(is_df, symbol)
        plot_rankings_chart(is_df, symbol)

        # Console top rankings
        print_top_rankings(is_df, symbol)

    # WFO Summary CSV
    if all_wfo_rows:
        wfo_df = pd.DataFrame(all_wfo_rows)
        wfo_path = os.path.join(OUTPUT_DIR, "wfo_summary.csv")
        wfo_df.to_csv(wfo_path, index=False, float_format="%.5f")
        log(f"\nWFO summary saved → {wfo_path}")
        print_wfo_report(wfo_df)

    # Combined OOS equity curve
    if any(v for v in all_oos_trades_by_symbol.values()):
        log("Generating WFO equity curve...")
        plot_wfo_equity_curve(all_oos_trades_by_symbol)

    log("\n✅ Monte Carlo Optimization complete.")
    log(f"All outputs saved in: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()