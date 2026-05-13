#!/usr/bin/env python3
"""
benchmark_all.py — PortfolioBench comprehensive testing & benchmarking suite.

Runs every layer of the framework and produces a formatted terminal report:
  1. Data integrity checks
  2. Alpha factor unit tests
  3. Portfolio pipeline unit tests
  4. Standalone portfolio pipeline benchmark (real data)
  5. Trading strategy backtests  (8 strategies × asset categories × timeframes)
  6. Portfolio strategy backtests (8 strategies × asset categories × timeframes)

Usage:
    python benchmark_all.py                 # full suite
    python benchmark_all.py --quick         # fast smoke-test (5m, crypto only)
    python benchmark_all.py --trading-only  # skip portfolio strategies
    python benchmark_all.py --portfolio-only # skip trading strategies
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# freqtrade lives in a git submodule; add its root so `import freqtrade` resolves.
_FT_ROOT = os.path.join(PROJECT_ROOT, "freqtrade")
if os.path.isdir(os.path.join(_FT_ROOT, "freqtrade")) and _FT_ROOT not in sys.path:
    sys.path.insert(0, _FT_ROOT)


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

DATA_DIR = os.path.join(PROJECT_ROOT, "user_data", "data", "usstock")

TRADING_STRATEGIES = [
    "EmaCrossStrategy",
    "MacdAdxStrategy",
    "RsiBollingerStrategy",
    "IchimokuCloudStrategy",
    "StochasticCciStrategy",
    "MlpSpeculativeStrategy",
    "PolymarketMomentumStrategy",
    "PolymarketMeanReversionStrategy",
    "ZHMeanReversionStrategy",
]

PORTFOLIO_STRATEGIES = [
    "ONS_Portfolio",
    "InverseVolatilityPortfolio",
    "MinimumVariancePortfolio",
    "BestSingleAssetPortfolio",
    "ExponentialGradientPortfolio",
    "MaxSharpePortfolio",
    "RiskParityPortfolio",
    "PolymarketPortfolio",
]

ASSET_CATEGORIES = {
    "crypto":  ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
    "stocks":  ["AAPL/USDT", "MSFT/USDT", "NVDA/USDT", "GOOG/USDT"],
    "indices": ["DJI/USDT", "FTSE/USDT", "GSPC/USDT"],
    "mixed":   ["BTC/USDT", "ETH/USDT", "AAPL/USDT", "MSFT/USDT", "DJI/USDT"],
}

TIMEFRAME_CONFIG = {
    "5m":  {"timerange": "20260101-20260108",  "label": "5-Minute"},
    "4h":  {"timerange": "20260101-20260131",  "label": "4-Hour"},
    "1d":  {"timerange": "20240101-20260131",  "label": "Daily"},
}


# ═══════════════════════════════════════════════════════════════════════════
# TERMINAL FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

class Colors:
    BOLD      = "\033[1m"
    DIM       = "\033[2m"
    GREEN     = "\033[92m"
    RED       = "\033[91m"
    YELLOW    = "\033[93m"
    CYAN      = "\033[96m"
    MAGENTA   = "\033[95m"
    WHITE     = "\033[97m"
    RESET     = "\033[0m"
    BG_GREEN  = "\033[42m"
    BG_RED    = "\033[41m"
    BG_CYAN   = "\033[46m"

C = Colors


def banner(text: str, char: str = "═", width: int = 80):
    line = char * width
    print(f"\n{C.CYAN}{C.BOLD}{line}")
    print(f"  {text.upper()}")
    print(f"{line}{C.RESET}\n")


def section(text: str, width: int = 80):
    line = "─" * width
    print(f"\n{C.MAGENTA}{line}")
    print(f"  {text}")
    print(f"{line}{C.RESET}")


def ok(msg: str):
    print(f"  {C.GREEN}✓{C.RESET} {msg}")


def fail(msg: str):
    print(f"  {C.RED}✗{C.RESET} {msg}")


def warn(msg: str):
    print(f"  {C.YELLOW}⚠{C.RESET} {msg}")


def info(msg: str):
    print(f"  {C.DIM}→{C.RESET} {msg}")


def elapsed_str(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.1f}s"


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════

def check_data_integrity() -> Dict[str, Any]:
    banner("Phase 1: Data Integrity Checks")
    results = {"passed": 0, "failed": 0, "skipped": 0, "details": []}

    # Check data directory exists
    if not os.path.isdir(DATA_DIR):
        fail(f"Data directory not found: {DATA_DIR}")
        results["failed"] += 1
        return results

    ok(f"Data directory exists: {DATA_DIR}")
    results["passed"] += 1

    # Count feather files
    import glob
    feather_files = glob.glob(os.path.join(DATA_DIR, "*.feather"))
    total_files = len(feather_files)
    if total_files == 0:
        fail("No feather files found")
        results["failed"] += 1
        return results

    ok(f"Found {total_files} feather files")
    results["passed"] += 1

    # Check for LFS pointers vs real data
    sample_path = os.path.join(DATA_DIR, "BTC_USDT-1d.feather")
    if os.path.isfile(sample_path):
        with open(sample_path, "rb") as f:
            header = f.read(20)
        if header.startswith(b"version "):
            warn("Files appear to be Git LFS pointers (not pulled)")
            results["skipped"] += 1
            results["details"].append("LFS pointers detected — run `git lfs pull`")
            return results
        ok("Files are real data (not LFS pointers)")
        results["passed"] += 1
    else:
        warn("BTC_USDT-1d.feather not found — cannot verify LFS status")
        results["skipped"] += 1

    # Validate schema on representative files
    import pandas as pd
    required_cols = {"date", "open", "high", "low", "close", "volume"}
    timeframes = ["5m", "4h", "1d"]
    sample_tickers = ["BTC", "ETH", "AAPL", "DJI"]

    for ticker in sample_tickers:
        for tf in timeframes:
            fname = f"{ticker}_USDT-{tf}.feather"
            fpath = os.path.join(DATA_DIR, fname)
            if not os.path.isfile(fpath):
                warn(f"Missing: {fname}")
                results["skipped"] += 1
                continue
            try:
                df = pd.read_feather(fpath)
                missing = required_cols - set(df.columns)
                if missing:
                    fail(f"{fname}: missing columns {missing}")
                    results["failed"] += 1
                elif len(df) == 0:
                    fail(f"{fname}: empty file")
                    results["failed"] += 1
                elif (df["close"] <= 0).any():
                    fail(f"{fname}: non-positive close prices")
                    results["failed"] += 1
                else:
                    ok(f"{fname}: {len(df):,} rows, schema valid")
                    results["passed"] += 1
            except Exception as e:
                fail(f"{fname}: {e}")
                results["failed"] += 1

    # Check naming convention
    bad_names = [os.path.basename(f) for f in feather_files if "_USDT-" not in os.path.basename(f)]
    if bad_names:
        fail(f"Non-standard filenames: {bad_names[:5]}")
        results["failed"] += 1
    else:
        ok("All filenames follow {TICKER}_USDT-{timeframe}.feather convention")
        results["passed"] += 1

    # Count unique tickers and timeframes
    tickers = set()
    tfs_found = set()
    for f in feather_files:
        base = os.path.basename(f).replace(".feather", "")
        parts = base.rsplit("-", 1)
        if len(parts) == 2:
            tickers.add(parts[0])
            tfs_found.add(parts[1])
    info(f"Universe: {len(tickers)} instruments × {len(tfs_found)} timeframes ({', '.join(sorted(tfs_found))})")
    results["details"].append(f"{len(tickers)} instruments × {len(tfs_found)} timeframes")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — UNIT TESTS (pytest)
# ═══════════════════════════════════════════════════════════════════════════

def run_unit_tests() -> Dict[str, Any]:
    banner("Phase 2: Unit Tests (pytest)")
    results = {"passed": 0, "failed": 0, "skipped": 0, "details": []}

    try:
        import pytest
    except ImportError:
        warn("pytest not installed — skipping unit tests")
        results["skipped"] += 1
        results["details"].append("pytest not available")
        return results

    test_dir = os.path.join(PROJECT_ROOT, "tests")
    test_files = [
        ("test_data_integrity.py", "Data integrity tests"),
        ("test_alpha.py", "Alpha factor tests"),
        ("test_portfolio_management.py", "Portfolio pipeline tests"),
    ]

    for filename, label in test_files:
        fpath = os.path.join(test_dir, filename)
        if not os.path.isfile(fpath):
            warn(f"{label}: file not found ({filename})")
            results["skipped"] += 1
            continue

        section(f"Running: {label}")
        t0 = time.time()
        try:
            exit_code = pytest.main([
                fpath, "-v", "--tb=short", "--no-header", "-q",
            ])
            dt = time.time() - t0
            if exit_code == 0:
                ok(f"{label} — all passed ({elapsed_str(dt)})")
                results["passed"] += 1
            elif exit_code == 5:
                warn(f"{label} — no tests collected ({elapsed_str(dt)})")
                results["skipped"] += 1
            else:
                fail(f"{label} — failures detected ({elapsed_str(dt)})")
                results["failed"] += 1
        except Exception as e:
            fail(f"{label} — error: {e}")
            results["failed"] += 1

    return results


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — STANDALONE PORTFOLIO PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_portfolio_pipeline() -> Dict[str, Any]:
    banner("Phase 3: Standalone Portfolio Pipeline")
    results = {"passed": 0, "failed": 0, "skipped": 0, "details": [], "metrics": {}}

    try:
        from portfolio.PortfolioManagement import run_portfolio
    except ImportError as e:
        warn(f"Cannot import portfolio pipeline: {e}")
        results["skipped"] += 1
        return results

    configs = [
        ("Crypto 2-asset",  ["BTC/USDT", "ETH/USDT"],                              "1d", 100_000),
        ("Crypto 4-asset",  ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],     "1d", 100_000),
        ("Mixed 5-asset",   ["BTC/USDT", "ETH/USDT", "AAPL/USDT", "MSFT/USDT", "DJI/USDT"], "1d", 1_000_000),
    ]

    for label, pairs, tf, capital in configs:
        section(f"Pipeline: {label} ({len(pairs)} assets, {tf})")
        t0 = time.time()
        try:
            result, weights, metrics = run_portfolio(
                data_dir=DATA_DIR, pairs=pairs, timeframe=tf, initial_capital=capital,
            )
            dt = time.time() - t0
            ok(f"Completed in {elapsed_str(dt)}")
            ok(f"Total return: {metrics['total_return_pct']:+.2f}%")
            ok(f"Sharpe ratio: {metrics['annualised_sharpe']:.4f}")
            ok(f"Max drawdown: {metrics['max_drawdown_pct']:.2f}%")
            ok(f"Bars: {metrics['n_bars']}")
            results["passed"] += 1
            results["metrics"][label] = metrics
        except Exception as e:
            dt = time.time() - t0
            fail(f"Failed after {elapsed_str(dt)}: {e}")
            results["failed"] += 1

    return results


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4 & 5 — FREQTRADE BACKTESTS
# ═══════════════════════════════════════════════════════════════════════════

def _extract_metrics(bt_results: Any, strategy_name: str) -> Dict[str, Any]:
    """Extract key performance metrics from a freqtrade backtest result."""
    metrics: Dict[str, Any] = {}
    if bt_results is None:
        return metrics

    try:
        # Navigate to the strategy-level result dict
        # bt.results has structure {"strategy": {name: stats}, "metadata": {...}, ...}
        if isinstance(bt_results, dict) and "strategy" in bt_results:
            strat_dict = bt_results["strategy"]
            if isinstance(strat_dict, dict) and strategy_name in strat_dict:
                sr = strat_dict[strategy_name]
            elif isinstance(strat_dict, dict):
                first_key = next(iter(strat_dict), None)
                sr = strat_dict.get(first_key, bt_results)
            else:
                sr = bt_results
        elif isinstance(bt_results, dict) and strategy_name in bt_results:
            sr = bt_results[strategy_name]
        elif isinstance(bt_results, dict):
            first_key = next(iter(bt_results), None)
            sr = bt_results.get(first_key, bt_results)
        else:
            return metrics

        if not isinstance(sr, dict):
            return metrics

        # Collect all candidate sub-dicts
        candidates = [sr]
        for sub_key in ["results_per_pair", "results", "strategy_comparison",
                        "backtest_results", "backtest_result"]:
            sub = sr.get(sub_key)
            if isinstance(sub, dict):
                candidates.append(sub)
            elif isinstance(sub, list):
                for entry in sub:
                    if isinstance(entry, dict):
                        candidates.append(entry)

        # Search for common metric keys across all candidates
        _METRIC_MAP = {
            "total_trades": ("trades", None),
            "trade_count": ("trades", None),
            "profit_total": ("total_return_pct", lambda v: round(v * 100, 2)),
            "profit_total_abs": ("profit_abs", lambda v: round(v, 2)),
            "max_drawdown": ("max_drawdown_pct", lambda v: round(v * 100, 2)),
            "max_drawdown_account": ("max_drawdown_pct", lambda v: round(v * 100, 2)),
            "sharpe": ("sharpe", lambda v: round(v, 4)),
            "sharpe_ratio": ("sharpe", lambda v: round(v, 4)),
            "sortino": ("sortino", lambda v: round(v, 4)),
            "sortino_ratio": ("sortino", lambda v: round(v, 4)),
            "calmar": ("calmar", lambda v: round(v, 4)),
            "winrate": ("win_rate_pct", lambda v: round(v * 100, 2)),
            "win_rate": ("win_rate_pct", lambda v: round(v * 100, 2)),
            "profit_factor": ("profit_factor", lambda v: round(v, 4)),
            "profit_mean": ("avg_profit_pct", lambda v: round(v * 100, 2)),
            "holding_avg": ("avg_duration", str),
            "duration_avg": ("avg_duration", str),
        }

        for d in candidates:
            for src_key, (dst_key, transform) in _METRIC_MAP.items():
                if dst_key in metrics:
                    continue
                val = d.get(src_key)
                if val is not None:
                    metrics[dst_key] = transform(val) if transform else val

        # Fallback: compute from trades DataFrame
        if "trades" not in metrics:
            trades_df = sr.get("trades")
            if trades_df is not None and hasattr(trades_df, "__len__"):
                metrics["trades"] = len(trades_df)

    except Exception:
        pass
    return metrics


def _run_single_backtest(
    strategy_name: str,
    strategy_path: str,
    pairs: List[str],
    timeframe: str,
    timerange: str,
    wallet: float = 1_000_000,
) -> Dict[str, Any]:
    """Run a single freqtrade backtest programmatically and return results."""
    from freqtrade.commands.optimize_commands import setup_optimize_configuration
    from freqtrade.enums import RunMode
    from freqtrade.optimize.backtesting import Backtesting

    config_path = os.path.join(PROJECT_ROOT, "user_data", "config.json")

    args = {
        "config": [config_path],
        "strategy": strategy_name,
        "strategy_path": strategy_path,
        "timerange": timerange,
        "timeframe": timeframe,
        "pairs": pairs,
        "dry_run_wallet": wallet,
    }

    config = setup_optimize_configuration(args, RunMode.BACKTEST)
    backtesting = Backtesting(config)

    try:
        backtesting.start()
        bt_results = backtesting.results
        # Extract key metrics from the result dict
        metrics = _extract_metrics(bt_results, strategy_name)
        return {"status": "ok", "metrics": metrics}
    finally:
        if backtesting.exchange:
            backtesting.exchange.close()


def _backtest_worker(
    strategy_name: str,
    strategy_path: str,
    pairs: List[str],
    timeframe: str,
    timerange: str,
    wallet: float,
) -> Dict[str, Any]:
    """Top-level function that can be pickled for ProcessPoolExecutor."""
    t0 = time.time()
    try:
        result = _run_single_backtest(
            strategy_name=strategy_name,
            strategy_path=strategy_path,
            pairs=pairs,
            timeframe=timeframe,
            timerange=timerange,
            wallet=wallet,
        )
        dt = time.time() - t0
        return {
            "status": "pass",
            "metrics": result.get("metrics", {}),
            "duration_s": round(dt, 1),
        }
    except Exception as e:
        dt = time.time() - t0
        return {
            "status": "fail",
            "error": str(e).split("\n")[0][:120],
            "duration_s": round(dt, 1),
        }


def run_freqtrade_backtests(
    strategies: List[str],
    strategy_path: str,
    phase_name: str,
    categories: Dict[str, List[str]],
    timeframes: Dict[str, Dict],
    wallet: float = 1_000_000,
    max_workers: int = 1,
) -> Dict[str, Any]:
    """Run backtests for a set of strategies across categories and timeframes.

    When *max_workers* > 1, backtests are dispatched to a process pool for
    concurrent execution.
    """
    banner(f"{phase_name}")
    results = {
        "passed": 0, "failed": 0, "skipped": 0,
        "details": [], "backtest_results": [],
    }

    # Build ordered task list
    tasks: List[Tuple[str, str, List[str], str, str]] = []
    for strat in strategies:
        for cat_name, pairs in categories.items():
            for tf, tf_cfg in timeframes.items():
                tasks.append((strat, cat_name, pairs, tf, tf_cfg["timerange"]))

    total = len(tasks)

    if max_workers > 1 and total > 1:
        info(f"Running {total} backtests across {max_workers} workers")
        task_results: Dict[int, Dict[str, Any]] = {}

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {}
            for idx, (strat, cat_name, pairs, tf, timerange) in enumerate(tasks):
                fut = pool.submit(
                    _backtest_worker,
                    strategy_name=strat,
                    strategy_path=strategy_path,
                    pairs=pairs,
                    timeframe=tf,
                    timerange=timerange,
                    wallet=wallet,
                )
                future_to_idx[fut] = idx

            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    task_results[idx] = fut.result()
                except Exception as exc:
                    task_results[idx] = {
                        "status": "fail",
                        "error": str(exc)[:120],
                        "duration_s": 0.0,
                    }

        # Print in original order
        for idx, (strat, cat_name, pairs, tf, timerange) in enumerate(tasks):
            r = task_results[idx]
            label = f"{strat} | {cat_name} | {tf}"
            progress = f"[{idx + 1}/{total}]"
            section(f"{progress} {label}")
            _print_and_record_result(r, strat, cat_name, tf, pairs, timerange, results)
    else:
        # Sequential fallback
        for idx, (strat, cat_name, pairs, tf, timerange) in enumerate(tasks):
            label = f"{strat} | {cat_name} | {tf}"
            progress = f"[{idx + 1}/{total}]"

            section(f"{progress} {label}")
            info(f"Pairs: {', '.join(pairs)}")
            info(f"Range: {timerange}")

            r = _backtest_worker(
                strategy_name=strat,
                strategy_path=strategy_path,
                pairs=pairs,
                timeframe=tf,
                timerange=timerange,
                wallet=wallet,
            )
            _print_and_record_result(r, strat, cat_name, tf, pairs, timerange, results)

    return results


def _print_and_record_result(
    r: Dict[str, Any],
    strat: str,
    cat_name: str,
    tf: str,
    pairs: List[str],
    timerange: str,
    results: Dict[str, Any],
):
    """Print a single backtest result and append it to the results dict."""
    dt = r.get("duration_s", 0.0)
    if r["status"] == "pass":
        m = r.get("metrics", {})
        ok(f"Completed in {elapsed_str(dt)}")
        if m.get("total_return_pct") is not None:
            ret_color = C.GREEN if m["total_return_pct"] >= 0 else C.RED
            info(
                f"Return: {ret_color}{m['total_return_pct']:+.2f}%{C.RESET}  "
                f"Sharpe: {m.get('sharpe', 'N/A')}  "
                f"DD: {m.get('max_drawdown_pct', 'N/A')}%  "
                f"Trades: {m.get('trades', 'N/A')}  "
                f"Win: {m.get('win_rate_pct', 'N/A')}%"
            )
        results["passed"] += 1
        results["backtest_results"].append({
            "strategy": strat,
            "category": cat_name,
            "timeframe": tf,
            "duration_s": round(dt, 1),
            "status": "pass",
            "metrics": m,
        })
    else:
        err_msg = r.get("error", "unknown error")
        fail(f"Failed after {elapsed_str(dt)}: {err_msg}")
        results["failed"] += 1
        results["backtest_results"].append({
            "strategy": strat,
            "category": cat_name,
            "timeframe": tf,
            "duration_s": round(dt, 1),
            "status": "fail",
            "error": err_msg,
        })


# ═══════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════

def print_summary_report(
    phase_results: Dict[str, Dict[str, Any]],
    total_time: float,
):
    width = 80
    banner("Benchmark Summary Report", "█", width)

    print(f"  {C.DIM}Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}")
    print(f"  {C.DIM}Total runtime: {elapsed_str(total_time)}{C.RESET}")
    print()

    # Phase-by-phase summary table
    header = f"  {'Phase':<40} {'Pass':>6} {'Fail':>6} {'Skip':>6} {'Result':>8}"
    print(f"{C.BOLD}{header}{C.RESET}")
    print(f"  {'─' * 68}")

    grand_pass = grand_fail = grand_skip = 0

    for phase_name, res in phase_results.items():
        p, f, s = res.get("passed", 0), res.get("failed", 0), res.get("skipped", 0)
        grand_pass += p
        grand_fail += f
        grand_skip += s

        if f > 0:
            status = f"{C.RED}FAIL{C.RESET}"
        elif p == 0 and s > 0:
            status = f"{C.YELLOW}SKIP{C.RESET}"
        else:
            status = f"{C.GREEN}PASS{C.RESET}"

        print(f"  {phase_name:<40} {p:>6} {f:>6} {s:>6}   {status}")

    print(f"  {'─' * 68}")
    total_label = "TOTAL"
    if grand_fail > 0:
        overall = f"{C.BG_RED}{C.WHITE}{C.BOLD} FAIL {C.RESET}"
    else:
        overall = f"{C.BG_GREEN}{C.WHITE}{C.BOLD} PASS {C.RESET}"
    print(f"  {total_label:<40} {grand_pass:>6} {grand_fail:>6} {grand_skip:>6}   {overall}")
    print()

    # Portfolio pipeline metrics (if available)
    pipeline_res = phase_results.get("3. Portfolio Pipeline", {})
    pipeline_metrics = pipeline_res.get("metrics", {})
    if pipeline_metrics:
        section("Portfolio Pipeline Performance")
        print()
        header = f"  {'Config':<25} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10} {'Bars':>8}"
        print(f"{C.BOLD}{header}{C.RESET}")
        print(f"  {'─' * 65}")
        for label, m in pipeline_metrics.items():
            ret_color = C.GREEN if m["total_return_pct"] >= 0 else C.RED
            print(
                f"  {label:<25} "
                f"{ret_color}{m['total_return_pct']:>+9.2f}%{C.RESET} "
                f"{m['annualised_sharpe']:>10.4f} "
                f"{m['max_drawdown_pct']:>9.2f}% "
                f"{m['n_bars']:>8}"
            )
        print()

    # Backtest grid (if available)
    for phase_key in ["4. Trading Backtests", "5. Portfolio Backtests"]:
        bt_results = phase_results.get(phase_key, {}).get("backtest_results", [])
        if not bt_results:
            continue

        section(f"{phase_key} — Results Grid")
        print()

        # Collect unique strategies, categories, timeframes
        strats = list(dict.fromkeys(r["strategy"] for r in bt_results))
        cats = list(dict.fromkeys(r["category"] for r in bt_results))
        tfs = list(dict.fromkeys(r["timeframe"] for r in bt_results))

        # Build lookup
        lookup = {}
        for r in bt_results:
            lookup[(r["strategy"], r["category"], r["timeframe"])] = r

        # Print a compact grid per timeframe showing return %
        for tf in tfs:
            print(f"  {C.BOLD}Timeframe: {tf}{C.RESET}")
            cat_header = "".join(f"{c:>12}" for c in cats)
            print(f"  {'Strategy':<35}{cat_header}")
            print(f"  {'─' * (35 + 12 * len(cats))}")

            for strat in strats:
                row = f"  {strat:<35}"
                for cat in cats:
                    r = lookup.get((strat, cat, tf))
                    if r is None:
                        row += f"{'—':>12}"
                    elif r["status"] == "pass":
                        m = r.get("metrics", {})
                        ret = m.get("total_return_pct")
                        if ret is not None:
                            color = C.GREEN if ret >= 0 else C.RED
                            row += f"{color}{ret:>+11.2f}%{C.RESET}"
                        else:
                            t = f"{r['duration_s']}s"
                            row += f"{C.GREEN}{t:>12}{C.RESET}"
                    else:
                        row += f"{C.RED}{'FAIL':>12}{C.RESET}"
                print(row)
            print()

    # Final verdict
    print(f"  {C.BOLD}{'═' * 68}{C.RESET}")
    if grand_fail == 0:
        print(f"  {C.GREEN}{C.BOLD}All benchmarks passed successfully.{C.RESET}")
    else:
        print(f"  {C.RED}{C.BOLD}{grand_fail} benchmark(s) failed — see details above.{C.RESET}")
    print()

    return grand_fail == 0


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PortfolioBench — comprehensive testing & benchmarking suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke-test: 5m timeframe, crypto only")
    parser.add_argument("--trading-only", action="store_true",
                        help="Only run trading strategy backtests (skip portfolio)")
    parser.add_argument("--portfolio-only", action="store_true",
                        help="Only run portfolio strategy backtests (skip trading)")
    parser.add_argument("--skip-backtests", action="store_true",
                        help="Only run data checks, unit tests, and pipeline")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Run only specified strategies (by class name)")
    parser.add_argument("--timeframes", nargs="+", default=None,
                        choices=["5m", "4h", "1d"],
                        help="Limit to specific timeframes")
    parser.add_argument("--categories", nargs="+", default=None,
                        choices=["crypto", "stocks", "indices", "mixed"],
                        help="Limit to specific asset categories")
    parser.add_argument("--json-output", type=str, default=None,
                        help="Write results to a JSON file")
    parser.add_argument("--workers", type=int, default=1, metavar="N",
                        help="Number of parallel worker processes for backtests (default: 1 = sequential)")
    args = parser.parse_args()

    print(f"{C.CYAN}{C.BOLD}")
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║         PORTFOLIOBENCH — FULL BENCHMARK SUITE              ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print(f"{C.RESET}")
    print(f"  {C.DIM}Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}")

    if args.quick:
        info("Quick mode: 5m timeframe, crypto pairs only")

    t_start = time.time()
    phase_results = {}

    # ── Phase 1: Data Integrity ──
    phase_results["1. Data Integrity"] = check_data_integrity()

    # ── Phase 2: Unit Tests ──
    phase_results["2. Unit Tests"] = run_unit_tests()

    # ── Phase 3: Standalone Portfolio Pipeline ──
    phase_results["3. Portfolio Pipeline"] = run_portfolio_pipeline()

    # ── Phase 4 & 5: Freqtrade Backtests ──
    if not args.skip_backtests:
        # Determine which categories and timeframes to use
        if args.quick:
            categories = {"crypto": ASSET_CATEGORIES["crypto"]}
            timeframes = {"5m": TIMEFRAME_CONFIG["5m"]}
        else:
            categories = {k: v for k, v in ASSET_CATEGORIES.items()
                          if args.categories is None or k in args.categories}
            timeframes = {k: v for k, v in TIMEFRAME_CONFIG.items()
                          if args.timeframes is None or k in args.timeframes}

        # Trading strategies
        if not args.portfolio_only:
            trading_strats = TRADING_STRATEGIES
            if args.strategies:
                trading_strats = [s for s in trading_strats if s in args.strategies]

            if trading_strats:
                phase_results["4. Trading Backtests"] = run_freqtrade_backtests(
                    strategies=trading_strats,
                    strategy_path=os.path.join(PROJECT_ROOT, "strategy"),
                    phase_name="Phase 4: Trading Strategy Backtests",
                    categories=categories,
                    timeframes=timeframes,
                    max_workers=max(1, args.workers),
                )

        # Portfolio strategies
        if not args.trading_only:
            portfolio_strats = PORTFOLIO_STRATEGIES
            if args.strategies:
                portfolio_strats = [s for s in portfolio_strats if s in args.strategies]

            if portfolio_strats:
                phase_results["5. Portfolio Backtests"] = run_freqtrade_backtests(
                    strategies=portfolio_strats,
                    strategy_path=os.path.join(PROJECT_ROOT, "user_data", "strategies"),
                    phase_name="Phase 5: Portfolio Strategy Backtests",
                    categories=categories,
                    timeframes=timeframes,
                    wallet=1_000_000,
                    max_workers=max(1, args.workers),
                )

    total_time = time.time() - t_start
    all_passed = print_summary_report(phase_results, total_time)

    # Optional JSON output
    if args.json_output:
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "total_runtime_s": round(total_time, 2),
            "phases": {},
        }
        for name, res in phase_results.items():
            phase_data: Dict[str, Any] = {
                "passed": res.get("passed", 0),
                "failed": res.get("failed", 0),
                "skipped": res.get("skipped", 0),
                "details": res.get("details", []),
            }
            if "metrics" in res:
                phase_data["metrics"] = res["metrics"]
            if "backtest_results" in res:
                # Ensure metrics are serializable (no DataFrames)
                clean_results = []
                for bt in res["backtest_results"]:
                    entry = {k: v for k, v in bt.items() if k != "metrics"}
                    m = bt.get("metrics", {})
                    entry["metrics"] = {
                        k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in m.items()
                    }
                    clean_results.append(entry)
                phase_data["backtest_results"] = clean_results
            json_data["phases"][name] = phase_data

        with open(args.json_output, "w") as f:
            json.dump(json_data, f, indent=2)
        info(f"JSON report written to: {args.json_output}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
