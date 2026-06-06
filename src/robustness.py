"""Robustness analysis for the multi-strategy book: transaction-cost sensitivity,
block-bootstrap confidence intervals, sleeve-correlation stability, benchmark
comparison (SPY / 60-40), and crisis stress tests.

Operates on the out-of-sample overlay returns (data/regime/overlay_returns.csv)
and recomputes the risk-parity book from the sleeves for the cost-sensitivity test.
Everything is look-ahead-free; all return streams are vol-targeted to 10% so Sharpe
and drawdown are comparable.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from .allocate import risk_parity_weights
from .metrics import performance, vol_target

TD = 252
CRISES = {
    "Aug 2015 (China)":   ("2015-08-01", "2015-09-30"),
    "Q4 2018 selloff":    ("2018-10-01", "2018-12-31"),
    "COVID crash 2020":   ("2020-02-19", "2020-03-31"),
    "2022 bear market":   ("2022-01-01", "2022-10-31"),
    "Aug 2024 unwind":    ("2024-07-25", "2024-08-15"),
}


def load_overlay(repo: Path) -> pd.DataFrame:
    return pd.read_csv(repo / "data" / "regime" / "overlay_returns.csv",
                       index_col=0, parse_dates=True)


def _ret(repo, name):
    s = pd.read_csv(repo / "data" / "benchmarks" / f"{name}.csv",
                    parse_dates=["Date"], index_col="Date").iloc[:, 0]
    return np.log(s / s.shift(1))


# 1 -------- transaction-cost sensitivity (recompute RP book gross, vary bps) ----
def cost_sensitivity(repo: Path, oos_index, bps_levels=(0, 5, 10, 25, 50)) -> pd.DataFrame:
    comp = pd.read_csv(repo / "data" / "components" / "components.csv",
                       index_col=0, parse_dates=True).dropna()
    w = risk_parity_weights(comp)
    gross = (w * comp).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(0)
    rows = {}
    for bps in bps_levels:
        net = (gross - turn * bps / 1e4).reindex(oos_index).dropna()
        m = performance(vol_target(net))
        rows[f"{bps} bps"] = {"Sharpe": m["sharpe"], "CAGR": m["cagr"], "MaxDD": m["max_dd"]}
    return pd.DataFrame(rows).T


# 2 -------- block bootstrap of Sharpe & maxDD --------------------------------
def block_bootstrap(ret: pd.Series, n=2000, block=21, seed=0):
    r = ret.dropna().values
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(len(r) / block))
    sharpes, dds = np.empty(n), np.empty(n)
    for i in range(n):
        starts = rng.integers(0, len(r) - block, nblocks)
        samp = np.concatenate([r[s:s + block] for s in starts])[:len(r)]
        sharpes[i] = samp.mean() / samp.std() * np.sqrt(TD)
        eq = np.cumprod(1 + samp); dds[i] = (eq / np.maximum.accumulate(eq) - 1).min()
    return sharpes, dds


# 3 -------- rolling sleeve correlation ---------------------------------------
def rolling_sleeve_corr(overlay: pd.DataFrame):
    a, b = overlay["energy_statarb (sleeve)"], overlay["vix_carry (sleeve)"]
    return pd.DataFrame({"126d": a.rolling(126).corr(b), "252d": a.rolling(252).corr(b)})


# 4 -------- benchmark comparison ---------------------------------------------
def benchmarks(repo: Path, oos_index) -> dict:
    spy = _ret(repo, "SPY").reindex(oos_index).dropna()
    lqd = _ret(repo, "LQD").reindex(oos_index).dropna()
    idx = spy.index.intersection(lqd.index)
    sixty40 = 0.6 * spy.reindex(idx) + 0.4 * lqd.reindex(idx)
    return {"SPY": spy, "60/40 (SPY/LQD)": sixty40}


# 5 -------- crisis stress tests ----------------------------------------------
def crisis_table(overlay: pd.DataFrame, cols) -> pd.DataFrame:
    rows = {}
    for name, (a, b) in CRISES.items():
        win = overlay.loc[a:b]
        if len(win) < 3:
            continue
        rows[name] = {c: (np.exp(win[c].sum()) - 1) for c in cols}   # cumulative return in window
    return pd.DataFrame(rows).T
