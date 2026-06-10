"""Cross-asset time-series momentum (trend) sleeve.

A diversified trend book à la Moskowitz, Ooi & Pedersen (2012): for each market,
take a blended 3/6/12-month trend sign, size each market by inverse volatility
(equal risk), and average across markets. Look-ahead-free (signal & vol lagged).

A trend sleeve is the classic *orthogonal* complement to carry / mean-reversion
books — but it only adds value to a multi-strategy book if it is BOTH decorrelated
AND has a positive stand-alone Sharpe; breadth (many markets) is what gives it that.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Candidate cross-asset universe (continuation futures). FX excluded (no access).
TREND_UNIVERSE: dict[str, list[str]] = {
    "equity": ["ESc1", "NQc1", "YMc1", "FESXc1"],          # S&P, Nasdaq, Dow, EuroStoxx
    "rates":  ["TYc1", "TUc1", "USc1", "FGBLc1"],          # UST 10Y/2Y/30Y, Bund
    "commodity": ["CLc1", "LCOc1", "NGc1", "GCc1", "SIc1", "HGc1"],  # WTI, Brent, gas, gold, silver, copper
}
ALL_RICS: list[str] = [r for v in TREND_UNIVERSE.values() for r in v]


def load_trend_prices(data_dir: Path) -> pd.DataFrame:
    """Load the fetched trend-universe prices (wide frame, one column per RIC)."""
    p = Path(data_dir) / "trend" / "prices.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — run scripts/pull_trend_universe.py first (LSEG session).")
    return pd.read_csv(p, index_col=0, parse_dates=True).sort_index()


def build_trend_sleeve(prices: pd.DataFrame, vol_window: int = 63) -> pd.Series:
    """Build the diversified trend sleeve return.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide daily price frame, one column per market (NaNs before a market starts).
    vol_window : int, default 63
        Look-back for the inverse-volatility risk weighting.

    Returns
    -------
    pd.Series
        Daily sleeve return (equal-risk blended TS-momentum), look-ahead-free.
    """
    px = prices.where(prices > 0)          # guard non-positive prints (e.g. WTI April-2020 = -$37)
    rets = np.log(px / px.shift(1)).clip(-0.5, 0.5)   # cap roll/data artifacts
    # blended trend sign in [-1, 1]: 3m / 6m / 12m
    sig = (0.4 * np.sign(px / px.shift(63) - 1.0)
           + 0.3 * np.sign(px / px.shift(126) - 1.0)
           + 0.3 * np.sign(px / px.shift(252) - 1.0))
    inv_vol = 1.0 / rets.rolling(vol_window).std()
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)        # equal risk across available markets
    pos = (sig * weights).shift(1)                            # no look-ahead
    return (pos * rets).sum(axis=1, min_count=1).rename("trend")
