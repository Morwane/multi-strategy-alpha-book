"""Does adding a cross-asset TREND sleeve diversify the book?

Builds a time-series-momentum sleeve (Moskowitz-Ooi-Pedersen style) on a small
cross-asset universe (equity / oil / gas / dollar) from local LSEG data, measures
its correlation to the existing energy + VIX sleeves, and compares the 2-sleeve
vs 3-sleeve risk-parity book. Honest test: only worth adding if it decorrelates.

Usage: python scripts/trend_sleeve_analysis.py
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
from src.allocate import risk_parity
from src.metrics import performance, vol_target
from src.trend import build_trend_sleeve, load_trend_prices

DATA = REPO / "data"
# fallback universe (4 local assets) if the broad LSEG universe hasn't been fetched
FALLBACK = ["SPX", "Brent", "NatGas", "DXY"]


def main():
    comp = pd.read_csv(DATA / "components" / "components.csv", index_col=0, parse_dates=True)
    try:
        prices = load_trend_prices(DATA)
        universe = f"broad LSEG universe ({prices.shape[1]} markets)"
    except FileNotFoundError:
        raw = pd.read_csv(DATA / "regime" / "raw_prices.csv", index_col=0, parse_dates=True)
        prices = raw[FALLBACK]
        universe = f"FALLBACK 4 local assets {FALLBACK} (run pull_trend_universe.py for the broad book)"
    trend = build_trend_sleeve(prices)

    out = [f"Trend universe: {universe}"]
    m_tr = performance(vol_target(trend))
    out.append("TREND sleeve (standalone): Sharpe %+.2f | CAGR %+.1f%% | maxDD %.1f%%"
               % (m_tr["sharpe"], m_tr["cagr"] * 100, m_tr["max_dd"] * 100))

    # correlation to existing sleeves
    all3 = pd.concat([comp, trend], axis=1).dropna()
    C = all3.corr()
    out.append("\nSleeve correlation matrix:")
    out.append(C.round(2).to_string())

    # 2-sleeve vs 3-sleeve risk-parity book (same window)
    comp2 = all3[["energy_statarb", "vix_carry"]]
    comp3 = all3[["energy_statarb", "vix_carry", "trend"]]
    book2, _ = risk_parity(comp2)
    book3, _ = risk_parity(comp3)
    m2, m3 = performance(vol_target(book2.dropna())), performance(vol_target(book3.dropna()))
    out.append("\nRisk-parity book (same window, vol-targeted 10%):")
    out.append("  2 sleeves (energy+vix)        Sharpe %+.2f | CAGR %+.1f%% | maxDD %.1f%% | Calmar %+.2f"
               % (m2["sharpe"], m2["cagr"] * 100, m2["max_dd"] * 100, m2["calmar"]))
    out.append("  3 sleeves (+trend)            Sharpe %+.2f | CAGR %+.1f%% | maxDD %.1f%% | Calmar %+.2f"
               % (m3["sharpe"], m3["cagr"] * 100, m3["max_dd"] * 100, m3["calmar"]))
    out.append("\nVerdict: %s" % (
        "ADD IT — diversifies (Sharpe up, low corr)" if m3["sharpe"] > m2["sharpe"] + 0.03
        else "DILUTES or neutral — keep 2 sleeves" if m3["sharpe"] < m2["sharpe"] - 0.03
        else "MARGINAL — roughly neutral"))
    report = "\n".join(out)
    (REPO / "reports" / "trend_sleeve_study.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
