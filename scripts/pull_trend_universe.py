"""Fetch the cross-asset trend universe from LSEG (run with an active Workspace session).

Audit-first: tries each candidate RIC, saves what returns data, reports coverage.
    python scripts/pull_trend_universe.py
Output: data/trend/prices.csv  (+ per-RIC coverage printed)
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd
from src.trend import TREND_UNIVERSE, ALL_RICS

OUT = REPO / "data" / "trend"
OUT.mkdir(parents=True, exist_ok=True)
START, END = "2007-01-01", "2026-06-05"


def main():
    import lseg.data as ld
    ld.open_session()
    cols, log = {}, []
    try:
        for ric in ALL_RICS:
            try:
                df = ld.get_history(universe=ric, fields=["TRDPRC_1"],
                                    interval="daily", start=START, end=END)
                if df is None or df.empty:
                    log.append((ric, "EMPTY")); continue
                s = df.iloc[:, 0].dropna(); s.index = pd.to_datetime(s.index)
                if len(s) < 500:
                    log.append((ric, f"too short ({len(s)})")); continue
                cols[ric] = s.sort_index()
                log.append((ric, f"OK {len(s)} obs {s.index.min().date()}→{s.index.max().date()}"))
            except Exception as e:
                log.append((ric, f"ERR {type(e).__name__}"))
    finally:
        ld.close_session()

    if cols:
        pd.concat(cols, axis=1).to_csv(OUT / "prices.csv")
    print("=" * 64)
    print(f"TREND UNIVERSE FETCH — {len(cols)}/{len(ALL_RICS)} RICs usable")
    print("=" * 64)
    for cls, rics in TREND_UNIVERSE.items():
        print(f"\n[{cls}]")
        for r in rics:
            status = dict(log).get(r, "?")
            print(f"  {r:9} {status}")
    print(f"\nSaved: {OUT / 'prices.csv'}  →  now run: python scripts/trend_sleeve_analysis.py")


if __name__ == "__main__":
    main()
