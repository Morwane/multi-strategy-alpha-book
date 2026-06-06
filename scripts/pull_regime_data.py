"""Pull the raw market data needed for HMM regime features and persist to CSV.

Run ONCE with LSEG Workspace open and logged in:
    ../.venv/bin/python scripts/pull_regime_data.py

All downstream development (features, HMM, backtest) reads the saved CSV and
needs no live session. Every series is MEASURED from LSEG; nothing is faked.
RICs were confirmed by scripts/probe_macro_dxy_10y.py (2026-06-06).
"""
from pathlib import Path
import sys
import pandas as pd
import lseg.data as ld

START, END = "2010-01-01", "2026-06-04"
OUT = Path(__file__).resolve().parent.parent / "data" / "regime" / "raw_prices.csv"

# name -> (ric, how). how="px" => get_history TRDPRC_1 ; how="yld" => default col (YLDTOMAT)
SERIES = {
    # energy fronts (spreads / crack / vol features)
    "WTI":       ("CLc1", "px"),
    "Brent":     ("LCOc1", "px"),
    "Gasoline":  ("RBc1", "px"),
    "HeatingOil":("HOc1", "px"),
    "NatGas":    ("NGc1", "px"),
    # WTI curve for slope / carry feature
    "WTI_c2":    ("CLc2", "px"),
    "WTI_c12":   ("CLc12", "px"),
    # equity / vol
    "SPX":       (".SPX", "px"),
    "VIX":       (".VIX", "px"),
    "VIX3M":     (".VIX3M", "px"),
    # macro
    "DXY":       (".DXY", "px"),
    "US10Y":     ("US10YT=RR", "yld"),
    "US2Y":      ("US2YT=RR", "yld"),
}


def fetch(ric: str, how: str) -> pd.Series | None:
    try:
        if how == "px":
            df = ld.get_history(universe=ric, fields=["TRDPRC_1"],
                                start=START, end=END, interval="daily")
        else:  # yield: default history column carries YLDTOMAT
            df = ld.get_history(universe=ric, start=START, end=END, interval="daily")
    except Exception as e:  # noqa: BLE001
        print(f"  [{ric}] ERROR {str(e)[:70]}")
        return None
    if df is None or df.empty:
        print(f"  [{ric}] EMPTY")
        return None
    num = df.select_dtypes(include="number")
    if num.shape[1] == 0:
        print(f"  [{ric}] no numeric col {list(df.columns)}")
        return None
    s = num.iloc[:, 0].dropna()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def main() -> int:
    ld.open_session()
    cols = {}
    try:
        for name, (ric, how) in SERIES.items():
            s = fetch(ric, how)
            if s is None:
                print(f"{name:11s} <- {ric:11s} FAILED")
                continue
            cov = len(s) / len(pd.bdate_range(s.index.min(), s.index.max())) * 100
            print(f"{name:11s} <- {ric:11s} n={len(s):5d} {s.index.min().date()}"
                  f"->{s.index.max().date()} cov={cov:.1f}% last={s.iloc[-1]:.3f}")
            cols[name] = s
    finally:
        ld.close_session()

    if not cols:
        print("No data pulled — is Workspace open?")
        return 1
    raw = pd.DataFrame(cols).sort_index()
    raw.index.name = "Date"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUT)
    print(f"\nSaved {raw.shape[0]} rows x {raw.shape[1]} cols -> {OUT}")
    miss = raw.isna().mean().mul(100).round(1)
    print("missing % per col:\n" + miss.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
