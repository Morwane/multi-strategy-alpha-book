"""Run the multi-strategy allocation: metrics + integrity checks.  Usage: python scripts/run_backtest.py"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd
from src.data import load_components
from src.allocate import build, quant_checks
from src.metrics import vol_target, performance

DATA = REPO / "data"


def main():
    comp = load_components(DATA)
    res = build(comp)
    names = list(comp.columns)
    print("=" * 76)
    print("MULTI-STRATEGY ALPHA BOOK — allocation backtest")
    print(f"Period: {comp.index[0].date()} → {comp.index[-1].date()} ({len(comp)} days)")
    print(f"Sleeve correlation: {res['corr_full']:+.2f}  (≈0 → diversifying)")
    print("-" * 76)
    print(f"{'Strategy':22} | Sharpe | CAGR  | Vol   | MaxDD  | Calmar")
    series = [(names[0], comp[names[0]]), (names[1], comp[names[1]]),
              ("Equal-weight", res["port_ew"]), ("Risk-parity book", res["port_rp"])]
    for nm, r in series:
        m = performance(vol_target(r))
        print(f"{nm:22} | {m['sharpe']:+.2f}  | {m['cagr']:+.1%} | {m['vol']:.1%} | "
              f"{m['max_dd']:.1%} | {m['calmar']:+.2f}")
    print("-" * 76)
    best_sleeve = max(performance(vol_target(comp[n]))["sharpe"] for n in names)
    book = performance(vol_target(res["port_rp"]))["sharpe"]
    print(f"Diversification: best sleeve Sharpe {best_sleeve:+.2f} → combined {book:+.2f} "
          f"({'+' if book > best_sleeve else ''}{book - best_sleeve:+.2f})")
    print("-" * 76)
    ok = True
    for name, passed, detail in quant_checks(comp, res):
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:24} — {detail}")
    print("=" * 76); print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    pd.concat([res["port_ew"], res["port_rp"]], axis=1).to_csv(DATA / "results.csv")


if __name__ == "__main__":
    main()
