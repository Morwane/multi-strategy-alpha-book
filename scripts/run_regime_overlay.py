"""Regime-aware overlay backtest: HMM regimes -> dynamic sleeve allocation,
benchmarked head-to-head against the static risk-parity book on the SAME
out-of-sample window.  Usage: python scripts/run_regime_overlay.py
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
from src.data import load_components
from src.allocate import equal_weight, risk_parity
from src.metrics import vol_target, performance, subperiod_metrics
from src.regime.features import load_raw, build_features, HMM_COLS, coverage_report
from src.regime.hmm import walk_forward_regimes, regime_stats, LABELS
from src.regime.allocate import regime_book, disciplined_book, throttle_book

DATA = REPO / "data"
OUT = DATA / "regime"
STRESS_PERIODS = {
    "COVID crash 2020": ("2020-02-01", "2020-04-30"),
    "Rates+energy 2022": ("2022-01-01", "2022-12-31"),
    "Recent 2024-2025": ("2024-01-01", "2025-12-31"),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_raw(DATA)
    feats = build_features(raw)
    print("=" * 78)
    print("REGIME-AWARE MULTI-STRATEGY OVERLAY — walk-forward HMM")
    print(f"Raw panel: {raw.index[0].date()} → {raw.index[-1].date()}  "
          f"({raw.shape[1]} series)")

    wf = walk_forward_regimes(feats, HMM_COLS, n_states=3)
    regime = wf.regime
    print(f"Out-of-sample regimes: {regime.index[0].date()} → {regime.index[-1].date()} "
          f"({len(regime)} days, {len(wf.refit_log)} refits, "
          f"all converged={all(r['converged'] for r in wf.refit_log)})")

    comp = load_components(DATA)
    # Common out-of-sample window: where BOTH regimes and sleeves exist.
    common = comp.index.intersection(regime.index)
    comp = comp.loc[common]
    regime = regime.loc[common]

    rb = regime_book(comp, regime)
    db = disciplined_book(comp, regime, wf.proba.loc[common])
    tb = throttle_book(comp, regime, wf.proba.loc[common])
    ew = equal_weight(comp)
    rp, rp_w = risk_parity(comp)

    # Fair comparison: evaluate every book on the common valid (net) index.
    valid = (rb["port_net"].dropna().index
             .intersection(db["port_net"].dropna().index)
             .intersection(tb["port_net"].dropna().index))
    series = {
        "energy_statarb (sleeve)": comp["energy_statarb"].loc[valid],
        "vix_carry (sleeve)": comp["vix_carry"].loc[valid],
        "Equal-weight": ew.loc[valid],
        "Risk-parity (benchmark)": rp.loc[valid],
        "Regime naive (net)": rb["port_net"].loc[valid],
        "Regime disciplined (net)": db["port_net"].loc[valid],
        "Regime risk-throttle (net)": tb["port_net"].loc[valid],
    }

    print(f"\nEvaluation window: {valid[0].date()} → {valid[-1].date()} ({len(valid)} days)")
    print("-" * 78)
    print(f"{'Strategy':26} | Sharpe | CAGR  | Vol   | MaxDD  | Calmar | Turn/yr")
    turn_map = {"Regime naive (net)": rb["avg_turnover_annual"],
                "Regime disciplined (net)": db["avg_turnover_annual"],
                "Regime risk-throttle (net)": tb["avg_turnover_annual"]}
    perf_rows = {}
    for nm, r in series.items():
        m = performance(vol_target(r))
        perf_rows[nm] = m
        turn = f"{turn_map[nm]:5.1f}" if nm in turn_map else "    —"
        print(f"{nm:26} | {m['sharpe']:+.2f}  | {m['cagr']:+.1%} | {m['vol']:.1%} | "
              f"{m['max_dd']:.1%} | {m['calmar']:+.2f}  | {turn}")

    # ---- head-to-head verdict (the honest part) ----
    rp_m = perf_rows["Risk-parity (benchmark)"]
    print("-" * 78)
    print("REGIME OVERLAY vs STATIC RISK-PARITY (the honest test):")
    # higher is better for all three (max_dd is negative: closer to 0 = better)
    for label in ["Regime naive (net)", "Regime disciplined (net)",
                  "Regime risk-throttle (net)"]:
        rg_m = perf_rows[label]
        print(f"  {label}:")
        for k, fmt in [("sharpe", "{:+.2f}"), ("max_dd", "{:.1%}"), ("calmar", "{:+.2f}")]:
            d = rg_m[k] - rp_m[k]
            verdict = "better" if d > 0 else "worse"
            print(f"    {k:8}: RP {fmt.format(rp_m[k])}  ->  {fmt.format(rg_m[k])}  "
                  f"(Δ {fmt.format(d)}, {verdict})")

    # ---- regime economics ----
    print("-" * 78)
    print("REGIME ECONOMICS (conditional feature means — sanity check):")
    rstats = regime_stats(feats, regime)
    print(rstats.round(3).to_string())

    # ---- stress periods ----
    print("-" * 78)
    print("STRESS-PERIOD PERFORMANCE (Sharpe | MaxDD):")
    print(f"{'period':20} | {'Risk-parity':>17} | {'Regime throttle':>17}")
    for label, (a, b) in STRESS_PERIODS.items():
        rp_s = performance(vol_target(rp.loc[valid]).loc[a:b])
        rg_s = performance(vol_target(tb['port_net'].loc[valid]).loc[a:b])
        def cell(m):
            return f"{m.get('sharpe', float('nan')):+.2f} | {m.get('max_dd', float('nan')):.1%}" if m else "n/a"
        print(f"{label:20} | {cell(rp_s):>17} | {cell(rg_s):>17}")

    # ---- integrity checks ----
    print("-" * 78)
    checks = []
    checks.append(("regime_oos_only", regime.index[0] > raw.index[0],
                   f"first regime {regime.index[0].date()} after train window"))
    checks.append(("weights_lagged", bool(rb["port_net"].isna().iloc[0]),
                   "first regime-book return NaN (shift(1) weights)"))
    checks.append(("regimes_three", regime.nunique() == 3,
                   f"{regime.nunique()} distinct regimes realised"))
    checks.append(("costs_charged", rb["avg_turnover_annual"] > 0,
                   f"avg turnover {rb['avg_turnover_annual']:.1f}x/yr, costs applied"))
    ok = True
    for name, passed, detail in checks:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:18} — {detail}")
    print("=" * 78)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")

    # ---- exports ----
    pd.DataFrame({"regime": regime, "regime_label": regime.map(LABELS)}).to_csv(OUT / "regimes.csv")
    wf.proba.loc[valid].to_csv(OUT / "regime_proba.csv")
    pd.DataFrame({nm: vol_target(r) for nm, r in series.items()}).to_csv(OUT / "overlay_returns.csv")
    pd.DataFrame(perf_rows).T.to_csv(OUT / "overlay_performance.csv")
    rstats.to_csv(OUT / "regime_stats.csv")
    coverage_report(feats).to_csv(OUT / "feature_coverage.csv")
    rb["weights"].loc[valid].to_csv(OUT / "regime_weights.csv")
    print(f"Saved CSV outputs -> {OUT}")

    # ---- figures + tearsheet ----
    from src.regime.plots import make_regime_figures
    vt = {nm: vol_target(r) for nm, r in series.items()}
    assets = REPO / "docs" / "assets"
    make_regime_figures(raw["SPX"], regime, wf.proba.loc[common], vt,
                        tb["confirmed"], assets)
    print(f"Saved figures -> {assets}")
    write_tearsheet(REPO / "reports" / "regime_tearsheet.md", valid, perf_rows,
                    rstats, turn_map, wf)
    print(f"Saved tearsheet -> {REPO / 'reports' / 'regime_tearsheet.md'}")
    return 0 if ok else 1


def write_tearsheet(path, valid, perf, rstats, turn_map, wf):
    rp, naive = perf["Risk-parity (benchmark)"], perf["Regime naive (net)"]
    disc, thr = perf["Regime disciplined (net)"], perf["Regime risk-throttle (net)"]
    def row(nm, m, t="—"):
        return (f"| {nm} | {m['sharpe']:+.2f} | {m['cagr']:+.1%} | {m['vol']:.1%} | "
                f"{m['max_dd']:.1%} | {m['calmar']:+.2f} | {t} |")
    lines = [
        "# Regime overlay — tearsheet",
        "",
        f"Walk-forward Gaussian HMM (3 states), out-of-sample "
        f"{valid[0].date()} → {valid[-1].date()} ({len(valid)} days), "
        f"{len(wf.refit_log)} quarterly refits, net of 2bps/turnover costs.",
        "",
        "| Strategy | Sharpe | CAGR | Vol | MaxDD | Calmar | Turn/yr |",
        "|---|---|---|---|---|---|---|",
        row("Risk-parity (benchmark)", rp),
        row("Regime naive", naive, f"{turn_map['Regime naive (net)']:.1f}"),
        row("Regime disciplined", disc, f"{turn_map['Regime disciplined (net)']:.1f}"),
        row("**Regime risk-throttle**", thr, f"{turn_map['Regime risk-throttle (net)']:.1f}"),
        "",
        "## Finding",
        "- Regime as an **alpha-timing / reallocation** signal (naive, disciplined) "
        "**underperforms** static risk-parity: the switching lag (persistence trap) "
        "and breaking the diversification mix destroy value.",
        "- Regime as a **risk throttle** (keep the risk-parity mix, cut gross in "
        f"confirmed stress) keeps Sharpe ({thr['sharpe']:+.2f} vs {rp['sharpe']:+.2f}) "
        f"while cutting max drawdown ({thr['max_dd']:.1%} vs {rp['max_dd']:.1%}) and "
        f"lifting Calmar ({thr['calmar']:+.2f} vs {rp['calmar']:+.2f}) at trivial "
        f"turnover ({turn_map['Regime risk-throttle (net)']:.1f}x/yr).",
        "- **Lesson:** regime models here are useful for risk management, not return timing.",
        "",
        "## Regime economics (conditional feature means)",
        "",
        rstats.round(3).to_markdown(),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
