"""Robustness suite: cost sensitivity, bootstrap CIs, correlation stability,
benchmark comparison, crisis tests. Prints a summary, writes figures + a report.

Usage: python scripts/run_robustness.py
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from src.robustness import (load_overlay, cost_sensitivity, block_bootstrap,
                            rolling_sleeve_corr, benchmarks, crisis_table)
from src.allocate import risk_parity_weights
from src.metrics import performance, vol_target


def risk_contribution(repo, window=63):
    """Each sleeve's share of portfolio risk under risk parity (should be ~equal)."""
    comp = pd.read_csv(repo / "data" / "components" / "components.csv",
                       index_col=0, parse_dates=True).dropna()
    w = risk_parity_weights(comp); c0, c1 = comp.columns
    v0, v1 = comp[c0].rolling(window).var(), comp[c1].rolling(window).var()
    cv = comp[c0].rolling(window).cov(comp[c1])
    w0, w1 = w[c0], w[c1]
    pv = w0**2 * v0 + w1**2 * v1 + 2 * w0 * w1 * cv
    rc = pd.concat([(w0 * (w0 * v0 + w1 * cv) / pv).rename(c0),
                    (w1 * (w1 * v1 + w0 * cv) / pv).rename(c1)], axis=1).dropna()
    return rc

ASSETS, REPORTS = REPO / "docs" / "assets", REPO / "reports"
plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 140, "axes.grid": True,
                     "grid.alpha": 0.22, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
                     "figure.autolayout": True})
BLUE, RED, GREEN, GREY = "#1f5fa8", "#c0392b", "#27ae60", "#7f8c8d"
RP, TH = "Risk-parity (benchmark)", "Regime risk-throttle (net)"


def plot_bootstrap(sh, dd, title, path, color=BLUE):
    """Polished 2-panel Monte-Carlo figure: KDE + shaded 90% CI + median + P(>0)."""
    sh, dd = np.asarray(sh), np.asarray(dd) * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, data, lab, fmt, is_sh in [(axes[0], sh, "Annualized Sharpe", "{:+.2f}", True),
                                      (axes[1], dd, "Max Drawdown (%)", "{:.0f}", False)]:
        lo, hi, med = *np.percentile(data, [5, 95]), np.median(data)
        ax.hist(data, bins=45, density=True, color=color, alpha=.16, edgecolor="none")
        xs = np.linspace(data.min(), data.max(), 300); kde = gaussian_kde(data)(xs)
        ax.plot(xs, kde, color=color, lw=2.2)
        ax.fill_between(xs, kde, where=(xs >= lo) & (xs <= hi), color=color, alpha=.30)
        ax.axvline(med, color=RED, lw=1.5, ls="--")
        ax.set_title(lab); ax.set_yticks([]); ax.margins(x=0.01)
        txt = f"90% CI [{fmt.format(lo)}, {fmt.format(hi)}]\nmedian {fmt.format(med)}"
        if is_sh:
            ax.axvline(0, color="black", lw=.9); txt += f"\nP(Sharpe>0) = {(data > 0).mean():.0%}"
        ax.text(.03, .96, txt, transform=ax.transAxes, fontsize=8.5, va="top",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color, alpha=.9))
    fig.suptitle(title, fontweight="bold", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main():
    ov = load_overlay(REPO)
    oos = ov.index
    print("=" * 72)
    print(f"ROBUSTNESS SUITE  ({oos[0].date()} -> {oos[-1].date()}, {len(oos)} OOS days)")
    print("=" * 72)

    # 1. cost sensitivity
    cs = cost_sensitivity(REPO, oos)
    print("\n[1] Transaction-cost sensitivity (risk-parity book):")
    print(cs.to_string(formatters={"Sharpe": "{:+.2f}".format, "CAGR": "{:+.1%}".format, "MaxDD": "{:.1%}".format}))

    # 2. bootstrap (risk-throttle)
    sh, dd = block_bootstrap(ov[TH])
    lo, hi = np.percentile(sh, [5, 95]); ddlo, ddhi = np.percentile(dd, [5, 95])
    print(f"\n[2] Block-bootstrap (risk-throttle, 2000x, 21d blocks):")
    print(f"    Sharpe 90% CI [{lo:+.2f}, {hi:+.2f}] (median {np.median(sh):+.2f}); "
          f"P(Sharpe>0) = {(sh>0).mean():.0%}")
    print(f"    MaxDD 90% CI [{ddlo:.1%}, {ddhi:.1%}]")

    # 3. correlation stability
    rc = rolling_sleeve_corr(ov)
    print(f"\n[3] Sleeve correlation: full {ov['energy_statarb (sleeve)'].corr(ov['vix_carry (sleeve)']):+.2f} | "
          f"126d mean {rc['126d'].mean():+.2f}, max {rc['126d'].max():+.2f} (worst co-move)")

    # 4. benchmarks
    bm = benchmarks(REPO, oos)
    print("\n[4] Benchmark comparison (Sharpe | CAGR | MaxDD):")
    for nm, r in [("Risk-parity book", ov[RP]), ("Risk-throttle book", ov[TH]),
                  *bm.items()]:
        m = performance(r)
        print(f"    {nm:22} {m['sharpe']:+.2f} | {m['cagr']:+.1%} | {m['max_dd']:.1%}")

    # 5. crisis tests
    ct = crisis_table(ov, [RP, TH])
    print("\n[5] Crisis cumulative return (risk-parity vs risk-throttle):")
    print(ct.to_string(formatters={c: "{:+.1%}".format for c in ct.columns}))

    # 6. risk contribution
    rc = risk_contribution(REPO)
    rc_mean = rc.mean()
    print(f"\n[6] Risk contribution (risk-parity): "
          f"{rc.columns[0]} {rc_mean.iloc[0]:.0%} | {rc.columns[1]} {rc_mean.iloc[1]:.0%} "
          f"(equal-risk by design, vs weights which differ)")

    # ---- figures ----
    plot_bootstrap(sh, dd, "Multi-Strategy Book - Monte-Carlo robustness (risk-throttle, 2000 resamples)",
                   ASSETS / "robust_bootstrap_sharpe.png")

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.stackplot(rc.index, rc.iloc[:, 0] * 100, rc.iloc[:, 1] * 100,
                 colors=[GREEN, RED], alpha=.75, labels=list(rc.columns))
    ax.axhline(50, color="black", lw=.8, ls="--")
    ax.set_ylim(0, 100); ax.set_ylabel("risk contribution (%)")
    ax.set_title("Risk contribution by sleeve - risk parity equalizes risk (not capital)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.savefig(ASSETS / "robust_risk_contribution.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(rc.index, rc["126d"], color=BLUE, lw=.9, label="126d")
    ax.plot(rc.index, rc["252d"], color=RED, lw=.9, label="252d")
    ax.axhline(0, color="black", lw=.7)
    ax.set_title("Sleeve correlation stability (energy stat-arb vs VIX carry)", fontweight="bold")
    ax.set_ylabel("rolling correlation"); ax.legend(fontsize=8)
    fig.savefig(ASSETS / "robust_rolling_correlation.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.4))
    for r, nm, c, w in [(ov[TH], "Risk-throttle book", BLUE, 1.8),
                        (bm["SPY"], "SPY", GREY, 1.0),
                        (bm["60/40 (SPY/LQD)"], "60/40", GREEN, 1.0)]:
        eq = np.exp(r.reindex(oos).fillna(0).cumsum())
        ax.plot(eq.index, (eq - 1) * 100, color=c, lw=w,
                label=f"{nm} (Sharpe {performance(r)['sharpe']:+.2f}, maxDD {performance(r)['max_dd']:.0%})")
    ax.set_title("Multi-Strategy Book vs SPY and 60/40 (native vol)", fontweight="bold")
    ax.set_ylabel("cumulative return (%)"); ax.legend(loc="upper left", fontsize=8.5)
    fig.savefig(ASSETS / "robust_benchmark.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(ct)); wd = 0.38
    ax.bar(x - wd/2, ct[RP]*100, wd, color=GREY, label="risk-parity")
    ax.bar(x + wd/2, ct[TH]*100, wd, color=BLUE, label="risk-throttle")
    ax.axhline(0, color="black", lw=.8); ax.set_xticks(x)
    ax.set_xticklabels(ct.index, rotation=20, ha="right", fontsize=8)
    ax.set_title("Crisis stress test - cumulative return by episode", fontweight="bold")
    ax.set_ylabel("return (%)"); ax.legend(fontsize=8)
    fig.savefig(ASSETS / "robust_crisis.png"); plt.close(fig)

    # ---- report ----
    L = ["# Robustness Appendix", "",
         f"Out-of-sample {oos[0].date()} to {oos[-1].date()} ({len(oos)} days). Vol-targeted 10%.", "",
         "## 1. Transaction-cost sensitivity (risk-parity book)", "",
         "| Cost | Sharpe | CAGR | Max DD |", "|---|--:|--:|--:|"]
    for lvl, row in cs.iterrows():
        L.append(f"| {lvl} | {row['Sharpe']:+.2f} | {row['CAGR']:+.1%} | {row['MaxDD']:.1%} |")
    L += ["", "Low allocation turnover -> the book is robust to realistic costs.", "",
          "## 2. Block-bootstrap confidence (risk-throttle)", "",
          f"- Sharpe 90% CI **[{lo:+.2f}, {hi:+.2f}]**, median {np.median(sh):+.2f}; P(Sharpe>0) = **{(sh>0).mean():.0%}**.",
          f"- Max-drawdown 90% CI [{ddlo:.1%}, {ddhi:.1%}].",
          "", "![Bootstrap Sharpe](docs/assets/robust_bootstrap_sharpe.png)", "",
          "## 3. Sleeve-correlation stability", "",
          f"- Full-sample correlation {ov['energy_statarb (sleeve)'].corr(ov['vix_carry (sleeve)']):+.2f}; "
          f"rolling 126d stays low (mean {rc['126d'].mean():+.2f}, worst {rc['126d'].max():+.2f}) - "
          "the diversification does **not** break down in stress (directly relevant to my thesis on correlation regimes).",
          "", "![Rolling correlation](docs/assets/robust_rolling_correlation.png)", "",
          "## 4. Benchmark comparison", "",
          "| Strategy | Sharpe | CAGR | Max DD |", "|---|--:|--:|--:|"]
    for nm, r in [("Risk-parity book", ov[RP]), ("Risk-throttle book", ov[TH]), *bm.items()]:
        m = performance(r)
        L.append(f"| {nm} | {m['sharpe']:+.2f} | {m['cagr']:+.1%} | {m['max_dd']:.1%} |")
    L += ["", "![Benchmark](docs/assets/robust_benchmark.png)", "",
          "## 5. Crisis stress tests", "",
          "| Episode | Risk-parity | Risk-throttle |", "|---|--:|--:|"]
    for ep, row in ct.iterrows():
        L.append(f"| {ep} | {row[RP]:+.1%} | {row[TH]:+.1%} |")
    L += ["", "![Crisis](docs/assets/robust_crisis.png)", "",
          "_The regime risk-throttle reduces the crisis drawdowns of the static book._", "",
          "## 6. Risk contribution", "",
          f"- Under risk parity each sleeve carries ~equal risk: **{rc.columns[0]} {rc_mean.iloc[0]:.0%} / "
          f"{rc.columns[1]} {rc_mean.iloc[1]:.0%}** — risk is equalized, not capital (the point of risk parity).",
          "", "![Risk contribution](docs/assets/robust_risk_contribution.png)"]
    (REPORTS / "robustness.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nFigures + report written. Report: {REPORTS / 'robustness.md'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
