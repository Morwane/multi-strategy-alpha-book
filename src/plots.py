"""Figures for the multi-strategy alpha book → docs/assets/*.png."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import vol_target, performance, subperiod_metrics

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "figure.autolayout": True})
BLUE, RED, GREEN, GREY, PURPLE = "#1f5fa8", "#c0392b", "#27ae60", "#7f8c8d", "#8e44ad"
SUBPERIODS = {"2010-2014": ("2010", "2014"), "2015-2019": ("2015", "2019"),
              "2020-2022": ("2020", "2022"), "2023-2026": ("2023", "2026")}


def make_all(res: dict, assets: Path):
    assets = Path(assets); assets.mkdir(parents=True, exist_ok=True)
    comp, rp, ew = res["comp"], res["port_rp"], res["port_ew"]
    names = list(comp.columns)

    # 1 — equity: sleeves vs combined
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for r, nm, c, w in [(comp[names[0]], names[0], GREEN, 1.0),
                        (comp[names[1]], names[1], RED, 1.0),
                        (rp, "Risk-parity book", BLUE, 1.9)]:
        eq = (1 + vol_target(r)).cumprod()
        ax.plot(eq.index, (eq - 1) * 100, color=c, lw=w,
                label=f"{nm}  (Sharpe {performance(vol_target(r))['sharpe']:+.2f})")
    ax.set_title("Multi-Strategy Alpha Book — Sleeves vs Combined (vol-targeted 10%)", fontweight="bold")
    ax.set_ylabel("cumulative return (%)"); ax.legend(loc="upper left", fontsize=9)
    fig.savefig(assets / "equity_curve.png"); plt.close(fig)

    # 2 — rolling correlation between sleeves
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(res["roll_corr"].index, res["roll_corr"], color=PURPLE, lw=1)
    ax.axhline(0, color="black", lw=.7)
    ax.axhline(res["corr_full"], color=RED, ls="--", lw=.8, label=f"full-sample ρ={res['corr_full']:+.2f}")
    ax.set_title("Rolling 126d Correlation Between Sleeves (≈0 → strong diversification)", fontweight="bold")
    ax.set_ylabel("correlation"); ax.legend(fontsize=8)
    fig.savefig(assets / "sleeve_correlation.png"); plt.close(fig)

    # 3 — risk-parity weights over time
    w = res["weights"].dropna()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.stackplot(w.index, w[names[0]], w[names[1]], colors=[GREEN, RED], alpha=.7, labels=names)
    ax.set_ylim(0, 1); ax.set_title("Inverse-Volatility (Risk-Parity) Weights", fontweight="bold")
    ax.set_ylabel("weight"); ax.legend(loc="upper right", fontsize=8)
    fig.savefig(assets / "risk_parity_weights.png"); plt.close(fig)

    # 4 — drawdown comparison
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for r, nm, c in [(comp[names[0]], names[0], GREEN), (comp[names[1]], names[1], RED),
                     (rp, "Risk-parity book", BLUE)]:
        eq = (1 + vol_target(r)).cumprod(); dd = (eq / eq.cummax() - 1) * 100
        ax.plot(dd.index, dd, color=c, lw=1.0 if nm != "Risk-parity book" else 1.6, label=nm)
    ax.set_title("Drawdown — combining decorrelated sleeves shrinks the drawdown", fontweight="bold")
    ax.set_ylabel("drawdown (%)"); ax.legend(loc="lower left", fontsize=8)
    fig.savefig(assets / "drawdown.png"); plt.close(fig)

    # 5 — subperiod robustness (risk-parity)
    sp = subperiod_metrics(vol_target(rp), SUBPERIODS)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(sp.index, sp["sharpe"], color=[BLUE if v >= 0 else RED for v in sp["sharpe"]], alpha=.85)
    ax.axhline(0, color="black", lw=.7)
    for i, v in enumerate(sp["sharpe"]): ax.text(i, v + .03, f"{v:+.2f}", ha="center", fontsize=9)
    ax.set_title("Subperiod Robustness — Risk-Parity Book Sharpe", fontweight="bold"); ax.set_ylabel("Sharpe")
    fig.savefig(assets / "subperiod_robustness.png"); plt.close(fig)

    # 6 — performance table
    rows = {names[0]: performance(vol_target(comp[names[0]])),
            names[1]: performance(vol_target(comp[names[1]])),
            "Equal-weight": performance(vol_target(ew)),
            "Risk-parity book": performance(vol_target(rp))}
    keys = [("sharpe", "Sharpe", "{:+.2f}"), ("cagr", "CAGR", "{:.1%}"), ("vol", "Vol", "{:.1%}"),
            ("max_dd", "Max DD", "{:.1%}"), ("calmar", "Calmar", "{:+.2f}"), ("hit", "Hit", "{:.0%}"),
            ("var95", "VaR95", "{:.2%}"), ("skew", "Skew", "{:+.2f}")]
    table = [[fmt.format(m.get(k, np.nan)) for k, _, fmt in keys] for m in rows.values()]
    fig, ax = plt.subplots(figsize=(10.5, 2.1)); ax.axis("off")
    t = ax.table(cellText=table, rowLabels=list(rows), colLabels=[l for _, l, _ in keys],
                 cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.5)
    ax.set_title("Performance Summary (vol-targeted to 10% annual)", fontweight="bold", pad=14)
    fig.savefig(assets / "performance_summary_table.png", bbox_inches="tight"); plt.close(fig)
