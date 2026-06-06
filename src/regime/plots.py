"""Figures for the regime overlay → docs/assets/regime_*.png."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics import vol_target, performance
from src.regime.hmm import CALM, NORMAL, STRESS, LABELS

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10,
                     "figure.autolayout": True})
BLUE, RED, GREEN, GREY, ORANGE = "#1f5fa8", "#c0392b", "#27ae60", "#7f8c8d", "#e67e22"
REG_COLOR = {CALM: "#27ae60", NORMAL: "#f1c40f", STRESS: "#c0392b"}


def _regime_bands(ax, regime: pd.Series):
    """Shade the background by regime."""
    reg = regime.dropna()
    idx = reg.index
    for code, color in REG_COLOR.items():
        mask = (reg == code).values
        ax.fill_between(idx, 0, 1, where=mask, transform=ax.get_xaxis_transform(),
                        color=color, alpha=0.12, step="post", lw=0)


def make_regime_figures(spx: pd.Series, regime: pd.Series, proba: pd.DataFrame,
                        returns: dict, throttle_regime: pd.Series,
                        assets: Path) -> None:
    assets = Path(assets); assets.mkdir(parents=True, exist_ok=True)
    valid = next(iter(returns.values())).index

    # 1 — SPX with regime shading
    fig, ax = plt.subplots(figsize=(11, 4.2))
    s = spx.reindex(regime.index).ffill()
    ax.plot(s.index, s.values, color="black", lw=0.9)
    _regime_bands(ax, regime)
    ax.set_yscale("log")
    ax.set_title("S&P 500 with HMM regimes (green calm · yellow normal · red stress)",
                 fontweight="bold")
    ax.set_ylabel("S&P 500 (log)")
    fig.savefig(assets / "regime_timeline.png"); plt.close(fig)

    # 2 — regime probability stacked area
    fig, ax = plt.subplots(figsize=(11, 3.2))
    p = proba.reindex(regime.index).ffill().fillna(0)
    ax.stackplot(p.index, p[CALM], p[NORMAL], p[STRESS],
                 colors=[REG_COLOR[CALM], REG_COLOR[NORMAL], REG_COLOR[STRESS]],
                 labels=["calm", "normal", "stress"], alpha=0.85)
    ax.set_ylim(0, 1); ax.set_title("Filtered regime probabilities (walk-forward)",
                                    fontweight="bold")
    ax.set_ylabel("probability"); ax.legend(loc="upper left", ncol=3, fontsize=8)
    fig.savefig(assets / "regime_probabilities.png"); plt.close(fig)

    # 3 — gross exposure (risk throttle) over time
    from src.regime.allocate import THROTTLE
    expo = throttle_regime.reindex(regime.index).ffill().map(THROTTLE)
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.fill_between(expo.index, 0, expo.values, color=BLUE, alpha=0.5, step="post")
    ax.plot(expo.index, expo.values, color=BLUE, lw=0.8, drawstyle="steps-post")
    _regime_bands(ax, regime)
    ax.set_ylim(0, 1.15)
    ax.set_title("Risk-throttle gross exposure (cut to 50% in confirmed stress)",
                 fontweight="bold")
    ax.set_ylabel("gross exposure")
    fig.savefig(assets / "regime_exposure.png"); plt.close(fig)

    # 4 — cumulative PnL: benchmark vs the three overlay variants
    styling = {
        "Risk-parity (benchmark)": (BLUE, 2.0),
        "Regime naive (net)": (GREY, 1.0),
        "Regime disciplined (net)": (ORANGE, 1.0),
        "Regime risk-throttle (net)": (GREEN, 1.8),
    }
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for nm, (c, lw) in styling.items():
        if nm not in returns:
            continue
        eq = (1 + returns[nm]).cumprod()
        ax.plot(eq.index, (eq - 1) * 100, color=c, lw=lw,
                label=f"{nm}  (Sharpe {performance(returns[nm])['sharpe']:+.2f})")
    ax.set_title("Cumulative return — regime overlay variants vs static risk-parity "
                 "(vol-targeted 10%)", fontweight="bold")
    ax.set_ylabel("cumulative return (%)"); ax.legend(loc="upper left", fontsize=8.5)
    fig.savefig(assets / "regime_cumulative_pnl.png"); plt.close(fig)

    # 5 — drawdown: benchmark vs throttle
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for nm, c, lw in [("Risk-parity (benchmark)", BLUE, 1.6),
                      ("Regime risk-throttle (net)", GREEN, 1.6)]:
        eq = (1 + returns[nm]).cumprod(); dd = (eq / eq.cummax() - 1) * 100
        ax.plot(dd.index, dd, color=c, lw=lw, label=nm)
    ax.set_title("Drawdown — risk throttle shrinks the deepest drawdowns",
                 fontweight="bold")
    ax.set_ylabel("drawdown (%)"); ax.legend(loc="lower left", fontsize=8.5)
    fig.savefig(assets / "regime_drawdown.png"); plt.close(fig)

    # 6 — rolling 1y Sharpe: benchmark vs throttle
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for nm, c in [("Risk-parity (benchmark)", BLUE), ("Regime risk-throttle (net)", GREEN)]:
        r = returns[nm]
        rs = r.rolling(252).mean() / r.rolling(252).std() * np.sqrt(252)
        ax.plot(rs.index, rs, color=c, lw=1.2, label=nm)
    ax.axhline(0, color="black", lw=.7)
    ax.set_title("Rolling 1y Sharpe", fontweight="bold")
    ax.set_ylabel("Sharpe"); ax.legend(loc="upper left", fontsize=8.5)
    fig.savefig(assets / "regime_rolling_sharpe.png"); plt.close(fig)

    # 7 — performance table
    keys = [("sharpe", "Sharpe", "{:+.2f}"), ("cagr", "CAGR", "{:.1%}"),
            ("vol", "Vol", "{:.1%}"), ("max_dd", "Max DD", "{:.1%}"),
            ("calmar", "Calmar", "{:+.2f}"), ("hit", "Hit", "{:.0%}")]
    rows = {nm: performance(r) for nm, r in returns.items()}
    table = [[fmt.format(m.get(k, np.nan)) for k, _, fmt in keys] for m in rows.values()]
    fig, ax = plt.subplots(figsize=(10.5, 2.6)); ax.axis("off")
    t = ax.table(cellText=table, rowLabels=list(rows),
                 colLabels=[l for _, l, _ in keys], cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.5)
    ax.set_title("Regime overlay — performance summary (vol-targeted 10%)",
                 fontweight="bold", pad=14)
    fig.savefig(assets / "regime_performance_table.png", bbox_inches="tight")
    plt.close(fig)
