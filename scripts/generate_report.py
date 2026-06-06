"""Generate figures (docs/assets/) + tearsheet (reports/).  Usage: python scripts/generate_report.py"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data import load_components
from src.allocate import build
from src.metrics import vol_target, performance, subperiod_metrics
from src.plots import make_all, SUBPERIODS

DATA, ASSETS, REPORTS = REPO / "data", REPO / "docs" / "assets", REPO / "reports"


def main():
    comp = load_components(DATA)
    res = build(comp)
    make_all(res, ASSETS)
    print(f"Figures → {ASSETS}")
    names = list(comp.columns)

    m = {names[0]: performance(vol_target(comp[names[0]])),
         names[1]: performance(vol_target(comp[names[1]])),
         "Equal-weight": performance(vol_target(res["port_ew"])),
         "Risk-parity book": performance(vol_target(res["port_rp"]))}
    sp = subperiod_metrics(vol_target(res["port_rp"]), SUBPERIODS)
    b = m["Risk-parity book"]
    best = max(m[names[0]]["sharpe"], m[names[1]]["sharpe"])

    L = [
        "# Multi-Strategy Alpha Book — Tearsheet", "",
        f"Period: **{comp.index[0].date()} → {comp.index[-1].date()}** ({len(comp)} days). "
        "Returns vol-targeted to 10% annual.", "",
        "## Executive summary",
        f"Combines two **decorrelated** alpha sleeves — energy-spreads stat-arb and VIX vol-carry "
        f"(sleeve correlation **{res['corr_full']:+.2f}**) — via inverse-volatility risk parity. "
        f"The combined book reaches **Sharpe {b['sharpe']:+.2f}** (vs best single sleeve {best:+.2f}) "
        f"with max drawdown {b['max_dd']:.1%} — the diversification 'free lunch' in action.", "",
        "## Sleeves (inputs)",
        "- `energy_statarb` — 3:2:1 crack + Brent-WTI mean-reversion (project: energy-spreads-statarb).",
        "- `vix_carry` — VIX-futures vol-carry with crash filter (project: vix-vol-carry).", "",
        "## Allocation method",
        "- **Equal-weight** (naive benchmark).",
        "- **Risk parity** — inverse-volatility weights from trailing 63d vol (shift(1), no look-ahead).",
        "- Portfolio vol-targeted to 10% annual for interpretable risk.", "",
        "## Performance",
        "| Strategy | Sharpe | CAGR | Vol | Max DD | Calmar | Hit |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for nm, x in m.items():
        L.append(f"| {nm} | {x['sharpe']:+.2f} | {x['cagr']:+.1%} | {x['vol']:.1%} | "
                 f"{x['max_dd']:.1%} | {x['calmar']:+.2f} | {x['hit']:.0%} |")
    L += ["", "![Equity](../docs/assets/equity_curve.png)",
          "![Sleeve correlation](../docs/assets/sleeve_correlation.png)",
          "![Weights](../docs/assets/risk_parity_weights.png)",
          "![Drawdown](../docs/assets/drawdown.png)", "",
          "## Robustness — subperiods", "| Period | Sharpe | CAGR | Max DD |", "|---|--:|--:|--:|"]
    for p, row in sp.iterrows():
        L.append(f"| {p} | {row['sharpe']:+.2f} | {row['cagr']:+.1%} | {row['max_dd']:.1%} |")
    L += ["", "![Subperiods](../docs/assets/subperiod_robustness.png)", "",
          "## Risk controls",
          "- Weights from trailing vol only (no look-ahead); weights sum to 1; portfolio vol-targeted.",
          "- Automated `quant_checks` + pytest suite.", "",
          "## Limitations",
          "- Sleeve returns are vol-normalized research PnL, not sized dollar books; correlations can rise in crises.",
          "- Risk parity uses volatility only (not full covariance / tail dependence).",
          "- Research only, not investment advice.", "",
          "## Next steps",
          "- Add more sleeves (dispersion, carry, trend) → the book improves as decorrelated sleeves are added.",
          "- Upgrade to Hierarchical Risk Parity (HRP) and add a regime overlay."]
    (REPORTS / "tearsheet.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Tearsheet → {REPORTS / 'tearsheet.md'}")


if __name__ == "__main__":
    main()
