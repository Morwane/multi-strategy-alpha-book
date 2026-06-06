# Multi-Strategy Alpha Book

> Combining **decorrelated alpha sleeves** into one risk-managed book via inverse-volatility **risk parity** — the diversification "free lunch" in action. Built on LSEG-data strategies, look-ahead-free, vol-targeted.

![Equity curve](docs/assets/equity_curve.png)

## Why this project matters

A single strategy, however good, has its own risk and its own bad regimes. The way real multi-strategy desks build robust returns is by **stacking weakly-correlated alpha sources** and allocating risk across them. This repo is the **allocation layer** that does exactly that.

It combines two standalone sleeves I built:

| Sleeve | What it harvests | Repo |
|--------|------------------|------|
| `energy_statarb` | mean-reversion of the 3:2:1 crack & Brent-WTI spreads | [energy-spreads-statarb](https://github.com/Morwane/energy-spreads-statarb) |
| `vix_carry` | the variance risk premium via the VIX-futures roll | [vix-vol-carry](https://github.com/Morwane/vix-vol-carry) |

Their returns are **almost perfectly uncorrelated (ρ = +0.03)** — one trades petroleum spreads, the other equity volatility. That orthogonality is what makes the combination powerful.

## Key result — the diversification free lunch

(2010–2026, vol-targeted to 10% annual)

| Strategy | Sharpe | CAGR | Max DD | Calmar |
|----------|:------:|:----:|:------:|:------:|
| energy_statarb (sleeve) | +0.83 | +8.1% | −16.8% | +0.48 |
| vix_carry (sleeve) | +1.24 | +12.6% | −10.4% | +1.22 |
| Equal-weight | +0.93 | +9.2% | −15.3% | +0.60 |
| **Risk-parity book** | **+1.48** | **+15.4%** | −14.3% | +1.07 |

![Performance summary](docs/assets/performance_summary_table.png)

> The combined book's **Sharpe (1.48) exceeds the best single sleeve (1.24)** — the hallmark of genuine diversification. Risk parity also beats naive equal-weight (1.48 vs 0.93) by sizing each sleeve to contribute equal risk.

![Sleeve correlation](docs/assets/sleeve_correlation.png)

## Method

- **Inputs** — daily return streams of the two sleeves (`data/components/components.csv`).
- **Risk parity** — weights `∝ 1/volatility` from **trailing 63d** vol, `shift(1)` (no look-ahead).
- **Vol targeting** — the book is scaled to 10% annual vol for interpretable risk.
- **Benchmark** — naive equal-weight, to show risk parity adds value.

![Risk-parity weights](docs/assets/risk_parity_weights.png)
![Drawdown](docs/assets/drawdown.png)

## Risk controls & robustness

- Weights use **trailing volatility only**; they sum to 1; portfolio vol-targeted.
- **Subperiod robustness** across four regime eras; automated `quant_checks` + a `pytest` suite (6 tests).

![Subperiod robustness](docs/assets/subperiod_robustness.png)

## Limitations

- Sleeve returns are vol-normalized **research** PnL, not sized dollar books.
- Risk parity uses volatility only (not the full covariance / tail dependence); correlations can rise in crises.
- Research only — **not investment advice**.

## Repository structure

```
multi-strategy-alpha-book/
├── README.md · LICENSE · requirements.txt
├── data/components/components.csv   # sleeve return streams (from the two sleeve repos)
├── src/
│   ├── data.py        # load sleeves
│   ├── allocate.py    # equal-weight, risk parity, quant_checks
│   ├── metrics.py     # vol targeting + performance/risk metrics
│   └── plots.py       # figures
├── scripts/
│   ├── run_backtest.py
│   └── generate_report.py
├── tests/test_allocate.py
├── docs/assets/
└── reports/tearsheet.md
```

## How to run

```bash
pip install -r requirements.txt
python scripts/run_backtest.py        # allocation metrics + integrity checks
python scripts/generate_report.py     # figures + tearsheet
pytest -q
```

*Built with Python (pandas, numpy, matplotlib). Capstone of a multi-strategy volatility & relative-value research portfolio.*
