# Multi-Strategy Alpha Book — Tearsheet

Period: **2010-03-31 → 2026-05-22** (4026 days). Returns vol-targeted to 10% annual.

## Executive summary
Combines two **decorrelated** alpha sleeves — energy-spreads stat-arb and VIX vol-carry (sleeve correlation **+0.03**) — via inverse-volatility risk parity. The combined book reaches **Sharpe +1.48** (vs best single sleeve +1.24) with max drawdown -14.3% — the diversification 'free lunch' in action.

## Sleeves (inputs)
- `energy_statarb` — 3:2:1 crack + Brent-WTI mean-reversion (project: energy-spreads-statarb).
- `vix_carry` — VIX-futures vol-carry with crash filter (project: vix-vol-carry).

## Allocation method
- **Equal-weight** (naive benchmark).
- **Risk parity** — inverse-volatility weights from trailing 63d vol (shift(1), no look-ahead).
- Portfolio vol-targeted to 10% annual for interpretable risk.

## Performance
| Strategy | Sharpe | CAGR | Vol | Max DD | Calmar | Hit |
|---|--:|--:|--:|--:|--:|--:|
| energy_statarb | +0.83 | +8.1% | 10.0% | -16.8% | +0.48 | 51% |
| vix_carry | +1.24 | +12.6% | 10.0% | -10.4% | +1.22 | 58% |
| Equal-weight | +0.93 | +9.2% | 10.0% | -15.3% | +0.60 | 52% |
| Risk-parity book | +1.48 | +15.4% | 10.0% | -14.3% | +1.07 | 56% |

![Equity](../docs/assets/equity_curve.png)
![Sleeve correlation](../docs/assets/sleeve_correlation.png)
![Weights](../docs/assets/risk_parity_weights.png)
![Drawdown](../docs/assets/drawdown.png)

## Robustness — subperiods
| Period | Sharpe | CAGR | Max DD |
|---|--:|--:|--:|
| 2010-2014 | +1.39 | +13.1% | -9.1% |
| 2015-2019 | +1.52 | +16.2% | -9.9% |
| 2020-2022 | +1.12 | +13.4% | -14.3% |
| 2023-2026 | +1.99 | +19.1% | -9.4% |

![Subperiods](../docs/assets/subperiod_robustness.png)

## Risk controls
- Weights from trailing vol only (no look-ahead); weights sum to 1; portfolio vol-targeted.
- Automated `quant_checks` + pytest suite.

## Limitations
- Sleeve returns are vol-normalized research PnL, not sized dollar books; correlations can rise in crises.
- Risk parity uses volatility only (not full covariance / tail dependence).
- Research only, not investment advice.

## Next steps
- Add more sleeves (dispersion, carry, trend) → the book improves as decorrelated sleeves are added.
- Upgrade to Hierarchical Risk Parity (HRP) and add a regime overlay.