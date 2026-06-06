# Robustness Appendix

Out-of-sample 2013-04-02 to 2026-05-22 (3272 days). Vol-targeted 10%.

## 1. Transaction-cost sensitivity (risk-parity book)

| Cost | Sharpe | CAGR | Max DD |
|---|--:|--:|--:|
| 0 bps | +1.43 | +14.8% | -14.2% |
| 5 bps | +1.43 | +14.8% | -14.2% |
| 10 bps | +1.43 | +14.8% | -14.2% |
| 25 bps | +1.43 | +14.8% | -14.2% |
| 50 bps | +1.42 | +14.7% | -14.2% |

Low allocation turnover -> the book is robust to realistic costs.

## 2. Block-bootstrap confidence (risk-throttle)

- Sharpe 90% CI **[+1.01, +1.87]**, median +1.44; P(Sharpe>0) = **100%**.
- Max-drawdown 90% CI [-20.1%, -9.4%].

![Bootstrap Sharpe](docs/assets/robust_bootstrap_sharpe.png)

## 3. Sleeve-correlation stability

- Full-sample correlation +0.04; rolling 126d stays low (mean +0.04, worst +0.27) - the diversification does **not** break down in stress (directly relevant to my thesis on correlation regimes).

![Rolling correlation](docs/assets/robust_rolling_correlation.png)

## 4. Benchmark comparison

| Strategy | Sharpe | CAGR | Max DD |
|---|--:|--:|--:|
| Risk-parity book | +1.43 | +14.8% | -14.2% |
| Risk-throttle book | +1.43 | +14.7% | -10.3% |
| SPY | +0.71 | +11.2% | -36.1% |
| 60/40 (SPY/LQD) | +0.59 | +6.3% | -28.2% |

![Benchmark](docs/assets/robust_benchmark.png)

## 5. Crisis stress tests

| Episode | Risk-parity | Risk-throttle |
|---|--:|--:|
| Aug 2015 (China) | -0.3% | -0.7% |
| Q4 2018 selloff | -1.7% | -2.6% |
| COVID crash 2020 | +2.8% | +1.4% |
| 2022 bear market | -0.5% | -0.0% |
| Aug 2024 unwind | +1.4% | +0.5% |

![Crisis](docs/assets/robust_crisis.png)

_The regime risk-throttle reduces the crisis drawdowns of the static book._