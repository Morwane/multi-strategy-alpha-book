# Regime overlay — tearsheet

Walk-forward Gaussian HMM (3 states), out-of-sample 2013-04-02 → 2026-05-22 (3272 days), 56 quarterly refits, net of 2bps/turnover costs.

| Strategy | Sharpe | CAGR | Vol | MaxDD | Calmar | Turn/yr |
|---|---|---|---|---|---|---|
| Risk-parity (benchmark) | +1.43 | +14.8% | 10.0% | -14.2% | +1.04 | — |
| Regime naive | +1.04 | +10.4% | 10.0% | -16.6% | +0.63 | 40.4 |
| Regime disciplined | +0.87 | +8.5% | 10.0% | -15.2% | +0.56 | 4.3 |
| **Regime risk-throttle** | +1.43 | +14.7% | 10.0% | -10.3% | +1.43 | 1.7 |

## Finding
- Regime as an **alpha-timing / reallocation** signal (naive, disciplined) **underperforms** static risk-parity: the switching lag (persistence trap) and breaking the diversification mix destroy value.
- Regime as a **risk throttle** (keep the risk-parity mix, cut gross in confirmed stress) keeps Sharpe (+1.43 vs +1.43) while cutting max drawdown (-10.3% vs -14.2%) and lifting Calmar (+1.43 vs +1.04) at trivial turnover (1.7x/yr).
- **Lesson:** regime models here are useful for risk management, not return timing.

## Regime economics (conditional feature means)

| regime   |   rv20 |    vix |   drawdown |   spx_mom20 |   slope_2s10s |   vix_ts |   days |   share_% |
|:---------|-------:|-------:|-----------:|------------:|--------------:|---------:|-------:|----------:|
| calm     |  0.097 | 14.319 |     -0.009 |       0.212 |         0.842 |    0.17  |   1134 |      34   |
| normal   |  0.127 | 16.216 |     -0.035 |       0.153 |         0.842 |    0.13  |   1203 |      36.1 |
| stress   |  0.205 | 23.38  |     -0.07  |      -0.029 |         0.41  |    0.079 |   1000 |      30   |
