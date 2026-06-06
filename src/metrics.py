"""Performance & risk metrics. Works on vol-targeted (dollar-like) simple returns."""
import numpy as np
import pandas as pd

TD = 252


def vol_target(ret: pd.Series, target_annual_vol: float = 0.10) -> pd.Series:
    """Scale a (vol-normalized) PnL series to a target annual volatility, so that
    CAGR / VaR / ES / max-DD% become economically interpretable. Sharpe invariant."""
    ret = ret.fillna(0.0)
    realized = ret.std() * np.sqrt(TD)
    if realized == 0:
        return ret
    return ret * (target_annual_vol / realized)


def performance(ret: pd.Series, position: pd.Series | None = None) -> dict:
    """Full metric set on simple daily returns."""
    r = ret.dropna()
    if len(r) < 30 or r.std() == 0:
        return {}
    sharpe = r.mean() / r.std() * np.sqrt(TD)
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (TD / len(r)) - 1
    vol = r.std() * np.sqrt(TD)
    dd = (eq / eq.cummax() - 1)
    max_dd = dd.min()
    var95 = -np.quantile(r, 0.05)
    es95 = -r[r <= np.quantile(r, 0.05)].mean()
    turnover = float(position.diff().abs().mean() * TD) if position is not None else np.nan
    return {
        "sharpe": sharpe, "cagr": cagr, "vol": vol, "max_dd": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "hit": (r[r != 0] > 0).mean(), "skew": r.skew(), "kurt": r.kurt(),
        "var95": var95, "es95": es95, "turnover": turnover, "n": len(r),
    }


def subperiod_metrics(ret: pd.Series, periods) -> pd.DataFrame:
    rows = []
    for label, (a, b) in periods.items():
        sub = ret.loc[a:b].dropna()
        st = performance(sub)
        rows.append({"period": label, "sharpe": st.get("sharpe", np.nan),
                     "cagr": st.get("cagr", np.nan), "max_dd": st.get("max_dd", np.nan),
                     "n": st.get("n", 0)})
    return pd.DataFrame(rows).set_index("period")
