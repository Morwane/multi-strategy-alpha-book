"""Regime-aware allocation across the sleeves, vs the static risk-parity benchmark.

Thesis (testable, desk-credible): a static risk-parity book is ALWAYS on. The VIX
vol-carry sleeve harvests the variance risk premium but bleeds in vol spikes; the
energy stat-arb sleeve is a relative-value diversifier. If the HMM can flag stress
EARLY ENOUGH (net of the switching lag — the "persistence trap"), tilting away from
carry into the diversifier and cutting gross in stress should improve the book's
drawdown/Calmar without giving up much Sharpe.

The honest question this backtest answers is exactly: does that hold out-of-sample,
or does the regime lag eat the benefit vs simply running risk parity?

All weights are decided from the PRIOR day's regime (shift(1)); transaction costs
are charged on realised turnover.
"""
import numpy as np
import pandas as pd

from src.allocate import risk_parity_weights
from src.regime.hmm import CALM, NORMAL, STRESS

# regime -> (sleeve weight dict, gross exposure scalar).
# normal uses inverse-vol risk parity (handled specially below).
REGIME_PLAYBOOK = {
    CALM:   ({"energy_statarb": 0.35, "vix_carry": 0.65}, 1.0),   # carry-on
    NORMAL: ("risk_parity", 1.0),                                 # default RP
    STRESS: ({"energy_statarb": 0.60, "vix_carry": 0.00}, 0.50),  # de-risk carry
}


def regime_weights(comp: pd.DataFrame, regime: pd.Series,
                   rp_window: int = 63) -> pd.DataFrame:
    """Per-day sleeve weights implied by the regime (already scaled by gross)."""
    rp = risk_parity_weights(comp, rp_window)
    w = pd.DataFrame(0.0, index=comp.index, columns=comp.columns)
    reg = regime.reindex(comp.index).ffill()
    for code, spec in REGIME_PLAYBOOK.items():
        mask = reg == code
        if spec[0] == "risk_parity":
            w.loc[mask] = rp.loc[mask].values * spec[1]
        else:
            base, gross = spec
            for sleeve, wt in base.items():
                w.loc[mask, sleeve] = wt * gross
    return w


def regime_book(comp: pd.DataFrame, regime: pd.Series, rp_window: int = 63,
                cost_per_turnover: float = 2e-4) -> dict:
    """Backtest the regime-aware book (look-ahead-free, net of costs)."""
    w_raw = regime_weights(comp, regime, rp_window)
    w = w_raw.shift(1)                                   # decide on yesterday's regime
    gross = (w * comp).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1)                # one-way notional traded
    cost = turnover * cost_per_turnover
    net = (gross - cost)
    first = w.dropna().index[0]
    net.loc[:first] = np.nan                             # no position until weights set
    return {
        "weights": w, "port_gross": gross.rename("regime_gross"),
        "port_net": net.rename("regime_net"), "turnover": turnover,
        "avg_turnover_annual": float(turnover.dropna().mean() * 252),
    }


def _target_weights(code: int, rp_row: pd.Series, cols) -> np.ndarray:
    spec = REGIME_PLAYBOOK[code]
    if spec[0] == "risk_parity":
        return rp_row.values * spec[1]
    base, gross = spec
    return np.array([base.get(c, 0.0) * gross for c in cols])


def confirm_regime(regime: pd.Series, proba: pd.DataFrame,
                   conf: float = 0.60, min_hold: int = 21) -> pd.Series:
    """Hysteresis filter against the persistence trap's evil twin — whipsaw.

    Switch to a new regime only when (a) its filtered posterior clears `conf` AND
    (b) the current regime has been held at least `min_hold` days. Otherwise stay
    put. This trades a little detection lag for far fewer round-trips.
    """
    reg = regime.copy()
    out = pd.Series(index=reg.index, dtype="float")
    current = reg.iloc[0]
    hold = 0
    for i, (dt, r) in enumerate(reg.items()):
        if i > 0:
            p = proba.loc[dt, r] if (dt in proba.index and r in proba.columns) else np.nan
            if r != current and hold >= min_hold and np.isfinite(p) and p >= conf:
                current, hold = r, 0
            else:
                hold += 1
        out.iloc[i] = current
    return out


def disciplined_book(comp: pd.DataFrame, regime: pd.Series, proba: pd.DataFrame,
                     rp_window: int = 63, rebalance_every: int = 21,
                     conf: float = 0.60, min_hold: int = 21,
                     cost_per_turnover: float = 2e-4) -> dict:
    """Regime book with confidence gating, hysteresis and scheduled rebalancing.

    Weights are refreshed only on a fixed cadence OR when the confirmed regime
    changes, and held flat in between — collapsing the turnover that sank the
    naive overlay, while keeping the same regime playbook.
    """
    confirmed = confirm_regime(regime, proba, conf, min_hold).reindex(comp.index).ffill()
    rp = risk_parity_weights(comp, rp_window)
    w = pd.DataFrame(index=comp.index, columns=comp.columns, dtype="float")
    last = None
    prev_code = None
    for i, dt in enumerate(comp.index):
        code = confirmed.loc[dt]
        changed = code != prev_code
        if last is None or changed or (i % rebalance_every == 0):
            if pd.notna(code) and rp.loc[dt].notna().all():
                last = _target_weights(int(code), rp.loc[dt], comp.columns)
        if last is not None:
            w.iloc[i] = last
        prev_code = code

    w = w.shift(1)
    gross = (w * comp).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1)
    net = gross - turnover * cost_per_turnover
    first = w.dropna().index[0]
    net.loc[:first] = np.nan
    return {
        "confirmed": confirmed, "weights": w,
        "port_gross": gross.rename("disc_gross"),
        "port_net": net.rename("disciplined_net"), "turnover": turnover,
        "avg_turnover_annual": float(turnover.dropna().mean() * 252),
    }


THROTTLE = {CALM: 1.0, NORMAL: 1.0, STRESS: 0.50}   # gross scalar by regime


def throttle_book(comp: pd.DataFrame, regime: pd.Series, proba: pd.DataFrame,
                  rp_window: int = 63, rebalance_every: int = 21,
                  conf: float = 0.60, min_hold: int = 21,
                  cost_per_turnover: float = 2e-4) -> dict:
    """Regime as a RISK THROTTLE, not an alpha-timing signal.

    Keep the proven risk-parity sleeve mix ALWAYS on (don't disturb the
    diversification 'free lunch'); use the confirmed regime only to scale gross
    exposure down in stress. The most defensible desk use of regime detection.
    """
    confirmed = confirm_regime(regime, proba, conf, min_hold).reindex(comp.index).ffill()
    rp = risk_parity_weights(comp, rp_window)
    throttle = confirmed.map(THROTTLE)
    w = pd.DataFrame(index=comp.index, columns=comp.columns, dtype="float")
    last = None
    prev = None
    for i, dt in enumerate(comp.index):
        g = throttle.loc[dt]
        changed = g != prev
        if last is None or changed or (i % rebalance_every == 0):
            if rp.loc[dt].notna().all() and pd.notna(g):
                last = rp.loc[dt].values * g
        if last is not None:
            w.iloc[i] = last
        prev = g

    w = w.shift(1)
    gross = (w * comp).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1)
    net = gross - turnover * cost_per_turnover
    first = w.dropna().index[0]
    net.loc[:first] = np.nan
    return {
        "confirmed": confirmed, "weights": w,
        "port_net": net.rename("throttle_net"), "turnover": turnover,
        "avg_turnover_annual": float(turnover.dropna().mean() * 252),
    }


def regime_exposure(regime: pd.Series) -> pd.Series:
    """Gross exposure scalar over time (for the allocation chart)."""
    gross_map = {code: (spec[1] if spec[0] == "risk_parity" else spec[1])
                 for code, spec in REGIME_PLAYBOOK.items()}
    return regime.map(gross_map)
