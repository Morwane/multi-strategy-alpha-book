"""Multi-strategy allocation: combine decorrelated alpha sleeves into one book.

Inputs are the daily return streams of standalone strategies (here: an energy
spreads stat-arb sleeve and a VIX vol-carry sleeve). The allocator's job is the
RISK LAYER:
  - equal-weight (naive benchmark)
  - inverse-volatility risk parity (each sleeve contributes equal risk)
  - portfolio-level vol targeting

No look-ahead: risk-parity weights use TRAILING volatility (shift(1)) and are
applied to the next day's returns.
"""
import numpy as np
import pandas as pd


def equal_weight(comp: pd.DataFrame) -> pd.Series:
    return comp.mean(axis=1).rename("equal_weight")


def risk_parity_weights(comp: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Inverse-vol weights from trailing volatility (look-ahead-free)."""
    vol = comp.rolling(window).std().shift(1)        # yesterday's vol → decide today's weight
    inv = 1.0 / vol
    w = inv.div(inv.sum(axis=1), axis=0)
    return w


def risk_parity(comp: pd.DataFrame, window: int = 63) -> tuple[pd.Series, pd.DataFrame]:
    w = risk_parity_weights(comp, window)
    port = (w * comp).sum(axis=1)
    port[w.isna().any(axis=1)] = np.nan              # no position until weights defined
    return port.rename("risk_parity"), w


def build(comp: pd.DataFrame, window: int = 63) -> dict:
    ew = equal_weight(comp)
    rp, w = risk_parity(comp, window)
    return {
        "comp": comp, "weights": w,
        "port_ew": ew, "port_rp": rp,
        "corr_full": float(comp.corr().iloc[0, 1]),
        "roll_corr": comp.iloc[:, 0].rolling(126).corr(comp.iloc[:, 1]),
    }


def quant_checks(comp: pd.DataFrame, res: dict) -> list[tuple[str, bool, str]]:
    checks = []
    checks.append(("components_loaded", comp.shape[1] == 2 and len(comp) > 1000,
                   f"{comp.shape[1]} sleeves, {len(comp)} days"))
    checks.append(("weights_sum_to_one",
                   bool(np.allclose(res["weights"].dropna().sum(axis=1), 1.0, atol=1e-9)),
                   "risk-parity weights normalized"))
    checks.append(("weights_lagged", bool(res["port_rp"].isna().iloc[0]),
                   "first portfolio return NaN (trailing-vol weights)"))
    checks.append(("low_sleeve_correlation", abs(res["corr_full"]) < 0.4,
                   f"sleeve corr {res['corr_full']:+.2f}"))
    return checks
