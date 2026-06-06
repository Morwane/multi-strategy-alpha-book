"""Build regime features from the raw LSEG price panel (data/regime/raw_prices.csv).

Every feature is CAUSAL: rolling windows use only past/current observations, so a
feature dated t never embeds information from t+1. The HMM consumes a compact,
low-collinearity subset (HMM_COLS); the remaining columns are kept for economic
labelling and diagnostics.
"""
from pathlib import Path
import numpy as np
import pandas as pd

TD = 252

# Compact subset fed to the HMM. Fewer, weakly-collinear, economically-meaningful
# features keep a 3-state Gaussian HMM stable on ~4000 daily obs.
HMM_COLS = ["spx_mom20", "rv20", "vix", "vix_chg5", "drawdown", "slope_2s10s"]


def load_raw(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "regime" / "raw_prices.csv",
                     parse_dates=["Date"], index_col="Date")
    return df.sort_index()


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a causal feature frame aligned to the raw price index."""
    px = raw.ffill()                      # carry last price over non-trading gaps
    f = pd.DataFrame(index=px.index)

    spx_ret = np.log(px["SPX"]).diff()
    f["spx_ret"] = spx_ret
    f["spx_mom20"] = spx_ret.rolling(20).mean() * TD          # annualised drift
    f["rv20"] = spx_ret.rolling(20).std() * np.sqrt(TD)       # realised vol 20d
    f["rv60"] = spx_ret.rolling(60).std() * np.sqrt(TD)       # realised vol 60d
    f["skew60"] = spx_ret.rolling(60).skew()
    f["drawdown"] = px["SPX"] / px["SPX"].rolling(252, min_periods=20).max() - 1.0

    # volatility complex
    f["vix"] = px["VIX"]
    f["vix_chg5"] = px["VIX"].diff(5)
    f["vix_ts"] = px["VIX3M"] / px["VIX"] - 1.0               # >0 contango (calm)

    # rates / dollar
    f["slope_2s10s"] = px["US10Y"] - px["US2Y"]              # <0 inversion (late cycle)
    f["us10y_chg20"] = px["US10Y"].diff(20)
    f["dxy_ret20"] = np.log(px["DXY"]).diff(20)

    # energy relative-value (drives the energy stat-arb sleeve)
    f["brent_wti"] = px["Brent"] - px["WTI"]
    crack = (2.0 * px["Gasoline"] + 1.0 * px["HeatingOil"]) * 42.0 - 3.0 * px["WTI"]
    f["crack321"] = crack / 3.0                               # $/bbl
    f["wti_slope"] = px["WTI_c12"] / px["WTI"] - 1.0          # >0 contango

    return f


def coverage_report(f: pd.DataFrame) -> pd.DataFrame:
    """Per-feature availability — measured vs missing, for the methodology section."""
    return pd.DataFrame({
        "non_null": f.notna().sum(),
        "missing_pct": f.isna().mean().mul(100).round(2),
        "start": f.apply(lambda s: s.first_valid_index()),
        "end": f.apply(lambda s: s.last_valid_index()),
    })
