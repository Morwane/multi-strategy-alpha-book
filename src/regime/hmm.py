"""Walk-forward Gaussian HMM regime detection — strictly look-ahead-free.

The hard part of regime models is honest out-of-sample use. Here:

  * The HMM is REFIT periodically on an EXPANDING window of past data only.
  * Standardisation uses TRAIN-window mean/std only (no full-sample scaling).
  * Each day's regime is the Viterbi state decoded from a trailing window that
    ENDS at that day — observation t never sees t+1.
  * State indices from hmmlearn are arbitrary and switch between refits, so we
    relabel every refit to fixed ECONOMIC codes (0 calm, 1 normal, 2 stress)
    from each state's emission means. This kills label-switching.

The output regime series therefore reflects, at every date, only information a
trader would have had on that date.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

CALM, NORMAL, STRESS = 0, 1, 2
LABELS = {CALM: "calm", NORMAL: "normal", STRESS: "stress"}


@dataclass
class WalkForwardResult:
    regime: pd.Series                 # economic-coded regime per date (causal)
    proba: pd.DataFrame               # filtered prob of each economic regime
    refit_log: list = field(default_factory=list)


def _economic_label_map(model: GaussianHMM, cols: list[str]) -> dict[int, int]:
    """Map raw HMM state index -> economic code from emission means (train space).

    Stress = high vol/VIX and deep drawdown; calm = the opposite. Robust to the
    arbitrary ordering hmmlearn assigns, so labels are stable across refits.
    """
    means = pd.DataFrame(model.means_, columns=cols)
    z = (means - means.mean()) / means.std(ddof=0).replace(0, 1.0)
    score = z.get("rv20", 0) + z.get("vix", 0) + z.get("vix_chg5", 0) - z.get("drawdown", 0)
    order = score.sort_values().index.tolist()    # low stress -> high stress
    codes = [CALM, NORMAL, STRESS] if len(order) == 3 else list(range(len(order)))
    return {raw_state: codes[rank] for rank, raw_state in enumerate(order)}


def walk_forward_regimes(
    features: pd.DataFrame,
    cols: list[str],
    n_states: int = 3,
    train_min: int = 756,        # ~3y before first prediction
    refit_every: int = 63,       # refit quarterly
    decode_window: int = 504,    # trailing window for causal Viterbi decode
    seed: int = 42,
) -> WalkForwardResult:
    X_all = features[cols].dropna()
    idx = X_all.index
    n = len(idx)
    if n <= train_min:
        raise ValueError(f"Not enough observations ({n}) for train_min={train_min}")

    regime = pd.Series(index=idx, dtype="float")
    proba = pd.DataFrame(index=idx, columns=[CALM, NORMAL, STRESS], dtype="float")
    refit_log: list = []

    model: GaussianHMM | None = None
    label_map: dict[int, int] = {}
    mu = sd = None

    for i in range(train_min, n):
        if model is None or (i - train_min) % refit_every == 0:
            train = X_all.iloc[:i]                       # expanding, past only
            mu, sd = train.mean(), train.std(ddof=0).replace(0, 1.0)
            Z = ((train - mu) / sd).values
            model = GaussianHMM(n_components=n_states, covariance_type="full",
                                n_iter=200, tol=1e-3, random_state=seed)
            model.fit(Z)
            label_map = _economic_label_map(model, cols)
            refit_log.append({"date": idx[i], "train_n": len(train),
                              "converged": bool(model.monitor_.converged)})

        lo = max(0, i - decode_window + 1)
        Z_win = ((X_all.iloc[lo:i + 1] - mu) / sd).values  # ends at day i (causal)
        states = model.predict(Z_win)                       # Viterbi up to i
        post = model.predict_proba(Z_win)[-1]               # filtered posterior at i
        raw_state = states[-1]
        regime.iloc[i] = label_map[raw_state]
        for raw, code in label_map.items():
            proba.iloc[i, code] = post[raw]

    return WalkForwardResult(regime=regime.dropna(), proba=proba.dropna(how="all"),
                             refit_log=refit_log)


def regime_stats(features: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
    """Mean feature values conditional on regime — the economic sanity check."""
    cols = ["rv20", "vix", "drawdown", "spx_mom20", "slope_2s10s", "vix_ts"]
    cols = [c for c in cols if c in features.columns]
    df = features.loc[regime.index, cols].copy()
    df["regime"] = regime.map(LABELS)
    out = df.groupby("regime").agg(["mean"])
    out.columns = [c for c, _ in out.columns]
    counts = regime.map(LABELS).value_counts()
    out["days"] = counts
    out["share_%"] = (counts / counts.sum() * 100).round(1)
    return out
