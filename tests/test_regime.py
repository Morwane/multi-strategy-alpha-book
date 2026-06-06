"""Regime overlay tests — focus on the look-ahead-free guarantees. Run: pytest -q"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.regime.features import load_raw, build_features, HMM_COLS
from src.regime.hmm import walk_forward_regimes, CALM, NORMAL, STRESS
from src.regime.allocate import confirm_regime, throttle_book, disciplined_book

DATA = REPO / "data"


@pytest.fixture(scope="module")
def feats():
    return build_features(load_raw(DATA))


@pytest.fixture(scope="module")
def wf(feats):
    return walk_forward_regimes(feats, HMM_COLS, n_states=3)


def test_features_causal_no_future_leak(feats):
    # truncating the input must not change features on the surviving dates
    raw = load_raw(DATA)
    cut = raw.index[-200]
    full = build_features(raw)["rv20"]
    trunc = build_features(raw.loc[:cut])["rv20"]
    common = trunc.dropna().index.intersection(full.dropna().index)
    assert np.allclose(full.loc[common], trunc.loc[common], equal_nan=True)


def test_regimes_are_out_of_sample(wf):
    raw = load_raw(DATA)
    assert wf.regime.index[0] > raw.index[0]            # nothing before train window
    assert set(wf.regime.unique()) <= {CALM, NORMAL, STRESS}


def test_regime_economics_ordered(wf, feats):
    # stress must carry higher realised vol & VIX than calm — the sanity anchor
    f = feats.loc[wf.regime.index]
    by = f.groupby(wf.regime)
    assert by["rv20"].mean()[STRESS] > by["rv20"].mean()[CALM]
    assert by["vix"].mean()[STRESS] > by["vix"].mean()[CALM]


def test_confirm_reduces_switching(wf):
    conf = confirm_regime(wf.regime, wf.proba)
    raw_switches = (wf.regime.diff() != 0).sum()
    conf_switches = (conf.diff() != 0).sum()
    assert conf_switches < raw_switches                 # hysteresis cuts whipsaw


def test_throttle_lowers_turnover_vs_disciplined(wf):
    comp = pd.read_csv(DATA / "components" / "components.csv",
                       parse_dates=["Date"], index_col="Date").dropna()
    common = comp.index.intersection(wf.regime.index)
    tb = throttle_book(comp.loc[common], wf.regime.loc[common], wf.proba.loc[common])
    db = disciplined_book(comp.loc[common], wf.regime.loc[common], wf.proba.loc[common])
    assert tb["port_net"].isna().iloc[0]                # weights lagged (shift(1))
    assert tb["avg_turnover_annual"] < db["avg_turnover_annual"]
