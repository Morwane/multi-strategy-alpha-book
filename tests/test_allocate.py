"""Unit tests — run with: pytest -q"""
import sys
from pathlib import Path
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data import load_components
from src.allocate import build, risk_parity_weights, quant_checks

DATA = REPO / "data"


@pytest.fixture(scope="module")
def comp():
    return load_components(DATA)


@pytest.fixture(scope="module")
def res(comp):
    return build(comp)


def test_two_sleeves(comp):
    assert comp.shape[1] == 2 and len(comp) > 1000


def test_sleeves_decorrelated(res):
    assert abs(res["corr_full"]) < 0.4


def test_weights_sum_to_one(comp):
    w = risk_parity_weights(comp).dropna()
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-9)


def test_weights_lagged(res):
    assert res["port_rp"].isna().iloc[0]


def test_diversification_benefit(res, comp):
    # combined risk-parity Sharpe should beat the worse sleeve (usually both)
    from src.metrics import vol_target, performance
    s = [performance(vol_target(comp[c]))["sharpe"] for c in comp.columns]
    book = performance(vol_target(res["port_rp"]))["sharpe"]
    assert book >= min(s)            # diversification never hurts vs the worst sleeve


def test_all_quant_checks_pass(comp, res):
    assert all(p for _, p, _ in quant_checks(comp, res))
