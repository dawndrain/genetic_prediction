import math

import numpy as np
import pytest

from genepred import qaly


def test_liability_threshold_at_zero_r2():
    # at R²=0 the score is uninformative → risk == prevalence for any z
    for prev in (0.01, 0.1, 0.36):
        assert qaly.liability_threshold_risk(1.5, prev, 0.0) == pytest.approx(prev)
        assert qaly.liability_threshold_risk(-2.0, prev, 0.0) == pytest.approx(prev)


def test_liability_threshold_averages_to_prevalence():
    # E_z[risk(z)] = prevalence (the score doesn't change population
    # rate, only redistributes it). Check by Monte Carlo.
    rng = np.random.default_rng(0)
    z = rng.normal(size=100_000)
    for prev, r2 in ((0.05, 0.10), (0.20, 0.05)):
        risks = [qaly.liability_threshold_risk(float(zi), prev, r2) for zi in z[:5000]]
        assert abs(np.mean(risks) - prev) < 0.005


def test_liability_threshold_monotone():
    risks = [
        qaly.liability_threshold_risk(z, 0.1, 0.05)
        for z in (-2, -1, 0, 1, 2)
    ]
    assert risks == sorted(risks)


def test_ancestry_ratio_table():
    assert set(qaly.ANCESTRY_R2_RATIO) == {"EUR", "AMR", "SAS", "EAS", "AFR"}
    assert qaly.ANCESTRY_R2_RATIO["EUR"] == 1.0
    assert all(0 < r <= 1 for r in qaly.ANCESTRY_R2_RATIO.values())


def test_simulate_selection_sane_range():
    r = qaly.simulate_selection(n_embryos=5, n_simulations=2_000, seed=0)
    assert 0.3 < r["qaly_gain_mean"] < 1.0
    assert r["qaly_gain_p10"] < r["qaly_gain_mean"] < r["qaly_gain_p90"]


def test_simulate_selection_monotone_in_n():
    g = [
        qaly.simulate_selection(n_embryos=n, n_simulations=2_000, seed=0)[
            "qaly_gain_mean"
        ]
        for n in (2, 5, 10)
    ]
    assert g[0] < g[1] < g[2]


def test_ancestry_ratio_scales_gain():
    eur = qaly.simulate_selection(
        n_embryos=5, n_simulations=2_000, seed=0, ancestry_ratio=1.0
    )["qaly_gain_mean"]
    afr = qaly.simulate_selection(
        n_embryos=5, n_simulations=2_000, seed=0, ancestry_ratio=0.30
    )["qaly_gain_mean"]
    # gain ~ √R², so AFR ≈ √0.30 × EUR ≈ 0.55×; allow slack for
    # liability nonlinearity.
    assert 0.45 < afr / eur < 0.65


def test_disease_costs_split():
    for k, t in qaly.DISEASE_TRAITS.items():
        med, prod = t.costs()
        assert med >= 0 and prod >= 0, k
        assert math.isclose(t.cost_societal(), med + prod)
        assert t.cost_personal() <= t.cost_societal()


def test_genetic_correlation_matrix_psd():
    # The simulation needs a PSD correlation matrix to draw from.
    # If someone edits the rg table without keeping it consistent,
    # this catches it.
    R = qaly.build_genetic_correlation_matrix()
    assert R.shape[0] == R.shape[1]
    assert np.allclose(R, R.T)
    eig = np.linalg.eigvalsh(R)
    assert eig.min() > -1e-6, f"rg matrix not PSD; min eigenvalue {eig.min():.3g}"
