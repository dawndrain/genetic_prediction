"""Regression tests for genepred.embryo.

The HMM tests build a small synthetic chromosome (no 1KG dependency)
so they run in CI without data downloads. They pin recovery accuracy
on fixed seeds to within a tolerance, so any silent breakage of the
WHT / numba / linear-rescaling machinery shows up as a test failure.
"""

import numpy as np
import pytest

from genepred import embryo as E
from genepred import pharmgx as px
from genepred import rare_variants as rv


def _synthetic_parents(M: int = 20_000, het_rate: float = 0.03, seed: int = 0):
    """A toy chromosome with realistic het density and spacing."""
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.integers(200, 3000, size=M)).astype(np.int64)
    af = rng.beta(0.4, 6.0, size=M)  # mostly rare, like real data
    pat = (rng.random((2, M)) < af).astype(np.int8)
    mat = (rng.random((2, M)) < af).astype(np.int8)
    # Force a target het rate so the test is stable across seeds.
    want = int(M * het_rate)
    for hap in (pat, mat):
        het_idx = np.flatnonzero(hap[0] != hap[1])
        if len(het_idx) < want:
            extra = rng.choice(
                np.flatnonzero(hap[0] == hap[1]), want - len(het_idx), replace=False
            )
            hap[0, extra] = 0
            hap[1, extra] = 1
    return E.Parents(
        chrom="22",
        pos=pos,
        ref=np.full(M, "A", dtype="<U1"),
        alt=np.full(M, "G", dtype="<U1"),
        pat=pat,
        mat=mat,
    )


@pytest.fixture(scope="module")
def par():
    return _synthetic_parents()


def _make_embryos(par, n_emb, cov, seed_base=1):
    truth = []
    biop = []
    for e in range(n_emb):
        g, _, _ = E.simulate_child(par, np.random.default_rng((seed_base, e)))
        truth.append(g)
        biop.append(
            E.simulate_biopsy(g, cov, 0.01, np.random.default_rng((seed_base + 1, e)))
        )
    return np.stack(truth), biop


def test_switch_errors_preserve_genotype(par):
    rng = np.random.default_rng(7)
    obs, sp, sm = E.apply_switch_errors(par, 0.02, rng)
    assert (obs.pat.sum(0) == par.pat.sum(0)).all()
    assert (obs.mat.sum(0) == par.mat.sum(0)).all()
    # at least some switches happened
    assert (np.diff(sp.astype(int)) != 0).sum() > 5
    assert (np.diff(sm.astype(int)) != 0).sum() > 5


def test_hmm_oracle_near_perfect(par):
    """With true parental phase, the 4-state HMM should recover the
    embryo genotype almost exactly at moderate coverage."""
    truth, biop = _make_embryos(par, 3, cov=0.2)
    ctx = E.build_hmm_context(par)
    concs = [
        (E.hmm_recover(par, ctx, *biop[e], 0.01)[0] == truth[e]).mean()
        for e in range(3)
    ]
    assert min(concs) > 0.995, concs


def test_naive_vs_joint_under_phasing_error(par):
    """At SER=1%, naive HMM should be badly degraded at parent-het
    sites and the joint should recover most of the gap. Pinned to a
    fixed seed; tolerances are generous so this isn't flaky."""
    het = (par.pat[0] != par.pat[1]) | (par.mat[0] != par.mat[1])
    obs, _, _ = E.apply_switch_errors(par, 0.01, np.random.default_rng(0))
    truth, biop = _make_embryos(par, 5, cov=0.05)

    ctx_o = E.build_hmm_context(obs)
    naive = np.stack(
        [E.hmm_recover(obs, ctx_o, *biop[e], 0.01)[0] for e in range(5)]
    )
    naive_hc = (naive[:, het] == truth[:, het]).mean()

    _, dose, var, _, _ = E.joint_recover(obs, biop, 0.01, switch_rate=0.01, n_iter=2)
    joint_hc = (np.round(dose[:, het]) == truth[:, het]).mean()

    assert naive_hc < 0.80, f"naive het-conc {naive_hc:.3%} unexpectedly high"
    assert joint_hc > 0.85, f"joint het-conc {joint_hc:.3%} too low"
    assert joint_hc - naive_hc > 0.10, (
        f"joint should beat naive by ≥10pp; got {joint_hc - naive_hc:.3%}"
    )
    # Posterior variance is non-trivial at het sites and zero at hom.
    assert var[:, het].mean() > 0.01
    assert var[:, ~het].max() == 0.0


def test_joint_full_at_least_as_good(par):
    """The exact both-parents joint should never be worse than the
    coordinate-ascent approximation (modulo noise)."""
    het = (par.pat[0] != par.pat[1]) | (par.mat[0] != par.mat[1])
    obs, _, _ = E.apply_switch_errors(par, 0.01, np.random.default_rng(0))
    truth, biop = _make_embryos(par, 4, cov=0.02)

    _, d1, *_ = E.joint_recover(obs, biop, 0.01, switch_rate=0.01)
    _, d2, *_ = E.joint_recover_full(obs, biop, 0.01, switch_rate=0.01)
    h1 = (np.round(d1[:, het]) == truth[:, het]).mean()
    h2 = (np.round(d2[:, het]) == truth[:, het]).mean()
    assert h2 >= h1 - 0.01, f"full {h2:.3%} should be ≥ per-parent {h1:.3%}"


def test_sibling_recovers_oracle(par):
    """A 30× born sibling appended to the joint HMM should bring
    recovery back to roughly the oracle level."""
    het = (par.pat[0] != par.pat[1]) | (par.mat[0] != par.mat[1])
    obs, _, _ = E.apply_switch_errors(par, 0.01, np.random.default_rng(0))
    truth, biop = _make_embryos(par, 4, cov=0.05)

    sib_g, _, _ = E.simulate_child(par, np.random.default_rng((99,)))
    sib = E.simulate_biopsy(sib_g, 30.0, 0.01, np.random.default_rng((98,)))

    ctx_t = E.build_hmm_context(par)
    oracle_hc = np.mean(
        [
            (E.hmm_recover(par, ctx_t, *biop[e], 0.01)[0][het] == truth[e][het]).mean()
            for e in range(4)
        ]
    )
    _, d_no, *_ = E.joint_recover(obs, biop, 0.01, switch_rate=0.01)
    _, d_sib, *_ = E.joint_recover(obs, biop + [sib], 0.01, switch_rate=0.01)
    h_no = (np.round(d_no[:4, het]) == truth[:, het]).mean()
    h_sib = (np.round(d_sib[:4, het]) == truth[:, het]).mean()
    assert h_sib >= oracle_hc - 0.02, (
        f"with sibling {h_sib:.3%} should be ≈ oracle {oracle_hc:.3%}"
    )
    assert h_sib - h_no > 0.05, f"sibling should add ≥5pp; got {h_sib - h_no:.3%}"


def test_embryo_count_guard(par):
    biop = [(np.zeros(len(par.pos), np.int64),) * 2] * 20
    with pytest.raises(ValueError, match="states"):
        E.joint_recover(par, biop, 0.01)
    with pytest.raises(ValueError, match="impractical"):
        E.joint_recover_full(par, biop[:12], 0.01)


def test_trio_phase_resolves_hets():
    """Mendelian phasing from grandparents recovers the parent's
    haplotypes exactly at sites where ≥1 grandparent is hom."""
    par = _synthetic_parents(M=5000, seed=3)
    rng = np.random.default_rng(11)
    af = (par.pat.sum(0) + par.mat.sum(0)) / 4.0
    M = len(par.pos)
    gp1 = par.pat[0].astype(int) + (rng.random(M) < af).astype(int)
    gp2 = par.pat[1].astype(int) + (rng.random(M) < af).astype(int)
    hap, resolved = E.trio_phase(par.pat.sum(0), gp1, gp2)
    het = par.pat[0] != par.pat[1]
    assert (resolved & het).sum() / max(het.sum(), 1) > 0.6
    ok = resolved & het
    assert (hap[0, ok] == par.pat[0, ok]).all()
    assert (hap[1, ok] == par.pat[1, ok]).all()


def test_biopsy_wga_artefacts(par):
    """ADO and coverage dispersion are per-site noise that the HMM
    smooths over; they should not collapse recovery the way phasing
    switches do."""
    het = (par.pat[0] != par.pat[1]) | (par.mat[0] != par.mat[1])
    obs, _, _ = E.apply_switch_errors(par, 0.01, np.random.default_rng(0))
    ne = 4
    truth = np.stack(
        [E.simulate_child(par, np.random.default_rng((1, e)))[0] for e in range(ne)]
    )
    results = {}
    for ado, cd in [(0.0, 0.0), (0.2, 0.7)]:
        biop = [
            E.simulate_biopsy(
                truth[e], 0.05, 0.01, np.random.default_rng((2, e)),
                ado=ado, cov_dispersion=cd,
            )
            for e in range(ne)
        ]
        _, d, *_ = E.joint_recover(obs, biop, 0.01, switch_rate=0.01)
        results[(ado, cd)] = (np.round(d[:, het]) == truth[:, het]).mean()
    clean, dirty = results[(0.0, 0.0)], results[(0.2, 0.7)]
    # On real chr22 the gap is <1 pp; the synthetic chrom is denser
    # so the effect is larger but still bounded.
    assert dirty > clean - 0.10, (
        f"WGA artefacts cost {clean - dirty:.3%} het-conc (>10pp)"
    )
    assert dirty > 0.70


def test_rare_variants_check_and_flag():
    """check_parents finds a variant on the father's hap-1 and
    flag_embryos marks exactly the embryos whose paternal path is 1
    at that site."""
    v = next(x for x in rv.PROTECTIVE if x.gene == "PCSK9" and x.pos_grch37)
    M = 5
    pos = np.array(
        [v.pos_grch37 - 200, v.pos_grch37 - 50, v.pos_grch37,
         v.pos_grch37 + 30, v.pos_grch37 + 400],
        dtype=np.int64,
    )
    ref = np.array(["A", "A", v.ref, "C", "G"], dtype="<U1")
    alt = np.array(["G", "C", v.alt, "T", "A"], dtype="<U1")
    pat = np.zeros((2, M), dtype=np.int8)
    pat[1, 2] = 1  # father carries the ALT on hap-1 at the variant site
    mat = np.zeros((2, M), dtype=np.int8)
    par = E.Parents(chrom=v.chrom, pos=pos, ref=ref, alt=alt, pat=pat, mat=mat)

    carried = rv.check_parents(par, rv.PROTECTIVE)
    assert len(carried) == 1
    cv, who, h = carried[0]
    assert (cv.gene, who, h) == ("PCSK9", "father", 1)

    # 3 embryos: e0 inherits paternal hap0 everywhere, e1 hap1, e2 mixed
    wp = np.array([[0] * M, [1] * M, [0, 0, 1, 0, 0]], dtype=np.int8)
    wm = np.zeros((3, M), dtype=np.int8)
    flags = rv.flag_embryos(carried, wp, wm, par)
    assert flags[0]["PCSK9"] is False
    assert flags[1]["PCSK9"] is True
    assert flags[2]["PCSK9"] is True

    s = rv.summarise(carried, flags)
    assert "PCSK9" in s and "father" in s

    # ACMG list sanity
    assert len(rv.PATHOGENIC_GENES) >= 75
    assert rv.PATHOGENIC_GENES["BRCA1"]["inheritance"] == "AD"
    assert all(v.direction == "protective" for v in rv.PROTECTIVE)


def test_score_with_uncertainty():
    par = _synthetic_parents(M=2000, seed=5)
    obs, _, _ = E.apply_switch_errors(par, 0.01, np.random.default_rng(0))
    truth, biop = _make_embryos(par, 3, cov=0.05)
    rng = np.random.default_rng(2)
    idx = rng.choice(len(par.pos), 200, replace=False)
    pgs = {
        "PGS_TEST": (
            np.sort(idx),
            rng.choice([-1, 1], 200).astype(np.int8),
            rng.normal(0, 0.01, 200),
        )
    }
    _, dose, var, *_ = E.joint_recover(obs, biop, 0.01, switch_rate=0.01)
    out = E.score_with_uncertainty(dose, var, pgs)
    score, se = out["PGS_TEST"]
    assert score.shape == (3,) and se.shape == (3,)
    assert (se >= 0).all()
    # point estimate matches score_chrom on the same dosage
    for e in range(3):
        ref = E.score_chrom(dose[e], pgs)["PGS_TEST"]
        assert abs(score[e] - ref) < 1e-9


def test_carrier_and_lpa_lists():
    assert len(rv.CARRIER_GENES) >= 80
    for g in ("CFTR", "SMN1", "HEXA", "GJB2", "PAH", "DMD"):
        assert g in rv.CARRIER_GENES, g
    assert rv.CARRIER_GENES["CFTR"]["inheritance"] == "AR"
    assert rv.CARRIER_GENES["DMD"]["inheritance"] == "XL"
    # Lp(a) tags
    assert len(rv.LPA_RISK) == 2
    assert all(v.gene == "LPA" and v.direction == "risk" for v in rv.LPA_RISK)
    assert {v.rsid for v in rv.LPA_RISK} == {"rs10455872", "rs3798220"}
    # APP A673T was added to PROTECTIVE
    assert any(v.gene == "APP" and v.rsid == "rs63750847" for v in rv.PROTECTIVE)


def test_pharmgx_call_diplotype():
    cyp2c19 = px.CPIC_BY_GENE["CYP2C19"]
    # *1/*2 → one no-function allele → intermediate metabolizer
    gt_het2 = {"rs4244285": ("G", "A"), "rs4986893": ("G", "G"),
               "rs12248560": ("C", "C")}
    diplo, pheno = px.call_diplotype(cyp2c19, gt_het2)
    assert diplo == "*1/*2"
    assert pheno == "Intermediate metabolizer"
    # *2/*2 → poor metabolizer
    gt_hom2 = {"rs4244285": ("A", "A"), "rs4986893": ("G", "G"),
               "rs12248560": ("C", "C")}
    diplo, pheno = px.call_diplotype(cyp2c19, gt_hom2)
    assert diplo == "*2/*2"
    assert pheno == "Poor metabolizer"
    # *1/*17 → rapid
    gt_17 = {"rs4244285": ("G", "G"), "rs4986893": ("G", "G"),
             "rs12248560": ("C", "T")}
    diplo, pheno = px.call_diplotype(cyp2c19, gt_17)
    assert diplo == "*1/*17"
    assert pheno == "Rapid metabolizer"
    # report runs without error and mentions the gene
    rep = px.report_pgx(gt_het2)
    assert "CYP2C19" in rep and "Intermediate" in rep
    assert len(px.CPIC_GENES) >= 10
