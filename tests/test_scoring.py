"""Scoring smoke tests that run without 1KG / network. The full
end-to-end (with PC adjustment against the shipped resources) is
exercised separately by `genepred score data/example_genome.txt`."""

import csv

from genepred.paths import RESOURCES
from genepred.scoring import ScoreResult


def test_resources_exist():
    for name in (
        "pgs_pc_coef.tsv",
        "1kg_pgs_summary.tsv",
        "loadings.tsv.gz",
        "sample_pcs.tsv.gz",
    ):
        assert (RESOURCES / name).exists(), f"missing shipped resource: {name}"


def test_pc_coef_columns():
    with open(RESOURCES / "pgs_pc_coef.tsv") as f:
        r = csv.DictReader(f, delimiter="\t")
        cols = set(r.fieldnames or [])
    required = {"pgs_id", "intercept", "resid_sd"} | {f"PC{i}" for i in range(1, 11)}
    assert required <= cols, f"pgs_pc_coef.tsv missing {required - cols}"


def test_summary_has_all_superpops():
    pops = set()
    with open(RESOURCES / "1kg_pgs_summary.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pops.add(row["super_pop"])
    assert {"EUR", "AFR", "AMR", "EAS", "SAS"} <= pops


def test_score_result_fields():
    # Smoke: ScoreResult has the fields the report needs
    for f in ("trait", "pgs_id", "z", "percentile", "n_matched", "n_total"):
        assert f in ScoreResult.__annotations__ or hasattr(
            ScoreResult, "_fields"
        ), f"ScoreResult missing {f}"


def test_missingness_warning():
    from genepred.scoring import missingness_warning

    def res(overlap, n_imputed=0, n_total=1000, z=0.0):
        n_matched = int(overlap * n_total)
        return ScoreResult(
            pgs_id="PGS_TEST", trait="trait", n_total=n_total,
            n_matched=n_matched, n_imputed=n_imputed, n_ambiguous=0,
            raw=0.0, z=z, percentile=50.0, method="pc-adjusted",
        )

    # imputed / near-complete genome: no warning
    assert missingness_warning([res(0.99), res(0.97)]) is None
    # raw DTC array (~50% overlap): warn and point at imputation
    w = missingness_warning([res(0.55), res(0.50), res(0.60)])
    assert w is not None and "impute" in w.lower()
    # an extreme z at low overlap is flagged as an artifact ...
    w = missingness_warning([res(0.99), res(0.12, z=9.9)])
    assert w is not None and "artifact" in w and "+9.9" in w
    # ... but the same z at high overlap is left alone
    assert missingness_warning([res(0.99), res(0.98, z=9.9)]) is None
    # mostly mean-imputed score is called out even when others are fine
    w = missingness_warning([res(0.99), res(0.12, n_imputed=850)])
    assert w is not None and "population mean" in w
    # nothing scored: no warning
    unscored = ScoreResult(
        pgs_id="PGS_TEST", trait="trait", n_total=10, n_matched=0,
        n_imputed=0, n_ambiguous=0, raw=0.0, z=None, percentile=None,
        method="low-overlap",
    )
    assert missingness_warning([unscored]) is None


def test_mean_imputed_sites_do_not_shift_z(tmp_path):
    """Mean-imputed (2·EAF) sites keep `raw` on the full-file scale but must
    not move the normalized z — the partial-overlap correction already
    accounts for the missing fraction."""
    from math import sqrt

    from genepred.scoring import _normalize, score_one

    genome_rs = {"rs1": ("A", "A"), "rs2": ("C", "T")}
    rows = [
        ("rs1", "A", "G", "1.0", "0.5"),
        ("rs2", "C", "T", "2.0", "0.2"),
        ("rs3", "A", "G", "1.5", "0.4"),  # not in genome
        ("rs4", "C", "T", "0.5", "0.3"),  # not in genome
    ]
    with_af = tmp_path / "with_af.txt"
    with_af.write_text(
        "rsID\teffect_allele\tother_allele\teffect_weight\tallelefrequency_effect\n"
        + "".join("\t".join(r) + "\n" for r in rows)
    )
    no_af = tmp_path / "no_af.txt"
    no_af.write_text(
        "rsID\teffect_allele\tother_allele\teffect_weight\n"
        + "".join("\t".join(r[:4]) + "\n" for r in rows)
    )

    r_af = score_one(genome_rs, {}, with_af)
    r_plain = score_one(genome_rs, {}, no_af)

    assert r_af["n_matched"] == r_plain["n_matched"] == 2
    assert r_af["n_imputed"] == 2 and r_plain["n_imputed"] == 0
    # matched-only sum: rs1 dosage 2 (w=1.0) + rs2 dosage 1 (w=2.0)
    assert r_af["raw_matched"] == r_plain["raw_matched"] == 4.0
    # full-scale raw additionally carries the imputed sites at 2·EAF
    assert abs(r_af["raw"] - (4.0 + 1.5 * 0.8 + 0.5 * 0.6)) < 1e-9
    # imputed sites are fixed at their expectation: zero variance contribution
    assert abs(r_af["var"] - (1.0 * 2 * 0.5 * 0.5 + 4.0 * 2 * 0.2 * 0.8)) < 1e-9

    mean, sd, f = 3.0, 1.2, 2 / 4
    ref = {"PGSTEST": (mean, sd, 4)}  # mean, sd, n_snps in the reference
    z_af, _, m_af = _normalize(
        "PGSTEST", r_af, pcs=None, pca_model=None, ref_pop_stats=ref
    )
    z_plain, _, m_plain = _normalize(
        "PGSTEST", r_plain, pcs=None, pca_model=None, ref_pop_stats=ref
    )
    assert m_af == m_plain == "ref-pop"
    # Without frequencies: random-missingness partial-overlap correction.
    assert abs(z_plain - (4.0 - f * mean) / (sd * sqrt(f))) < 1e-9
    # With frequencies: center on the matched subset's own expectation
    # (plus the matched share of the ancestry shift) and scale by the
    # matched share of the frequency-implied variance.
    center = r_af["exp_matched"] + f * (mean - r_af["expected"])
    scale = sd * sqrt(r_af["var"] / r_af["var_all"])
    assert abs(z_af - (4.0 - center) / scale) < 1e-9
    # Either way, the mean-imputed sites themselves never enter the z.
    assert r_af["raw_matched"] == r_plain["raw_matched"]


def test_af_sidecar_lookup_feeds_subset_correction(tmp_path):
    """A weight file with no frequency column gets per-SNP AFs from the
    sidecar lookup, enabling the matched-subset normalization."""
    from genepred.scoring import _normalize, score_one

    genome_rs = {"rs1": ("A", "A"), "rs2": ("C", "T")}
    rows = [
        ("rs1", "1", "100", "A", "G", "1.0"),
        ("rs2", "1", "200", "C", "T", "2.0"),
        ("rs3", "1", "300", "A", "G", "1.5"),  # not in genome
        ("rs4", "1", "400", "C", "T", "0.5"),  # not in genome
    ]
    wf = tmp_path / "no_af.txt"
    wf.write_text(
        "rsID\tchr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\n"
        + "".join("\t".join(r) + "\n" for r in rows)
    )
    # sidecar records: (ref, alt, alt_af); rs2/rs4 are stored ref=T alt=C so
    # the effect allele is the ALT for rs2/rs4 and the REF for none.
    af_by_pos = {
        ("1", 100): ("G", "A", 0.5),  # effect allele A == alt
        ("1", 200): ("T", "C", 0.2),
        ("1", 300): ("A", "G", 0.6),  # effect allele A == ref -> 1-0.6
        ("1", 400): ("T", "C", 0.3),
    }
    af_lookup = ({}, af_by_pos)

    r = score_one(genome_rs, {}, wf, af_lookup=af_lookup)
    assert r["n_af"] == 4 and r["n_imputed"] == 2
    assert r["raw_matched"] == 4.0
    # exp_matched = 1.0*2*0.5 + 2.0*2*0.2 ; expected adds rs3 (1-0.6) and rs4
    assert abs(r["exp_matched"] - (1.0 + 0.8)) < 1e-9
    assert abs(r["expected"] - (1.0 + 0.8 + 1.5 * 2 * 0.4 + 0.5 * 2 * 0.3)) < 1e-9
    assert r["var_all"] > r["var"] > 0

    mean, sd, f = 3.0, 1.2, 0.5
    z, _, method = _normalize(
        "PGSTEST", r, pcs=None, pca_model=None, ref_pop_stats={"PGSTEST": (mean, sd, 4)}
    )
    assert method == "ref-pop"
    center = r["exp_matched"] + f * (mean - r["expected"])
    scale = sd * (r["var"] / r["var_all"]) ** 0.5
    assert abs(z - (4.0 - center) / scale) < 1e-9
