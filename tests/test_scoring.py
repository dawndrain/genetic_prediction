"""Scoring smoke tests that run without 1KG / network. The full
end-to-end (with PC adjustment against the shipped resources) is
exercised separately by `genepred report data/example_genome.txt`."""

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
