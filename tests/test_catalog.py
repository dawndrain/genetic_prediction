"""Catalog ↔ resources ↔ qaly consistency. These guard against the
brittleness that bites when CURATED is updated but the shipped 1KG
reference / qaly trait map isn't."""

import csv

from genepred import catalog, qaly
from genepred.paths import RESOURCES


def _resource_keys(filename: str, key_col: str) -> set[str]:
    with open(RESOURCES / filename) as f:
        return {row[key_col] for row in csv.DictReader(f, delimiter="\t")}


def _strip_suffix(trait: str) -> str:
    for suf in ("_sbrc", "_mtag"):
        if trait.endswith(suf):
            return trait[: -len(suf)]
    return trait


def test_curated_have_required_fields():
    for k, s in catalog.CURATED.items():
        assert s.pgs_id, k
        assert s.trait_reported, k
        assert s.n_variants > 0, k
        assert 0 <= s.r2_eur_pop < 1, k


def test_release_url_pattern():
    assert "{pgs}" in catalog.RELEASE
    assert "github.com" in catalog.RELEASE
    assert catalog.RELEASE.format(pgs="X").endswith("X_hmPOS_GRCh38.txt.gz")


def test_resources_cover_curated():
    """Every CURATED score must have a 1KG reference row, otherwise
    it can't get a z-score and silently lands in 'Not interpreted'.
    If this fails: rerun reference/onekg/build_1kg_reference.py and
    copy the outputs into genepred/resources/."""
    pc_keys = _resource_keys("pgs_pc_coef.tsv", "pgs_id")
    sum_keys = _resource_keys("1kg_pgs_summary.tsv", "pgs_id")
    missing_pc = []
    missing_sum = []
    for k, s in catalog.CURATED.items():
        # the reference build keys by the file's stem-prefix (first
        # token before '_'), so e.g. STROKE_gigastroke_sbayesrc → STROKE
        candidates = {s.pgs_id, s.pgs_id.split("_")[0], k}
        if not (candidates & pc_keys):
            missing_pc.append(f"{k} ({s.pgs_id})")
        if not (candidates & sum_keys):
            missing_sum.append(f"{k} ({s.pgs_id})")
    assert not missing_pc, (
        f"{len(missing_pc)} CURATED entries with no PC-coef row "
        f"(rerun build_1kg_reference.py): {missing_pc}"
    )
    assert not missing_sum, (
        f"{len(missing_sum)} CURATED entries with no 1KG-summary row: "
        f"{missing_sum}"
    )


def test_curated_traits_map_to_qaly():
    """Every CURATED trait (after method-suffix strip) must be a key
    in the QALY model, otherwise the report can't interpret it."""
    qaly_keys = set(qaly.DISEASE_TRAITS) | set(qaly.CONTINUOUS_TRAITS)
    unmapped = [
        k for k in catalog.CURATED if _strip_suffix(k) not in qaly_keys
    ]
    assert not unmapped, (
        f"CURATED traits with no qaly entry: {unmapped}. "
        f"Add to genepred.qaly or rename the catalog key."
    )


def test_no_duplicate_pgs_ids():
    seen: dict[str, str] = {}
    dupes = []
    for k, s in catalog.CURATED.items():
        if s.pgs_id in seen:
            dupes.append(f"{s.pgs_id}: {seen[s.pgs_id]} & {k}")
        seen[s.pgs_id] = k
    assert not dupes, f"Duplicate pgs_id in CURATED: {dupes}"
