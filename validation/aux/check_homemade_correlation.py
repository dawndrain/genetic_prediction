"""Cross-check homemade SBayesRC scores against the Catalog score for
the same trait, by correlating the per-sample 1KG-EUR scores. If the
column-mapping in build_open_sbayesrc.sh is right, both should pick
up the same genetic signal → cor in the +0.4–0.9 range. cor < 0.2
means something's wrong (sign flip, wrong column, or one of the two
references is too sparse to be informative — check the n_snps row in
1kg_pgs_summary.tsv before blaming the SBayesRC build)."""

import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parents[2] / "data"
PAIRS = [
    # (homemade key in 1kg_pgs_scores, Catalog pgs_id, label)
    ("STROKE", "PGS002724", "stroke"),
    ("BMD", "PGS000657", "osteoporosis (eBMD/SOS)"),
    ("AF", "PGS005168", "atrial fibrillation"),
    ("ASTHMA", "PGS001787", "asthma"),
    ("COGNITION", "PGS004427", "cognition"),
    ("EA4", "PGS004427", "EA4 vs cognition (cross-trait, expect ~0.3)"),
]


def main():
    s = pd.read_csv(DATA / "1kg_pgs_scores.tsv", sep="\t")
    summ = pd.read_csv(DATA / "1kg_pgs_summary.tsv", sep="\t")
    summ = summ[summ.super_pop == "EUR"].set_index("pgs_id")["n_snps"]
    eur = s[s.super_pop == "EUR"].pivot(
        index="sample", columns="pgs_id", values="score"
    )
    print(f"{'pair':<42} {'cor':>7} {'n_snps (hm/cat)':>20}  verdict")
    print("-" * 90)
    bad = []
    for hm, cat, label in PAIRS:
        if hm not in eur or cat not in eur:
            print(f"{label:<42}    (one or both missing from 1kg_pgs_scores.tsv)")
            continue
        r = eur[hm].corr(eur[cat])
        cov = f"{summ.get(hm, '?'):>9,} / {summ.get(cat, '?'):>7,}"
        verdict = "ok" if r > 0.4 else ("low" if r > 0.15 else "BAD")
        if verdict == "BAD" and "cross-trait" not in label:
            bad.append(label)
        print(f"{label:<42} {r:>+7.3f} {cov:>20}  {verdict}")
    if bad:
        print(
            f"\n{len(bad)} pair(s) with cor < 0.15. Before assuming a "
            f"sign flip, check whether the Catalog score's n_snps is "
            f"low — if its 1KG match is sparse, its scores are noise.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
