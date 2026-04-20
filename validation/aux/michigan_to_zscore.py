"""Convert Michigan Imputation Server raw PGS output to approximate
z-scores against the local 1KG-EUR reference, then to risk / trait shift.

Caveat: Michigan scores at HRC coverage (~97% of each PGS's SNPs); the
local 1KG reference scored a different (overlapping) ~95–99% subset.
The partial-sum correction (subtract f·mean, divide by √f·sd) handles
the coverage difference, but any systematic difference in scoring
convention (e.g., missing-genotype handling) shows up as a constant
z-offset across samples. So treat absolute percentiles as approximate;
between-sample rankings are reliable.
"""

import argparse
import json
import sys
from math import sqrt
from pathlib import Path

import pandas as pd

from genepred import qaly as q
from genepred.catalog import CURATED
from genepred.qaly import liability_threshold_risk


def liability_to_risk(z, r2, prev):
    # Original signature was (z, r2, prevalence); the package's
    # liability_threshold_risk takes (z, prevalence, r2).
    return liability_threshold_risk(z, prev, r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/michigan_pgs/scores.txt")
    ap.add_argument("--info", default="data/michigan_pgs/scores.info")
    ap.add_argument("--ref", default="data/1kg_pgs_summary.tsv")
    ap.add_argument("--ref-pop", default="EUR")
    ap.add_argument("--out", default="data/michigan_pgs/scores_z.tsv")
    args = ap.parse_args()

    scores = pd.read_csv(args.scores, sep=None, engine="python")
    sample_col = scores.columns[0]
    scores = scores.set_index(sample_col)
    pgs_cols = [c for c in scores.columns if c.upper().startswith("PGS")]
    print(f"Michigan: {len(scores)} samples × {len(pgs_cols)} scores", file=sys.stderr)

    info = {d["name"]: d for d in json.loads(Path(args.info).read_text())}

    ref = pd.read_csv(args.ref, sep="\t")
    ref = ref[ref.super_pop == args.ref_pop].set_index("pgs_id")
    print(f"local 1KG-{args.ref_pop} reference: {len(ref)} scores", file=sys.stderr)

    id2t = {s.pgs_id: t for t, s in CURATED.items()}

    out = []
    for pid in pgs_cols:
        if pid not in ref.index:
            continue
        mu = float(ref.at[pid, "mean"])
        sd = float(ref.at[pid, "sd"])
        n_ref = int(ref.at[pid, "n_snps"])
        n_mich = info.get(pid, {}).get("variantsUsed", n_ref)
        f = n_mich / max(n_ref, 1)
        if sd <= 0 or not (0.3 <= f <= 3.0):
            continue
        raw = scores[pid].to_numpy(dtype=float)
        z = (raw - f * mu) / (sd * sqrt(f))
        # Michigan ↔ local convention offset: re-center so the cohort mean
        # z is 0 ONLY when the offset looks systematic (cohort mean |z|>1
        # with low spread). Keep both the raw-approx z and the recentered z.
        z_centered = z - z.mean()

        tk = id2t.get(pid)
        for samp, zi, zc in zip(scores.index, z, z_centered):
            row = {
                "sample": samp,
                "pgs_id": pid,
                "trait": tk or "",
                "raw": float(scores.at[samp, pid]),
                "n_snps_michigan": n_mich,
                "n_snps_ref": n_ref,
                "f": round(f, 3),
                "z_approx": float(zi),
                "z_cohort_centered": float(zc),
            }
            if tk in q.DISEASE_TRAITS:
                dt = q.DISEASE_TRAITS[tk]
                row["risk"] = liability_to_risk(zc, dt.pgs_r2_population, dt.prevalence)
                row["baseline"] = dt.prevalence
                row["rr"] = row["risk"] / dt.prevalence
            elif tk in q.CONTINUOUS_TRAITS:
                ct = q.CONTINUOUS_TRAITS[tk]
                row["trait_shift_sd"] = float(zc * sqrt(ct.pgs_r2_population))
            out.append(row)

    df = pd.DataFrame(out)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {len(df)} rows to {args.out}", file=sys.stderr)

    if not df.empty:
        print(
            "\nz_approx summary by score (mean should be ~0 for a "
            "random EUR cohort; non-zero mean → convention offset):"
        )
        s = (
            df.groupby(["pgs_id", "trait"])["z_approx"]
            .agg(["mean", "std", "min", "max"])
            .round(2)
        )
        print(s.to_string())


if __name__ == "__main__":
    main()
