"""No-retrain pseudo-validation of cognition PGS variants.

For trained per-allele weights β and an independent target GWAS with
z-scores z and sample size N,

    cor(PGS, y_target) ≈ Σ_j β_j · sd(x_j) · z_j  / (√N · sd(PGS))

where sd(x_j) = √(2 p_j (1−p_j)) from the LD-reference allele
frequencies and sd(PGS) is estimated empirically from the openSNP
genomes (scored by validation/validate_height_archive.py and saved to
data/opensnp_archive_pgs.tsv). The empirical sd(PGS) stands in for
√(βᵀRβ) — the LD-naive ||β||² approximation gives R² > 1 because dense
(LDpred-inf) and sparse (SBayesRC) weight vectors have very different
LD-inflation factors.

Cross-evaluation design:
    v1, v2  (trained on Savage×EA4 MTAG) → target = de la Fuente g
    v3      (trained on de la Fuente g)  → target = Savage×EA4 MTAG

CAVEAT: Savage 2018 includes UKB and de la Fuente is pure UKB, so the
v1/v2→g pair has sample overlap. Absolute pseudo-r is therefore
inflated and is NOT a phenotypic R². The v1-vs-v2 *ratio* is still
informative since both share the same overlap.
"""

import gzip
import math
import os
import sys
from pathlib import Path

import pandas as pd

REF = Path(__file__).parent
DATA = REF / "data"
SNPINFO = Path(os.environ.get("SBAYESRC_LDM", "tools/sbayesrc/ukbEUR_HM3/ukbEUR_HM3")) / "snp.info"

WEIGHTS = {
    "v1_ldpredinf": (
        DATA / "pgs_scoring_files" / "COGNITION_mtag_ldpredinf_hmPOS_GRCh38.txt.gz",
        "raw_ea",
    ),
    "v2_sbayesrc": (
        DATA / "pgs_scoring_files" / "COGNITION_mtag_sbayesrc_hmPOS_GRCh38.txt.gz",
        "raw_ea_v2",
    ),
    "v3_geneticg": (
        DATA / "pgs_scoring_files" / "COGNITION_geneticg_sbayesrc_hmPOS_GRCh38.txt.gz",
        "raw_ea_v3",
    ),
}

TARGETS = {
    "delaFuente_g": dict(
        path=DATA / "sumstats" / "delaFuente2021_genetic_g.tsv.gz",
        cols=("SNP", "A1", "A2", "Z_Estimate"),
        n=282_014,
    ),
    "mtag_savage_ea4": dict(
        path=DATA / "sumstats" / "cognition_mtag.tsv.gz",
        cols=("snp", "effect_allele", "other_allele", "z_mtag"),
        n=402_000,
    ),
}

EVAL_PAIRS = [
    ("v1_ldpredinf", "delaFuente_g"),
    ("v2_sbayesrc", "delaFuente_g"),
    ("v3_geneticg", "mtag_savage_ea4"),
]


def _open(p):
    p = str(p)
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def load_freq() -> dict[str, tuple[str, str, float]]:
    out: dict[str, tuple[str, str, float]] = {}
    with open(SNPINFO) as f:
        next(f)
        for ln in f:
            r = ln.split()
            out[r[1]] = (r[5].upper(), r[6].upper(), float(r[7]))
    return out


def load_weights(path) -> tuple[dict, dict, dict]:
    a1: dict[str, str] = {}
    a2: dict[str, str] = {}
    w: dict[str, float] = {}
    ix: dict[str, int] = {}
    with _open(path) as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            r = ln.rstrip().split("\t")
            if r[0] == "rsID":
                ix = {c: i for i, c in enumerate(r)}
                continue
            sid = r[ix["rsID"]]
            a1[sid] = r[ix["effect_allele"]].upper()
            a2[sid] = r[ix["other_allele"]].upper()
            w[sid] = float(r[ix["effect_weight"]])
    return a1, a2, w


def load_target(spec) -> tuple[dict, dict, dict]:
    c_snp, c_a1, c_a2, c_z = spec["cols"]
    a1: dict[str, str] = {}
    a2: dict[str, str] = {}
    z: dict[str, float] = {}
    with _open(spec["path"]) as f:
        hdr = next(f).rstrip().split()
        ix = {c: i for i, c in enumerate(hdr)}
        for ln in f:
            r = ln.rstrip().split()
            try:
                zv = float(r[ix[c_z]])
            except (ValueError, IndexError):
                continue
            sid = r[ix[c_snp]]
            a1[sid] = r[ix[c_a1]].upper()
            a2[sid] = r[ix[c_a2]].upper()
            z[sid] = zv
    return a1, a2, z


def pseudo_r(wpath, tgt, n_target, sd_emp, freq) -> tuple[float, int]:
    wa1, wa2, w = load_weights(wpath)
    ta1, ta2, tz = tgt
    num = 0.0
    m = 0
    for sid, b in w.items():
        if sid not in tz or sid not in freq:
            continue
        ba1, ba2 = wa1[sid], wa2[sid]
        if (ba1, ba2) == (ta1[sid], ta2[sid]):
            zs = 1.0
        elif (ba1, ba2) == (ta2[sid], ta1[sid]):
            zs = -1.0
        else:
            continue
        fa1, fa2, p = freq[sid]
        if (ba1, ba2) == (fa1, fa2):
            pass
        elif (ba1, ba2) == (fa2, fa1):
            p = 1.0 - p
        else:
            continue
        num += b * math.sqrt(2 * p * (1 - p)) * zs * tz[sid]
        m += 1
    return num / (math.sqrt(n_target) * sd_emp), m


def main():
    freq = load_freq()
    print(f"loaded {len(freq):,} SNP frequencies", file=sys.stderr)
    emp = pd.read_csv(DATA / ".." / "data" / "opensnp_archive_pgs.tsv", sep="\t")
    tgt_cache: dict[str, tuple] = {}
    print(
        f"\n{'score':<14} {'target':<18} {'m_snps':>10} "
        f"{'sd(PGS)':>11} {'pseudo-r':>9} {'pseudo-R²':>10}"
    )
    for sname, tname in EVAL_PAIRS:
        wpath, emp_col = WEIGHTS[sname]
        if not wpath.exists():
            print(f"{sname:<14} {tname:<18}   (weights not built yet)")
            continue
        if emp_col not in emp.columns:
            print(f"{sname:<14} {tname:<18}   (no empirical sd — rerun validation)")
            continue
        if tname not in tgt_cache:
            print(f"  loading target {tname} ...", file=sys.stderr)
            tgt_cache[tname] = load_target(TARGETS[tname])
        sd_emp = float(emp[emp_col].std())
        r, m = pseudo_r(wpath, tgt_cache[tname], TARGETS[tname]["n"], sd_emp, freq)
        print(
            f"{sname:<14} {tname:<18} {m:>10,} {sd_emp:>11.4e} "
            f"{r:>+9.4f} {r * r:>10.4f}"
        )


if __name__ == "__main__":
    main()
