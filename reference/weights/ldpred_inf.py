"""Closed-form infinitesimal-model shrinkage (LDpred-inf) using the PRS-CS
1KG EUR LD-block reference. For each block b:

    β_post,b = (R_b + λ I)⁻¹ β̂_b,    λ = M / (N · h²)

where β̂ are per-SNP standardized marginal effects (z/√N), R_b is the
within-block LD correlation matrix, M is the total SNP count, N the GWAS
effective sample size, and h² the SNP heritability. This is the p→1 limit
of LDpred2 / the φ→∞ limit of PRS-CS; for highly polygenic traits the gap
to the full sparse-prior versions is small.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def load_sumstats(path, snpinfo):
    """Inner-join sumstats to the LD-reference SNP list, aligning alleles."""
    ss = pd.read_csv(path, sep="\t")
    ss = ss.rename(
        columns={"SNP": "snp", "A1": "a1", "A2": "a2", "BETA": "beta", "SE": "se"}
    )
    ss["a1"] = ss["a1"].str.upper()
    ss["a2"] = ss["a2"].str.upper()
    ss = ss.dropna(subset=["snp", "beta", "se"]).drop_duplicates("snp")
    ss = ss.set_index("snp")

    ref = snpinfo.set_index("SNP")
    j = ref.join(ss, how="inner")
    same = (j["A1"] == j["a1"]) & (j["A2"] == j["a2"])
    flip = (j["A1"] == j["a2"]) & (j["A2"] == j["a1"])
    j = j[same | flip].copy()
    j.loc[flip, "beta"] = -j.loc[flip, "beta"]
    j["z"] = j["beta"] / j["se"]
    print(
        f"  {len(j):,} SNPs in LD ref ∩ sumstats "
        f"({same.sum():,} same, {flip.sum():,} flipped)",
        file=sys.stderr,
    )
    return j


def run(ld_dir: Path, sumstats: pd.DataFrame, n_gwas: int, h2: float):
    """Returns dataframe with posterior effect sizes on the standardized scale."""
    M = len(sumstats)
    lam = M / (n_gwas * h2)
    print(f"  M={M:,}  N={n_gwas:,}  h²={h2}  λ={lam:.3f}", file=sys.stderr)

    ss_by_snp = sumstats["z"].to_dict()
    sqrt_n = np.sqrt(n_gwas)

    rows = []
    for chrom in range(1, 23):
        h5_path = ld_dir / f"ldblk_1kg_chr{chrom}.hdf5"
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as f:
            n_blk = len(f)
            for k in range(1, n_blk + 1):
                grp = f[f"blk_{k}"]
                snplist = [
                    s.decode() if isinstance(s, bytes) else s for s in grp["snplist"][:]
                ]
                idx = [i for i, s in enumerate(snplist) if s in ss_by_snp]
                if not idx:
                    continue
                R = np.asarray(grp["ldblk"])[np.ix_(idx, idx)]
                z = np.array([ss_by_snp[snplist[i]] for i in idx])
                beta_hat = z / sqrt_n
                beta_post = np.linalg.solve(R + lam * np.eye(len(idx)), beta_hat)
                for i, b in zip(idx, beta_post):
                    rows.append((snplist[i], chrom, b))
        print(f"  chr{chrom}: {n_blk} blocks processed", file=sys.stderr)

    out = pd.DataFrame(rows, columns=["snp", "chrom", "beta_post"])
    return out.set_index("snp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sst", default="data/sumstats/cognition_mtag_prscs.txt")
    ap.add_argument("--ld-dir", default="data/ld_reference/ldblk_1kg_eur")
    ap.add_argument("--n-gwas", type=int, default=402000)
    ap.add_argument("--h2", type=float, default=0.19)
    ap.add_argument(
        "--out",
        default="data/pgs_scoring_files/COGNITION_mtag_ldpredinf_hmPOS_GRCh38.txt.gz",
    )
    args = ap.parse_args()

    ld_dir = Path(args.ld_dir)
    snpinfo = pd.read_csv(ld_dir / "snpinfo_1kg_hm3", sep="\t")
    print(f"  LD reference: {len(snpinfo):,} HapMap3 SNPs", file=sys.stderr)

    ss = load_sumstats(args.sst, snpinfo)
    post = run(ld_dir, ss, args.n_gwas, args.h2)

    final = (
        snpinfo.set_index("SNP")
        .join(post, how="inner")
        .rename_axis("rsID")
        .reset_index()
    )
    out_df = pd.DataFrame(
        {
            "rsID": final["rsID"],
            "chr_name": final["CHR"],
            "chr_position": final["BP"],
            "effect_allele": final["A1"],
            "other_allele": final["A2"],
            "effect_weight": final["beta_post"],
        }
    )
    out_df.to_csv(args.out, sep="\t", index=False, compression="gzip")
    print(f"wrote {len(out_df):,} weights to {args.out}", file=sys.stderr)

    var_ratio = (out_df["effect_weight"] ** 2).sum() / (
        ss["z"] ** 2 / args.n_gwas
    ).sum()
    print(
        f"  posterior/marginal effect-variance ratio: {var_ratio:.3f} "
        f"(shrinkage strength)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
