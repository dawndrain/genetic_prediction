"""Compute ancestry PCs from 1KG Phase 3 and save loadings for projection.

Uses every k-th HapMap3 SNP as a crude LD-thin (ancestry PCs are robust
to the exact SNP set; proper LD-pruning would need a tool we don't
have). Builds the 2,504 × ~55k standardized genotype matrix in memory,
runs truncated SVD, and saves:

  data/1kg_pca/loadings.tsv   rsid, chrom, pos, ref_allele, alt_allele,
                              alt_af, then PC1..PCk loading per SNP
  data/1kg_pca/sample_pcs.tsv sample, pop, super_pop, PC1..PCk
  data/1kg_pca/pgs_pc_coef.tsv  pgs_id, intercept, b_PC1..b_PCk
                              (regression of each raw PGS on PCs in 1KG)

To project a target genome: dosage of alt_allele at each loading SNP,
standardize as (d - 2·alt_af)/√(2·alt_af·(1-alt_af)), dot with each PC's
loading column. Then PGS_adj = raw_PGS − (intercept + Σ b_k·PC_k).
"""

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KG_DIR = Path("data/1kg")
LD_SNPINFO = Path("data/ld_reference/ldblk_1kg_eur/snpinfo_1kg_hm3")
OUT_DIR = Path("data/1kg_pca")
N_PC = 10
THIN = 20  # take every 20th HapMap3 SNP → ~56k


def load_panel():
    p = pd.read_csv(KG_DIR / "integrated_call_samples_v3.20130502.ALL.panel", sep="\t")
    return p["sample"].tolist(), p


def select_snps():
    info = pd.read_csv(LD_SNPINFO, sep="\t")
    info = info.iloc[::THIN].copy()
    info["CHR"] = info["CHR"].astype(str)
    by_chr: dict[str, dict[int, str]] = {}
    for _, r in info.iterrows():
        by_chr.setdefault(str(r.CHR), {})[int(r.BP)] = r.SNP
    return info, by_chr


def build_matrix(samples, by_chr):
    n = len(samples)
    rows: list[tuple[str, str, int, str, str, float]] = []
    cols: list[np.ndarray] = []
    for chrom in [str(c) for c in range(1, 23)]:
        vcf = next(KG_DIR.glob(f"ALL.chr{chrom}.*.vcf.gz"), None)
        want = by_chr.get(chrom, {})
        if vcf is None or not want:
            continue
        col_for: list[int] = []
        with gzip.open(vcf, "rt") as f:
            for line in f:
                if line.startswith("##"):
                    continue
                if line.startswith("#CHROM"):
                    hdr = line.rstrip().split("\t")[9:]
                    idx = {s: i for i, s in enumerate(hdr)}
                    col_for = [9 + idx[s] for s in samples]
                    continue
                t1 = line.find("\t")
                t2 = line.find("\t", t1 + 1)
                try:
                    pos = int(line[t1 + 1 : t2])
                except ValueError:
                    continue
                if pos not in want:
                    continue
                r = line.rstrip().split("\t")
                ref, alt = r[3], r[4]
                if len(ref) != 1 or len(alt) != 1:
                    continue
                d = np.fromiter(
                    ((g[0] == "1") + (g[-1] == "1") for c in col_for for g in (r[c],)),
                    dtype=np.int8,
                    count=n,
                )
                af = float(d.mean()) / 2
                if not 0.01 < af < 0.99:
                    continue
                rows.append((want[pos], chrom, pos, ref, alt, af))
                cols.append(d)
        print(f"  chr{chrom}: {len(cols)} SNPs cumulative", file=sys.stderr)
    G = np.vstack(cols).astype(np.float32)  # (M, N)
    meta = pd.DataFrame(
        rows, columns=["rsid", "chrom", "pos", "ref_allele", "alt_allele", "alt_af"]
    )
    return G, meta


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples, panel = load_panel()
    print(f"{len(samples)} samples", file=sys.stderr)

    info, by_chr = select_snps()
    print(f"{len(info):,} target SNPs (HapMap3 thinned 1/{THIN})", file=sys.stderr)

    G, meta = build_matrix(samples, by_chr)
    M, N = G.shape
    print(f"genotype matrix: {M:,} SNPs × {N} samples", file=sys.stderr)

    af = meta["alt_af"].to_numpy(dtype=np.float32)
    denom = np.sqrt(2 * af * (1 - af))
    Z = (G - 2 * af[:, None]) / denom[:, None]

    print("computing truncated SVD ...", file=sys.stderr)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    pcs = Vt[:N_PC].T * S[:N_PC]  # (N, k) sample scores
    loadings = U[:, :N_PC]  # (M, k) SNP loadings

    pc_cols = [f"PC{i + 1}" for i in range(N_PC)]
    samp = panel[["sample", "pop", "super_pop"]].copy()
    for i, c in enumerate(pc_cols):
        samp[c] = pcs[:, i]
    samp.to_csv(OUT_DIR / "sample_pcs.tsv", sep="\t", index=False)

    load_df = meta.copy()
    for i, c in enumerate(pc_cols):
        load_df[c] = loadings[:, i]
    load_df.to_csv(OUT_DIR / "loadings.tsv", sep="\t", index=False)

    var_exp = (S[:N_PC] ** 2) / (S**2).sum()
    print(
        "variance explained by PC1..PC{}: {}".format(
            N_PC, " ".join(f"{v:.3%}" for v in var_exp)
        ),
        file=sys.stderr,
    )

    scores = pd.read_csv("data/1kg_pgs_scores.tsv", sep="\t")
    wide = scores.pivot(index="sample", columns="pgs_id", values="score")
    wide = wide.reindex(samples)
    X = np.column_stack([np.ones(N), pcs])
    coef_rows = []
    for pid in wide.columns:
        y = wide[pid].to_numpy()
        if np.std(y) == 0:
            continue
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ b
        coef_rows.append(
            [pid] + b.tolist() + [float(resid.mean()), float(resid.std(ddof=1))]
        )
    coef = pd.DataFrame(
        coef_rows,
        columns=["pgs_id", "intercept"] + pc_cols + ["resid_mean", "resid_sd"],
    )
    coef.to_csv(OUT_DIR / "pgs_pc_coef.tsv", sep="\t", index=False)
    print(f"wrote loadings/sample_pcs/pgs_pc_coef to {OUT_DIR}/", file=sys.stderr)


if __name__ == "__main__":
    main()
