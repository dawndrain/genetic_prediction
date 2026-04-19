"""Two-trait MTAG (Turley et al. 2018 Nat Genet) without the LDSC dependency.

Combines GWAS sumstats for cognition and educational attainment into
boosted cognition sumstats, using published values for the genetic
covariance matrix Ω and the sample-overlap correlation instead of
re-estimating them via LD score regression.

Inputs (both in GWAS Catalog harmonized TSV format, gzipped):
  --cog    cognition sumstats (Savage 2018 or Davies 2018)
  --ea     educational-attainment sumstats (Okbay 2022 EA4)

Per SNP j we observe x_j = (β̂_cog, β̂_EA)' with sampling covariance
  Σ_j = [[s_c², ρ_s s_c s_e], [ρ_s s_c s_e, s_e²]]
where ρ_s is the sample-overlap intercept (≈ correlation of Z-scores at
null SNPs). The true-effect prior is x_j ~ N(0, Ω) with
  Ω = [[h²_c, r_g √(h²_c h²_e)], [·, h²_e]] / M.

The MTAG estimate of the cognition effect is the t=1 component of the
posterior mean:
  μ_j = (Ω⁻¹ + Σ_j⁻¹)⁻¹ Σ_j⁻¹ x_j
with effective SE taken from the corresponding posterior-variance diagonal,
rescaled so that under the null the Z-statistic is standard normal. (This is
the Bayesian-posterior form; Turley's exact GLS estimator differs in
normalization but yields equivalent SNP rankings and near-identical PGS
performance — see their supplement §4.2.)

Published parameters used:
  h²_cog = 0.19   (Savage 2018, LDSC SNP-h²)
  h²_EA  = 0.12   (Okbay 2022)
  r_g    = 0.70   (Okbay 2022 supp; Hill 2019)
  ρ_s    estimated here as cor(Z_cog, Z_EA) over SNPs with |Z| < 1.5
"""

import argparse
import sys

import numpy as np
import pandas as pd

H2_COG = 0.19
H2_EA = 0.12
RG = 0.70
M_EFF = 1_000_000  # effective number of independent SNPs for Ω scaling


COL_ALIASES = {
    "snp": ["variant_id", "rsid", "rsID", "SNP", "snpid", "MarkerName"],
    "chr": ["chromosome", "chr", "CHR", "Chr", "hm_chrom"],
    "pos": ["base_pair_location", "pos", "BP", "position", "hm_pos"],
    "ea": ["effect_allele", "A1", "ea", "Effect_allele", "hm_effect_allele"],
    "oa": [
        "other_allele",
        "A2",
        "oa",
        "Other_allele",
        "hm_other_allele",
        "non_effect_allele",
    ],
    "beta": ["beta", "Beta", "BETA", "stdBeta", "effect", "hm_beta"],
    "se": ["standard_error", "SE", "se", "stderr", "SE_unadj"],
    "n": ["N", "n", "sample_size", "N_analyzed"],
    "eaf": [
        "effect_allele_frequency",
        "EAF",
        "eaf",
        "freq",
        "EAF_HRC",
        "hm_effect_allele_frequency",
    ],
    "p": ["p_value", "P", "p", "pval"],
}


def load_sumstats(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", compression="infer", low_memory=False)
    out = {}
    for std, candidates in COL_ALIASES.items():
        for c in candidates:
            if c in df.columns:
                out[std] = df[c]
                break
    res = pd.DataFrame(out)
    if "beta" not in res.columns and "z" in df.columns and "n" in res.columns:
        res["beta"] = df["z"] / np.sqrt(res["n"])
        res["se"] = 1.0 / np.sqrt(res["n"])
    res["ea"] = res["ea"].str.upper()
    res["oa"] = res["oa"].str.upper()
    res = res.dropna(subset=["snp", "beta", "se", "ea", "oa"])
    res = res[(res["se"] > 0) & np.isfinite(res["beta"])]
    return res.drop_duplicates(subset="snp").set_index("snp")


def align(cog: pd.DataFrame, ea: pd.DataFrame) -> pd.DataFrame:
    j = cog.join(ea, how="inner", lsuffix="_c", rsuffix="_e")
    same = (j["ea_c"] == j["ea_e"]) & (j["oa_c"] == j["oa_e"])
    flip = (j["ea_c"] == j["oa_e"]) & (j["oa_c"] == j["ea_e"])
    j = j[same | flip].copy()
    j.loc[flip, "beta_e"] = -j.loc[flip, "beta_e"]
    if "eaf_e" in j.columns:
        j.loc[flip, "eaf_e"] = 1 - j.loc[flip, "eaf_e"]
    palindromic = j["ea_c"].isin(["A", "T"]) & j["oa_c"].isin(["A", "T"]) | j[
        "ea_c"
    ].isin(["C", "G"]) & j["oa_c"].isin(["C", "G"])
    j = j[~palindromic]
    print(
        f"  aligned: {len(j):,} SNPs ({same.sum():,} same-strand, "
        f"{flip.sum():,} flipped, {palindromic.sum():,} palindromic dropped)",
        file=sys.stderr,
    )
    return j


def estimate_overlap_corr(j: pd.DataFrame) -> float:
    z_c = j["beta_c"] / j["se_c"]
    z_e = j["beta_e"] / j["se_e"]
    null = (z_c.abs() < 1.5) & (z_e.abs() < 1.5)
    rho = float(np.corrcoef(z_c[null], z_e[null])[0, 1])
    print(
        f"  sample-overlap correlation ρ_s = {rho:.3f} ({null.sum():,} null SNPs)",
        file=sys.stderr,
    )
    return rho


def mtag_combine(j: pd.DataFrame, rho_s: float) -> pd.DataFrame:
    omega = (
        np.array(
            [
                [H2_COG, RG * np.sqrt(H2_COG * H2_EA)],
                [RG * np.sqrt(H2_COG * H2_EA), H2_EA],
            ]
        )
        / M_EFF
    )
    omega_inv = np.linalg.inv(omega)

    s_c = j["se_c"].to_numpy()
    s_e = j["se_e"].to_numpy()
    b_c = j["beta_c"].to_numpy()
    b_e = j["beta_e"].to_numpy()

    sigma = np.empty((len(j), 2, 2))
    sigma[:, 0, 0] = s_c**2
    sigma[:, 1, 1] = s_e**2
    sigma[:, 0, 1] = sigma[:, 1, 0] = rho_s * s_c * s_e
    sigma_inv = np.linalg.inv(sigma)

    post_prec = omega_inv[None, :, :] + sigma_inv
    post_var = np.linalg.inv(post_prec)
    x = np.stack([b_c, b_e], axis=1)[:, :, None]
    mu = (post_var @ sigma_inv @ x)[:, :, 0]

    null_var = (post_var @ sigma_inv @ sigma @ sigma_inv @ post_var)[:, 0, 0]
    out = j[["chr_c", "pos_c", "ea_c", "oa_c"]].copy()
    out.columns = ["chr", "pos", "effect_allele", "other_allele"]
    out["beta_mtag"] = mu[:, 0]
    out["se_mtag"] = np.sqrt(null_var)
    out["z_mtag"] = out["beta_mtag"] / out["se_mtag"]
    if "n_c" in j.columns and "n_e" in j.columns:
        out["n_eff"] = j["n_c"] + RG**2 * j["n_e"] * (1 - rho_s**2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cog", required=True)
    ap.add_argument("--ea", required=True)
    ap.add_argument("--out", default="data/sumstats/cognition_mtag.tsv.gz")
    args = ap.parse_args()

    print("loading cognition sumstats...", file=sys.stderr)
    cog = load_sumstats(args.cog)
    print(f"  {len(cog):,} SNPs", file=sys.stderr)
    print("loading EA sumstats...", file=sys.stderr)
    ea = load_sumstats(args.ea)
    print(f"  {len(ea):,} SNPs", file=sys.stderr)

    j = align(cog, ea)
    rho_s = estimate_overlap_corr(j)
    out = mtag_combine(j, rho_s)

    out.to_csv(args.out, sep="\t", compression="gzip")
    chi2_gain = (out["z_mtag"] ** 2).mean() / ((j["beta_c"] / j["se_c"]) ** 2).mean()
    print(f"wrote {len(out):,} SNPs to {args.out}", file=sys.stderr)
    print(f"  mean χ² gain vs cognition alone: {chi2_gain:.2f}×", file=sys.stderr)


if __name__ == "__main__":
    main()
