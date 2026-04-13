"""Compute PGS directly from raw DTC genotype files (no imputation needed).

Parallelized with module-level global to avoid pickling the weights dict.
"""

import csv
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from snps import SNPs


DATA_DIR = Path("data")
GENOMES_DIR = Path("/Users/dawndrain/Code/genetics_regression/genomes")
PGS_PATH = DATA_DIR / "PGS001229.txt"
PHENOTYPES_PATH = DATA_DIR / "phenotypes.tsv"
N_WORKERS = 8

# Module-level global — set once by each worker via initializer
_weights = None


def _init_worker(weights):
    global _weights
    _weights = weights


def load_pgs_weights(pgs_path: Path) -> dict:
    weights = {}
    col_idx = None
    with open(pgs_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if col_idx is None:
                col_idx = {name: i for i, name in enumerate(fields)}
                continue
            chrom = fields[col_idx["chr_name"]]
            pos = fields[col_idx["chr_position"]]
            effect_allele = fields[col_idx["effect_allele"]]
            other_allele = fields[col_idx["other_allele"]]
            weight = float(fields[col_idx["effect_weight"]])
            if chrom in ("XY", "MT", "Y") or not pos:
                continue
            weights[(chrom, int(pos))] = (effect_allele, other_allele, weight)
    return weights


def load_phenotypes(path: Path) -> dict:
    pheno = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pheno[row["genome_id"]] = (float(row["height_cm"]), row.get("sex", ""))
    return pheno


def score_one(genome_path: str) -> dict | None:
    """Score a single genome. Uses module-level _weights global."""
    warnings.filterwarnings("ignore")
    gpath = Path(genome_path)

    try:
        s = SNPs(genome_path)
        if s.build and s.build != 37:
            return None  # skip non-build-37 for speed

        snps_df = s.snps
        if snps_df is None or snps_df.empty:
            return None

        snps_df = snps_df.reset_index()
        geno_lookup = dict(zip(
            zip(snps_df["chrom"], snps_df["pos"]),
            snps_df["genotype"]
        ))

        score = 0.0
        matched = 0

        for (chrom, pos), (effect_allele, other_allele, weight) in _weights.items():
            genotype = geno_lookup.get((chrom, pos))
            if not isinstance(genotype, str) or genotype == "--" or len(genotype) < 2:
                continue
            dosage = (1 if genotype[0] == effect_allele else 0) + \
                     (1 if genotype[1] == effect_allele else 0)
            score += dosage * weight
            matched += 1

        return {"genome_id": gpath.name, "pgs": score, "matched": matched}

    except Exception as e:
        return None


def main():
    print("Loading PGS weights...")
    weights = load_pgs_weights(PGS_PATH)
    print(f"  {len(weights)} variants")

    print("Loading phenotypes...")
    phenotypes = load_phenotypes(PHENOTYPES_PATH)
    print(f"  {len(phenotypes)} individuals with height")

    genome_files = sorted(GENOMES_DIR.iterdir())
    genome_files = [f for f in genome_files if f.name in phenotypes and f.is_file()]
    print(f"  {len(genome_files)} genomes with phenotype data")
    print(f"  Scoring with {N_WORKERS} workers...", flush=True)

    paths = [str(gf) for gf in genome_files]

    results = []
    done = 0
    with Pool(N_WORKERS, initializer=_init_worker, initargs=(weights,)) as pool:
        for result in pool.imap_unordered(score_one, paths):
            done += 1
            if result:
                gid = result["genome_id"]
                height_cm, sex = phenotypes[gid]
                result["height_cm"] = height_cm
                result["sex"] = sex
                results.append(result)
            if done % 100 == 0 or done == len(genome_files):
                print(f"  {done}/{len(genome_files)} done ({len(results)} scored)", flush=True)

    # Save results
    results.sort(key=lambda r: r["genome_id"])
    out_path = DATA_DIR / "pgs_results.tsv"
    with open(out_path, "w") as f:
        f.write("genome_id\tpgs\tmatched\theight_cm\tsex\n")
        for r in results:
            f.write(f"{r['genome_id']}\t{r['pgs']:.6f}\t{r['matched']}\t{r['height_cm']:.1f}\t{r['sex']}\n")
    print(f"\nSaved {len(results)} results to {out_path}")

    # Filter to high match rate
    good = [r for r in results if r["matched"] / len(weights) > 0.5]
    print(f"High match rate (>50%): {len(good)} / {len(results)}")

    if len(good) > 10:
        pgs_vals = np.array([r["pgs"] for r in good])
        height_vals = np.array([r["height_cm"] for r in good])
        corr = np.corrcoef(pgs_vals, height_vals)[0, 1]
        print(f"\nPGS-height correlation (match>50%): r = {corr:.3f} (r² = {corr**2:.3f}, n={len(good)})")

        for sex_label, sex_code in [("Male", "M"), ("Female", "F")]:
            sex_results = [r for r in good if r["sex"] == sex_code]
            if len(sex_results) > 10:
                p = np.array([r["pgs"] for r in sex_results])
                h = np.array([r["height_cm"] for r in sex_results])
                c = np.corrcoef(p, h)[0, 1]
                print(f"  {sex_label}: r = {c:.3f} (r² = {c**2:.3f}, n={len(sex_results)})")

    # Match rate stats
    match_rates = [r["matched"] / len(weights) for r in results]
    print(f"\nMatch rates: mean={np.mean(match_rates):.1%}, min={np.min(match_rates):.1%}, max={np.max(match_rates):.1%}")


if __name__ == "__main__":
    main()
