"""Batch pipeline: convert genomes -> merge VCFs -> impute (slim ref) -> PGS score.

Picks the best 100 genomes (with height phenotype data) and runs the full pipeline.
"""

import csv
import gzip
import shlex
import subprocess
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from snps import SNPs


DATA_DIR = Path("data")
GENOMES_DIR = Path("data/genomes_b37")
PGS_PATH = DATA_DIR / "PGS001229.txt"
PHENOTYPES_PATH = DATA_DIR / "phenotypes.tsv"
SLIM_REF = Path("reference_genomes_vcf_slim/all.1kg.phase3.v5a.vcf")
GENETIC_MAP = Path("resources/plink.all.GRCh37.map")
BEAGLE_JAR = Path("resources/beagle.jar")
MERGED_VCF_DIR = DATA_DIR / "vcf_merged"
SLIM_IMPUTED_DIR = DATA_DIR / "imputed_slim"

N_WORKERS = 6
N_GENOMES = 100


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


def select_genomes(phenotypes: dict, n: int, sex_filter: str | None = None) -> list[Path]:
    """Select genomes with phenotype data.

    Skips exome files and files >50MB. Optionally filters by sex.
    """
    candidates = []
    for gf in sorted(GENOMES_DIR.iterdir()):
        if not gf.is_file() or gf.name not in phenotypes:
            continue
        height, sex = phenotypes[gf.name]
        if sex_filter and sex != sex_filter:
            continue
        # Skip exome files (very large, slow to parse)
        if "exome" in gf.name.lower():
            continue
        # Skip very large files
        if gf.stat().st_size > 50_000_000:
            continue
        candidates.append(gf)

    if n and n < len(candidates):
        candidates = candidates[:n]
    return candidates


# -- Step 1: Convert + merge VCFs --

_pgs_weights_global = None

def _init_convert_worker(weights):
    global _pgs_weights_global
    _pgs_weights_global = weights


AUTOSOMES = set(str(c) for c in range(1, 23))

# Load reference alleles for proper VCF REF/ALT assignment
import json
with open("data/ref_alleles.json") as _f:
    _ref_raw = json.load(_f)
REF_ALLELES = {(k.split(":")[0], int(k.split(":")[1])): v for k, v in _ref_raw.items()}

VCF_HEADER = """##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
"""


def convert_and_merge_one(genome_path_str: str) -> dict | None:
    """Convert a DTC file to a merged VCF with proper REF/ALT from reference."""
    genome_path = Path(genome_path_str)
    sample_id = genome_path.name
    out_vcf = MERGED_VCF_DIR / f"{sample_id}.vcf"

    if out_vcf.exists():
        return {"genome_id": sample_id, "status": "exists"}

    warnings.filterwarnings("ignore")
    try:
        s = SNPs(genome_path_str)
        snps_df = s.snps
        if snps_df is None or snps_df.empty:
            return None

        # Filter to autosomes, deduplicate
        snps_df = snps_df[snps_df["chrom"].isin(AUTOSOMES)]
        snps_df = snps_df.drop_duplicates(subset=["chrom", "pos"])

        # Build (chrom, pos) -> genotype lookup
        geno = dict(zip(
            zip(snps_df["chrom"], snps_df["pos"]),
            snps_df["genotype"]
        ))

        # Write VCF with proper REF from reference
        MERGED_VCF_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_vcf, "w") as f:
            f.write(VCF_HEADER)
            for (chrom, pos), ref in sorted(REF_ALLELES.items(),
                                             key=lambda x: (int(x[0][0]) if x[0][0].isdigit() else 99, x[0][1])):
                gt = geno.get((chrom, pos))
                if not isinstance(gt, str) or gt == "--" or len(gt) < 2:
                    continue
                a1, a2 = gt[0], gt[1]
                if a1 not in "ACGT" or a2 not in "ACGT":
                    continue

                # Determine ALT and genotype based on reference allele
                alleles = {a1, a2}
                if alleles == {ref}:
                    # Homozygous reference
                    f.write(f"{chrom}\t{pos}\t.\t{ref}\t.\t.\t.\t.\tGT\t0/0\n")
                elif ref in alleles:
                    # Heterozygous
                    alt = (alleles - {ref}).pop()
                    f.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\tGT\t0/1\n")
                else:
                    # Homozygous alt (both alleles differ from ref)
                    if a1 == a2:
                        f.write(f"{chrom}\t{pos}\t.\t{ref}\t{a1}\t.\t.\t.\tGT\t1/1\n")
                    else:
                        # Both alleles differ from ref and from each other — skip
                        continue

        return {"genome_id": sample_id, "status": "converted"}

    except Exception:
        if out_vcf.exists():
            out_vcf.unlink()
        return None


# -- Step 2: Impute --

def impute_one(sample_id: str) -> dict | None:
    """Run Beagle imputation on a merged VCF with slim reference."""
    input_vcf = MERGED_VCF_DIR / f"{sample_id}.vcf"
    SLIM_IMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    out_prefix = SLIM_IMPUTED_DIR / sample_id
    out_file = Path(f"{out_prefix}.vcf.gz")

    if out_file.exists():
        return {"genome_id": sample_id, "status": "exists"}

    if not input_vcf.exists():
        return None

    t0 = time.time()
    cmd = (
        f"java -Xmx3g -jar {BEAGLE_JAR} "
        f"gt={input_vcf} "
        f"ref={SLIM_REF} "
        f"map={GENETIC_MAP} "
        f"out={out_prefix} "
        f"gp=true"
    )

    result = subprocess.run(
        shlex.split(cmd), capture_output=True, text=True, timeout=120
    )

    elapsed = time.time() - t0
    if result.returncode != 0 or not out_file.exists():
        return None

    return {"genome_id": sample_id, "status": "imputed", "time": elapsed}


# -- Step 3: Score --

def score_imputed(sample_id: str, weights: dict) -> dict | None:
    """Score an imputed VCF against PGS weights."""
    vcf_path = SLIM_IMPUTED_DIR / f"{sample_id}.vcf.gz"
    if not vcf_path.exists():
        return None

    score = 0.0
    matched = 0

    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            chrom = fields[0]
            pos = int(fields[1])
            ref = fields[3]
            alt = fields[4]

            key = (chrom, pos)
            if key not in weights:
                continue

            effect_allele, other_allele, weight = weights[key]

            # Get dosage
            fmt = fields[8].split(":")
            sample = fields[9].strip().split(":")

            if "DS" in fmt:
                ds_idx = fmt.index("DS")
                ds_val = sample[ds_idx]
                if "," in ds_val:
                    dosage = sum(float(x) for x in ds_val.split(","))
                else:
                    dosage = float(ds_val)
            elif "GP" in fmt:
                gp_idx = fmt.index("GP")
                gp = sample[gp_idx].split(",")
                dosage = float(gp[1]) + 2 * float(gp[2])
            else:
                continue

            # Align effect allele
            if effect_allele == alt:
                score += dosage * weight
            elif effect_allele == ref:
                score += (2 - dosage) * weight
            else:
                continue

            matched += 1

    return {"genome_id": sample_id, "pgs": score, "matched": matched}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sex", choices=["M", "F"], help="Filter by sex")
    parser.add_argument("-n", type=int, default=0, help="Max genomes (0=all)")
    args = parser.parse_args()

    print("Loading PGS weights...")
    weights = load_pgs_weights(PGS_PATH)
    print(f"  {len(weights)} variants")

    print("Loading phenotypes...")
    phenotypes = load_phenotypes(PHENOTYPES_PATH)
    print(f"  {len(phenotypes)} individuals with height")

    n = args.n if args.n > 0 else None
    sex_label = f" ({args.sex})" if args.sex else ""
    print(f"Selecting genomes{sex_label}...")
    genome_files = select_genomes(phenotypes, n, sex_filter=args.sex)
    print(f"  Selected {len(genome_files)} genomes")

    # Step 1: Convert (skip already converted)
    existing_vcfs = set(p.stem for p in MERGED_VCF_DIR.glob("*.vcf")) if MERGED_VCF_DIR.exists() else set()
    to_convert = [gf for gf in genome_files if gf.name not in existing_vcfs]
    print(f"\n=== Step 1: Convert to VCF ({N_WORKERS} workers) ===", flush=True)
    print(f"  {len(existing_vcfs & set(gf.name for gf in genome_files))} already converted, {len(to_convert)} to convert")
    MERGED_VCF_DIR.mkdir(parents=True, exist_ok=True)

    if to_convert:
        paths = [str(gf) for gf in to_convert]
        done = 0
        converted = 0
        with Pool(N_WORKERS, initializer=_init_convert_worker, initargs=(weights,)) as pool:
            for result in pool.imap_unordered(convert_and_merge_one, paths):
                done += 1
                if result:
                    converted += 1
                if done % 50 == 0 or done == len(to_convert):
                    print(f"  {done}/{len(to_convert)} processed ({converted} converted)", flush=True)

    # Collect all sample IDs (including previously converted)
    target_ids = set(gf.name for gf in genome_files)
    sample_ids = [f.stem for f in MERGED_VCF_DIR.iterdir()
                  if f.suffix == ".vcf" and f.stem in target_ids and f.stem in phenotypes]
    print(f"  {len(sample_ids)} VCFs ready")

    # Step 2: Impute
    print(f"\n=== Step 2: Impute with slim reference ({N_WORKERS} workers) ===", flush=True)
    SLIM_IMPUTED_DIR.mkdir(parents=True, exist_ok=True)

    done = 0
    imputed = 0
    with Pool(N_WORKERS) as pool:
        for result in pool.imap_unordered(impute_one, sample_ids):
            done += 1
            if result:
                imputed += 1
            if done % 20 == 0 or done == len(sample_ids):
                print(f"  {done}/{len(sample_ids)} processed ({imputed} imputed)", flush=True)

    print(f"  {imputed} genomes imputed")

    # Step 3: Score
    print(f"\n=== Step 3: Score ===", flush=True)
    results = []
    for sample_id in sample_ids:
        score_result = score_imputed(sample_id, weights)
        if score_result:
            height, sex = phenotypes[sample_id]
            score_result["height_cm"] = height
            score_result["sex"] = sex
            results.append(score_result)

    print(f"  {len(results)} scored")

    # Save
    out_path = DATA_DIR / "pgs_results_imputed.tsv"
    with open(out_path, "w") as f:
        f.write("genome_id\tpgs\tmatched\theight_cm\tsex\n")
        for r in sorted(results, key=lambda x: x["genome_id"]):
            f.write(f"{r['genome_id']}\t{r['pgs']:.6f}\t{r['matched']}\t{r['height_cm']:.1f}\t{r['sex']}\n")
    print(f"  Saved to {out_path}")

    # Correlation
    print(f"\n=== Results ===")
    pgs = np.array([r["pgs"] for r in results])
    height = np.array([r["height_cm"] for r in results])
    match_rates = np.array([r["matched"] / len(weights) for r in results])

    print(f"Match rate: mean={match_rates.mean():.1%}, min={match_rates.min():.1%}, max={match_rates.max():.1%}")

    corr = np.corrcoef(pgs, height)[0, 1]
    print(f"\nOverall: r={corr:.3f}, r²={corr**2:.3f}, n={len(results)}")

    for sex_label, sex_code in [("Male", "M"), ("Female", "F")]:
        sr = [r for r in results if r["sex"] == sex_code]
        if len(sr) > 10:
            p = np.array([r["pgs"] for r in sr])
            h = np.array([r["height_cm"] for r in sr])
            c = np.corrcoef(p, h)[0, 1]
            print(f"  {sex_label}: r={c:.3f}, r²={c**2:.3f}, n={len(sr)}")

    # Compare with raw results if available
    raw_path = DATA_DIR / "pgs_results.tsv"
    if raw_path.exists():
        raw_lookup = {}
        with open(raw_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                raw_lookup[row["genome_id"]] = float(row["pgs"])

        both = [r for r in results if r["genome_id"] in raw_lookup]
        if both:
            raw_pgs = np.array([raw_lookup[r["genome_id"]] for r in both])
            imp_pgs = np.array([r["pgs"] for r in both])
            raw_h = np.array([r["height_cm"] for r in both])
            r_raw = np.corrcoef(raw_pgs, raw_h)[0, 1]
            r_imp = np.corrcoef(imp_pgs, raw_h)[0, 1]
            print(f"\nComparison on {len(both)} overlapping samples:")
            print(f"  Raw (no imputation):  r={r_raw:.3f}, r²={r_raw**2:.3f}")
            print(f"  Imputed (slim ref):   r={r_imp:.3f}, r²={r_imp**2:.3f}")


if __name__ == "__main__":
    main()
