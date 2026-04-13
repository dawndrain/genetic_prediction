"""Compute PGS (Polygenic Score) from imputed VCF data.

Usage:
  python predict.py [--pgs PATH] [--sample SAMPLE]

Loads PGS weights, looks up each variant's dosage in the imputed VCFs,
and computes the weighted sum.
"""

import argparse
import gzip
from collections import defaultdict
from pathlib import Path


DATA_DIR = Path("data")
IMPUTED_DIR = DATA_DIR / "imputed"
VCF_DIR = DATA_DIR / "vcf"
DEFAULT_PGS = DATA_DIR / "PGS001229.txt"


def load_pgs_weights(pgs_path: Path) -> dict:
    """Load PGS scoring file. Returns {(chrom, pos): (effect_allele, other_allele, weight, rsid)}."""
    weights = {}
    col_idx = None
    with open(pgs_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if col_idx is None:
                # Parse header
                col_idx = {name: i for i, name in enumerate(fields)}
                continue

            rsid = fields[col_idx["rsID"]]
            chrom = fields[col_idx["chr_name"]]
            pos = fields[col_idx["chr_position"]]
            effect_allele = fields[col_idx["effect_allele"]]
            other_allele = fields[col_idx["other_allele"]]
            weight = float(fields[col_idx["effect_weight"]])

            # Skip non-autosomal/X for now (XY PAR, MT, Y)
            if chrom in ("XY", "MT", "Y"):
                continue

            weights[(chrom, pos)] = (effect_allele, other_allele, weight, rsid)
    return weights


def load_imputed_variants(sample_id: str, chrom: str) -> dict:
    """Load imputed variants for a chromosome. Returns {pos: (ref, alt, dosage)}."""
    chrom_file = IMPUTED_DIR / sample_id / f"{chrom}.vcf.gz"
    if not chrom_file.exists():
        return {}

    variants = {}
    with gzip.open(chrom_file, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            pos = fields[1]
            ref = fields[3]
            alt = fields[4]

            # Get dosage — try DS first, then compute from GP
            fmt = fields[8].split(":")
            sample = fields[9].strip().split(":")

            if "DS" in fmt:
                ds_idx = fmt.index("DS")
                ds_val = sample[ds_idx]
                # Beagle may output per-haplotype dosages "d1,d2" — sum them
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

            variants[pos] = (ref, alt, dosage)
    return variants


def compute_pgs(sample_id: str, weights: dict) -> dict:
    """Compute PGS for a sample. Returns detailed results."""
    # Group weights by chromosome
    by_chrom = defaultdict(list)
    for (chrom, pos), v in weights.items():
        by_chrom[chrom].append((pos, v))

    total_score = 0.0
    matched = 0
    missing = 0
    flipped = 0
    details_by_chrom = {}

    for chrom in sorted(by_chrom.keys(), key=lambda c: (0, int(c)) if c.isdigit() else (1, 0)):
        variants = load_imputed_variants(sample_id, chrom)
        chrom_score = 0.0
        chrom_matched = 0
        chrom_missing = 0
        chrom_flipped = 0

        for pos, (effect_allele, other_allele, weight, rsid) in by_chrom[chrom]:
            if pos not in variants:
                chrom_missing += 1
                continue

            ref, alt, dosage = variants[pos]

            # dosage is count of ALT allele
            # If effect_allele == ALT, use dosage directly
            # If effect_allele == REF, use (2 - dosage)
            if effect_allele == alt:
                score_contribution = dosage * weight
            elif effect_allele == ref:
                score_contribution = (2 - dosage) * weight
                chrom_flipped += 1
            elif other_allele == alt or other_allele == ref:
                # Strand flip or allele coding mismatch — try to align
                if other_allele == alt:
                    score_contribution = (2 - dosage) * weight
                    chrom_flipped += 1
                else:
                    score_contribution = dosage * weight
            else:
                # Can't align alleles
                chrom_missing += 1
                continue

            chrom_score += score_contribution
            chrom_matched += 1

        total_score += chrom_score
        matched += chrom_matched
        missing += chrom_missing
        flipped += chrom_flipped
        details_by_chrom[chrom] = {
            "score": chrom_score,
            "matched": chrom_matched,
            "missing": chrom_missing,
        }

    return {
        "sample_id": sample_id,
        "total_score": total_score,
        "matched": matched,
        "missing": missing,
        "flipped": flipped,
        "total_weights": len(weights),
        "match_rate": matched / len(weights) if weights else 0,
        "by_chrom": details_by_chrom,
    }


def get_samples() -> list[str]:
    if not IMPUTED_DIR.exists():
        return []
    return sorted(
        d.name for d in IMPUTED_DIR.iterdir()
        if d.is_dir() and any(d.glob("*.vcf.gz"))
    )


def main():
    parser = argparse.ArgumentParser(description="Compute PGS from imputed genotypes")
    parser.add_argument("--pgs", type=Path, default=DEFAULT_PGS, help="PGS scoring file")
    parser.add_argument("--sample", help="Filter samples by substring")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-chromosome breakdown")
    args = parser.parse_args()

    print(f"Loading PGS weights from {args.pgs}...")
    weights = load_pgs_weights(args.pgs)
    print(f"  {len(weights)} variants (excluding XY/MT/Y)")

    samples = get_samples()
    if args.sample:
        samples = [s for s in samples if args.sample in s]

    if not samples:
        print("No imputed samples found.")
        return

    results = []
    for sample_id in samples:
        print(f"\nScoring {sample_id}...")
        result = compute_pgs(sample_id, weights)
        results.append(result)

        print(f"  PGS = {result['total_score']:.4f}")
        print(f"  Variants matched: {result['matched']:,} / {result['total_weights']:,} ({result['match_rate']:.1%})")
        print(f"  Missing: {result['missing']:,}")
        print(f"  Allele flips: {result['flipped']:,}")

        if args.verbose:
            print(f"\n  Per-chromosome breakdown:")
            for chrom, d in sorted(result["by_chrom"].items(),
                                    key=lambda x: (0, int(x[0])) if x[0].isdigit() else (1, 0)):
                print(f"    chr{chrom:>2}: score={d['score']:>8.4f}  matched={d['matched']:>5,}  missing={d['missing']:>4,}")

    if len(results) > 1:
        print(f"\n{'='*50}")
        print("Summary:")
        print(f"{'Sample':<35} {'PGS':>10} {'Match%':>8}")
        for r in results:
            print(f"  {r['sample_id']:<33} {r['total_score']:>10.4f} {r['match_rate']:>7.1%}")


if __name__ == "__main__":
    main()
