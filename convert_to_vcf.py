"""Convert DTC genotype files to per-chromosome VCFs.

For each file in data/genomes/:
  1. Load with snps library (auto-detects 23andMe / Ancestry / FTDNA)
  2. Remap to build 37 if needed
  3. Deduplicate by (chrom, pos)
  4. Export to VCF with QC filtering
  5. Shard into per-chromosome VCFs

Output: data/vcf/{sample_id}/{chrom}.vcf for chromosomes 1-22
"""

import sys
from collections import defaultdict
from pathlib import Path

from snps import SNPs


DATA_DIR = Path("data")
GENOMES_DIR = DATA_DIR / "genomes"
VCF_DIR = DATA_DIR / "vcf"


def load_and_remap(genome_path: Path) -> SNPs:
    """Load a DTC genome file and ensure it's on build 37."""
    print(f"Loading {genome_path.name}...")
    s = SNPs(str(genome_path))

    print(f"  Source: {s.source}")
    print(f"  Build: {s.build}")
    print(f"  SNP count: {s.count}")

    if s.build != 37:
        print(f"  Remapping from build {s.build} to 37...")
        chromosomes_remapped, chromosomes_not_remapped = s.remap(37)
        print(f"  Remapped: {chromosomes_remapped}")
        if chromosomes_not_remapped:
            print(f"  WARNING: Could not remap: {chromosomes_not_remapped}")
        print(f"  New build: {s.build}")
        print(f"  SNP count after remap: {s.count}")

    return s


def deduplicate(s: SNPs) -> SNPs:
    """Remove duplicate (chrom, pos) entries."""
    before = len(s.snps)
    s._snps = s._snps.drop_duplicates(subset=["chrom", "pos"])
    after = len(s.snps)
    if before != after:
        print(f"  Removed {before - after} duplicate positions ({before} -> {after})")
    return s


def export_vcf(s: SNPs, sample_id: str) -> Path:
    """Export SNPs to a single VCF file.

    The snps library prepends "output/" to the filename, so we give it a path
    relative to that and ensure the output directory exists.
    """
    # snps writes to output/{filename} - so we want output/vcf/{sample_id}.vcf
    # and we pass "vcf/{sample_id}" as the filename
    output_dir = Path("output") / "vcf"
    output_dir.mkdir(parents=True, exist_ok=True)

    vcf_filename = f"vcf/{sample_id}"
    expected_path = Path("output") / "vcf" / f"{sample_id}.vcf"

    print(f"  Exporting VCF...")
    saved_path = s.to_vcf(filename=vcf_filename, qc_only=True, qc_filter=True)
    print(f"  Saved to: {saved_path}")

    # Move to our target location
    actual = Path(saved_path) if saved_path and Path(saved_path).exists() else expected_path
    if not actual.exists():
        raise FileNotFoundError(f"Could not find exported VCF at {actual}")

    target_dir = VCF_DIR / sample_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{sample_id}.vcf"
    actual.rename(target)
    print(f"  Moved to: {target}")
    return target


def shard_by_chromosome(vcf_path: Path, sample_id: str) -> dict[str, Path]:
    """Split a full-genome VCF into per-chromosome files."""
    print(f"  Sharding {vcf_path} by chromosome...")

    chrom_to_lines = defaultdict(list)
    header_lines = []

    for line in vcf_path.read_text().splitlines():
        if line.startswith("#"):
            header_lines.append(line)
        else:
            chrom = line.split("\t", 1)[0]
            chrom_to_lines[chrom].append(line)

    header = "\n".join(header_lines) + "\n"
    out_dir = VCF_DIR / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    chrom_paths = {}
    for chrom in sorted(chrom_to_lines.keys(), key=_chrom_sort_key):
        out_path = out_dir / f"{chrom}.vcf"
        out_path.write_text(header + "\n".join(chrom_to_lines[chrom]) + "\n")
        count = len(chrom_to_lines[chrom])
        chrom_paths[chrom] = out_path
        print(f"    chr{chrom}: {count} variants -> {out_path}")

    return chrom_paths


def _chrom_sort_key(chrom: str):
    """Sort chromosomes numerically, with X/Y/MT at end."""
    try:
        return (0, int(chrom))
    except ValueError:
        order = {"X": 1, "Y": 2, "MT": 3}
        return (1, order.get(chrom, 99))


def convert_one(genome_path: Path) -> dict:
    """Full conversion pipeline for a single genome file."""
    sample_id = genome_path.name

    print(f"\n{'='*60}")
    print(f"Processing: {sample_id}")
    print(f"{'='*60}")

    # Check if already done
    out_dir = VCF_DIR / sample_id
    if out_dir.exists() and any(out_dir.glob("*.vcf")):
        existing = list(out_dir.glob("[0-9]*.vcf")) + list(out_dir.glob("X.vcf"))
        if len(existing) >= 22:
            print(f"  Already converted ({len(existing)} chromosome VCFs found). Skipping.")
            return {"sample_id": sample_id, "status": "skipped", "chrom_count": len(existing)}

    s = load_and_remap(genome_path)
    s = deduplicate(s)
    vcf_path = export_vcf(s, sample_id)
    chrom_paths = shard_by_chromosome(vcf_path, sample_id)

    # Clean up the full-genome VCF to save space
    vcf_path.unlink()

    # Report autosomal stats
    autosomal = {k: v for k, v in chrom_paths.items() if k not in ("X", "Y", "MT")}
    total_variants = sum(
        sum(1 for line in p.read_text().splitlines() if not line.startswith("#"))
        for p in autosomal.values()
    )

    print(f"\n  Summary for {sample_id}:")
    print(f"    Chromosomes: {len(chrom_paths)} ({len(autosomal)} autosomal)")
    print(f"    Total autosomal variants: {total_variants:,}")

    return {
        "sample_id": sample_id,
        "status": "converted",
        "chrom_count": len(chrom_paths),
        "autosomal_variants": total_variants,
    }


def main():
    genome_files = sorted(GENOMES_DIR.iterdir())
    genome_files = [f for f in genome_files if f.is_file() and not f.name.startswith(".")]

    if not genome_files:
        print(f"No genome files found in {GENOMES_DIR}")
        sys.exit(1)

    print(f"Found {len(genome_files)} genome files:")
    for f in genome_files:
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

    results = []
    for genome_path in genome_files:
        result = convert_one(genome_path)
        results.append(result)

    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}")
    for r in results:
        if r["status"] == "converted":
            print(f"  {r['sample_id']}: {r['autosomal_variants']:,} autosomal variants across {r['chrom_count']} chromosomes")
        else:
            print(f"  {r['sample_id']}: {r['status']} ({r['chrom_count']} existing)")


if __name__ == "__main__":
    main()
