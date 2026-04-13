"""Filter reference VCFs to only PGS-relevant positions.

Creates a slimmed reference panel that makes Beagle imputation ~50-100x faster
by only including variants at positions in the PGS scoring file.
"""

import gzip
from pathlib import Path

DATA_DIR = Path("data")
REF_DIR = Path("reference_genomes_vcf")
SLIM_REF_DIR = Path("reference_genomes_vcf_slim")
PGS_PATH = DATA_DIR / "PGS001229.txt"


def load_pgs_positions(pgs_path: Path) -> dict[str, set[str]]:
    """Load PGS positions grouped by chromosome. Returns {chrom: {pos1, pos2, ...}}."""
    positions = {}
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
            if not pos:
                continue
            if chrom not in positions:
                positions[chrom] = set()
            positions[chrom].add(pos)
    return positions


def shrink_chromosome(chrom, positions: set[str]):
    """Filter one chromosome's reference VCF to only PGS positions."""
    gz_path = REF_DIR / f"chr{chrom}.1kg.phase3.v5a.vcf.gz"
    if not gz_path.exists():
        plain_path = REF_DIR / f"chr{chrom}.1kg.phase3.v5a.vcf"
        if plain_path.exists():
            gz_path = plain_path
        else:
            print(f"  chr{chrom}: reference not found, skipping")
            return

    out_path = SLIM_REF_DIR / f"chr{chrom}.1kg.phase3.v5a.vcf"

    total = 0
    kept = 0

    opener = gzip.open if str(gz_path).endswith(".gz") else open
    with opener(gz_path, "rt") as r, open(out_path, "w") as w:
        for line in r:
            if line.startswith("#"):
                w.write(line)
                continue
            total += 1
            pos = line.split("\t", 3)[1]
            if pos in positions:
                w.write(line)
                kept += 1

    print(f"  chr{chrom}: {kept:,} / {total:,} variants kept ({100*kept/total:.2f}%)")
    return kept, total


def main():
    print("Loading PGS positions...")
    positions = load_pgs_positions(PGS_PATH)
    total_positions = sum(len(v) for v in positions.values())
    print(f"  {total_positions} positions across {len(positions)} chromosomes")

    SLIM_REF_DIR.mkdir(parents=True, exist_ok=True)

    print("\nFiltering reference VCFs...")
    total_kept = 0
    total_total = 0

    chroms = list(range(1, 23)) + ["X"]
    for chrom in chroms:
        chrom_str = str(chrom)
        pos_set = positions.get(chrom_str, set())
        if not pos_set:
            print(f"  chr{chrom}: no PGS positions, skipping")
            continue
        result = shrink_chromosome(chrom, pos_set)
        if result:
            total_kept += result[0]
            total_total += result[1]

    print(f"\nDone! {total_kept:,} / {total_total:,} variants kept overall")
    print(f"Output: {SLIM_REF_DIR}/")


if __name__ == "__main__":
    main()
