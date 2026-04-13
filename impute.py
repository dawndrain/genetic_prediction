"""Imputation pipeline: local Beagle or Michigan Imputation Server prep.

Usage:
  python impute.py beagle [--sample SAMPLE] [--chrom CHROM] [--parallel N]
  python impute.py michigan [--sample SAMPLE]

Beagle mode: runs local imputation with 1000G Phase 3 reference panel + GRCh37 maps.
Michigan mode: prepares bgzipped, indexed VCFs for upload to Michigan Imputation Server.
"""

import argparse
import gzip
import shlex
import subprocess
import sys
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# Paths
DATA_DIR = Path("data")
VCF_DIR = DATA_DIR / "vcf"
IMPUTED_DIR = DATA_DIR / "imputed"
MICHIGAN_DIR = DATA_DIR / "michigan"
RESOURCES_DIR = Path("resources")
REF_DIR = Path("reference_genomes_vcf")

BEAGLE_JAR = RESOURCES_DIR / "beagle.jar"
BEAGLE_URL = "https://faculty.washington.edu/browning/beagle/beagle.22Jul22.46e.jar"

GENETIC_MAP_ZIP = RESOURCES_DIR / "plink.GRCh37.map.zip"
GENETIC_MAP_URL = "https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.GRCh37.map.zip"

AUTOSOMES = list(range(1, 23))
ALL_CHROMS = list(range(1, 23)) + ["X"]


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress reporting."""
    if dest.exists():
        print(f"  {desc or dest.name} already exists, skipping download.")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc or dest.name}...")
    print(f"    {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"    Saved to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def setup_beagle():
    """Download Beagle JAR and GRCh37 genetic maps if not present."""
    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

    download_file(BEAGLE_URL, BEAGLE_JAR, "Beagle JAR")

    # Download and extract genetic maps
    map_example = RESOURCES_DIR / "plink.chr1.GRCh37.map"
    if not map_example.exists():
        download_file(GENETIC_MAP_URL, GENETIC_MAP_ZIP, "GRCh37 genetic maps")
        print("  Extracting genetic maps...")
        with zipfile.ZipFile(GENETIC_MAP_ZIP, "r") as zf:
            zf.extractall(RESOURCES_DIR)
        print(f"  Extracted to {RESOURCES_DIR}/")
    else:
        print("  Genetic maps already extracted.")


def get_samples() -> list[str]:
    """List sample IDs that have been converted to VCF."""
    if not VCF_DIR.exists():
        return []
    return sorted(
        d.name for d in VCF_DIR.iterdir()
        if d.is_dir() and (d / "1.vcf").exists()
    )


def ref_vcf_path(chrom) -> Path:
    """Path to the 1000G reference VCF for a chromosome."""
    gz_path = REF_DIR / f"chr{chrom}.1kg.phase3.v5a.vcf.gz"
    if gz_path.exists():
        return gz_path
    # Fall back to uncompressed
    plain_path = REF_DIR / f"chr{chrom}.1kg.phase3.v5a.vcf"
    if plain_path.exists():
        return plain_path
    raise FileNotFoundError(f"Reference VCF not found for chr{chrom}")


def genetic_map_path(chrom) -> Path:
    """Path to the GRCh37 genetic map for a chromosome."""
    return RESOURCES_DIR / f"plink.chr{chrom}.GRCh37.map"


def call_streaming(command: str) -> int:
    """Run a command, streaming stdout/stderr in real-time."""
    args = shlex.split(command)
    process = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
    )
    for line in process.stdout:
        print(line, end="")
    return process.wait()


def count_variants(vcf_path: Path) -> int:
    """Count non-header lines in a VCF (plain or gzipped)."""
    count = 0
    if str(vcf_path).endswith(".gz"):
        with gzip.open(vcf_path, "rt") as f:
            for line in f:
                if not line.startswith("#"):
                    count += 1
    else:
        with open(vcf_path) as f:
            for line in f:
                if not line.startswith("#"):
                    count += 1
    return count


# ---------------------------------------------------------------------------
# Mode A: Beagle local imputation
# ---------------------------------------------------------------------------

def _diploidize_x_vcf(input_vcf: Path) -> Path:
    """Convert haploid X chromosome genotypes to homozygous diploid.

    Beagle requires consistent ploidy. Males have haploid X genotypes (0, 1)
    which we convert to diploid (0/0, 1/1).
    """
    out_path = input_vcf.with_suffix(".diploid.vcf")
    if out_path.exists():
        return out_path

    with open(input_vcf) as f, open(out_path, "w") as w:
        for line in f:
            if line.startswith("#"):
                w.write(line)
                continue
            fields = line.rstrip("\n").split("\t")
            gt = fields[9]
            # Convert haploid to diploid homozygous
            if gt in ("0", "1"):
                fields[9] = f"{gt}/{gt}"
            elif gt == ".":
                fields[9] = "./."
            w.write("\t".join(fields) + "\n")
    return out_path


def beagle_impute_chrom(sample_id: str, chrom) -> dict:
    """Run Beagle imputation for one sample/chromosome."""
    input_vcf = VCF_DIR / sample_id / f"{chrom}.vcf"
    if not input_vcf.exists():
        return {"chrom": chrom, "status": "skipped", "reason": "no input VCF"}

    out_dir = IMPUTED_DIR / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / str(chrom)
    out_file = Path(f"{out_prefix}.vcf.gz")

    if out_file.exists():
        return {"chrom": chrom, "status": "skipped", "reason": "already imputed"}

    ref = ref_vcf_path(chrom)
    gmap = genetic_map_path(chrom)

    if not gmap.exists():
        return {"chrom": chrom, "status": "error", "reason": f"genetic map not found: {gmap}"}

    # For X chromosome, ensure all genotypes are diploid
    actual_input = input_vcf
    if str(chrom) == "X":
        actual_input = _diploidize_x_vcf(input_vcf)

    input_count = count_variants(actual_input)

    cmd = (
        f"java -Xmx3g -jar {BEAGLE_JAR} "
        f"gt={actual_input} "
        f"ref={ref} "
        f"map={gmap} "
        f"out={out_prefix} "
        f"gp=true"
    )

    print(f"\n--- Beagle: {sample_id} chr{chrom} ({input_count:,} input variants) ---")
    print(f"  cmd: {cmd}")

    rc = call_streaming(cmd)
    if rc != 0:
        return {"chrom": chrom, "status": "error", "reason": f"beagle exited with code {rc}"}

    output_count = count_variants(out_file) if out_file.exists() else 0

    print(f"  chr{chrom}: {input_count:,} -> {output_count:,} variants")
    return {
        "chrom": chrom,
        "status": "done",
        "input_variants": input_count,
        "output_variants": output_count,
    }


def beagle_concatenate(sample_id: str):
    """Concatenate per-chromosome imputed VCFs into one file."""
    out_dir = IMPUTED_DIR / sample_id
    all_path = out_dir / "all.vcf.gz"

    if all_path.exists():
        print(f"  Concatenated file already exists: {all_path}")
        return all_path

    print(f"  Concatenating imputed chromosomes for {sample_id}...")
    with gzip.open(all_path, "wt") as w:
        for chrom in ALL_CHROMS:
            chrom_file = out_dir / f"{chrom}.vcf.gz"
            if not chrom_file.exists():
                print(f"    WARNING: missing chr{chrom}")
                continue
            with gzip.open(chrom_file, "rt") as r:
                for line in r:
                    if line.startswith("#"):
                        if chrom == 1:
                            w.write(line)
                    else:
                        w.write(line)
    print(f"  Saved: {all_path}")
    return all_path


def beagle_report(sample_id: str, results: list[dict]):
    """Print a summary report of Beagle imputation."""
    print(f"\n{'='*60}")
    print(f"Imputation report: {sample_id}")
    print(f"{'='*60}")

    total_in = 0
    total_out = 0
    for r in results:
        if r["status"] == "done":
            total_in += r["input_variants"]
            total_out += r["output_variants"]
            print(f"  chr{r['chrom']:>2}: {r['input_variants']:>8,} -> {r['output_variants']:>10,}")
        else:
            print(f"  chr{r['chrom']:>2}: {r['status']} ({r.get('reason', '')})")

    if total_in > 0:
        print(f"  {'':->50}")
        print(f"  Total: {total_in:>8,} -> {total_out:>10,}  ({total_out/total_in:.1f}x)")


def run_beagle(sample_filter: str | None, chrom_filter: int | None, parallel: int):
    """Run Beagle imputation for selected samples/chromosomes."""
    setup_beagle()

    samples = get_samples()
    if not samples:
        print("No converted VCF samples found in data/vcf/. Run convert_to_vcf.py first.")
        sys.exit(1)

    if sample_filter:
        samples = [s for s in samples if sample_filter in s]

    if not samples:
        print(f"No samples matching '{sample_filter}'")
        sys.exit(1)

    print(f"Samples to impute: {samples}")
    chroms = [chrom_filter] if chrom_filter else ALL_CHROMS

    for sample_id in samples:
        if parallel > 1 and len(chroms) > 1:
            results = []
            with ProcessPoolExecutor(max_workers=parallel) as executor:
                futures = {
                    executor.submit(beagle_impute_chrom, sample_id, c): c
                    for c in chroms
                }
                for future in as_completed(futures):
                    results.append(future.result())
            results.sort(key=lambda r: r["chrom"])
        else:
            results = [beagle_impute_chrom(sample_id, c) for c in chroms]

        beagle_report(sample_id, results)

        if not chrom_filter:
            beagle_concatenate(sample_id)


# ---------------------------------------------------------------------------
# Mode B: Michigan Imputation Server prep
# ---------------------------------------------------------------------------

def _read_vcf_variants(vcf_path: Path) -> tuple[list[str], dict[str, str]]:
    """Read a VCF file, return (header_lines, {(chrom,pos) -> data_line}).

    Strips 'chr' prefix, deduplicates, but does NOT sort (caller handles that).
    """
    header_lines = []
    variants = {}  # (chrom, pos) -> full line (with trailing newline stripped)

    with open(vcf_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                header_lines.append(line)
            else:
                fields = line.split("\t")
                if fields[0].startswith("chr"):
                    fields[0] = fields[0][3:]
                key = (fields[0], fields[1])
                if key not in variants:
                    variants[key] = fields
    return header_lines, variants


def _write_bgzipped_vcf(out_dir: Path, chrom: int, header: str, data_lines: list[str]) -> Path:
    """Write a VCF, bgzip it, and tabix index it. Returns .vcf.gz path."""
    temp_vcf = out_dir / f"chr{chrom}.vcf"
    out_vcf = out_dir / f"chr{chrom}.vcf.gz"

    with open(temp_vcf, "w") as f:
        f.write(header)
        for line in data_lines:
            f.write(line + "\n")

    subprocess.run(["bgzip", "-f", str(temp_vcf)], check=True)
    return out_vcf


def prep_michigan_dup(sample_id: str, n_copies: int = 5):
    """Prepare Michigan files by duplicating one sample N times."""
    out_dir = MICHIGAN_DIR / f"{sample_id}_dup{n_copies}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_names = [f"SAMPLE{i+1}" for i in range(n_copies)]

    print(f"\n{'='*60}")
    print(f"Michigan prep (duplicate): {sample_id} x{n_copies}")
    print(f"  Sample columns: {', '.join(sample_names)}")
    print(f"{'='*60}")

    for chrom in AUTOSOMES:
        input_vcf = VCF_DIR / sample_id / f"{chrom}.vcf"
        if not input_vcf.exists():
            print(f"  chr{chrom}: no input VCF, skipping")
            continue

        header_lines, variants = _read_vcf_variants(input_vcf)

        # Rebuild header with multiple sample columns
        new_header_lines = []
        for line in header_lines:
            if line.startswith("#CHROM"):
                cols = line.split("\t")[:9]  # fixed VCF columns
                new_header_lines.append("\t".join(cols + sample_names))
            else:
                new_header_lines.append(line)
        header = "\n".join(new_header_lines) + "\n"

        # Sort variants and duplicate the genotype column
        sorted_keys = sorted(variants.keys(), key=lambda k: int(k[1]))
        data_lines = []
        for key in sorted_keys:
            fields = variants[key]
            gt = fields[9]  # original genotype
            # fixed columns + same genotype repeated N times
            data_lines.append("\t".join(fields[:9] + [gt] * n_copies))

        out_vcf = _write_bgzipped_vcf(out_dir, chrom, header, data_lines)
        print(f"  chr{chrom}: {len(data_lines):,} variants -> {out_vcf}")

    _print_michigan_instructions(out_dir)
    return out_dir


def prep_michigan_multi(sample_ids: list[str]):
    """Merge multiple samples into multi-sample VCFs for Michigan.

    Variants present in some but not all samples get ./. for missing samples.
    """
    out_dir = MICHIGAN_DIR / "multi_" + "_".join(s[:10] for s in sample_ids)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Michigan prep (multi-sample): {len(sample_ids)} samples")
    for s in sample_ids:
        print(f"  - {s}")
    print(f"{'='*60}")

    for chrom in AUTOSOMES:
        # Load all samples for this chromosome
        all_variants = {}  # (chrom, pos) -> {sample_idx: fields}
        ref_header = None
        all_keys = set()

        for i, sid in enumerate(sample_ids):
            input_vcf = VCF_DIR / sid / f"{chrom}.vcf"
            if not input_vcf.exists():
                continue
            header_lines, variants = _read_vcf_variants(input_vcf)
            if ref_header is None:
                ref_header = header_lines
            for key, fields in variants.items():
                if key not in all_variants:
                    all_variants[key] = {}
                all_variants[key][i] = fields

        if ref_header is None:
            print(f"  chr{chrom}: no input VCFs found, skipping")
            continue

        # Build header with all sample names
        new_header_lines = []
        for line in ref_header:
            if line.startswith("#CHROM"):
                cols = line.split("\t")[:9]
                new_header_lines.append("\t".join(cols + sample_ids))
            else:
                new_header_lines.append(line)
        header = "\n".join(new_header_lines) + "\n"

        # Merge variants across samples, sorted by position
        sorted_keys = sorted(all_variants.keys(), key=lambda k: int(k[1]))
        data_lines = []
        for key in sorted_keys:
            sample_data = all_variants[key]
            # Use any sample that has this variant for the fixed columns (ref/alt/etc)
            ref_idx = next(iter(sample_data))
            ref_fields = sample_data[ref_idx]
            fixed = ref_fields[:9]

            # Build genotype columns
            gts = []
            for i in range(len(sample_ids)):
                if i in sample_data:
                    gts.append(sample_data[i][9])
                else:
                    gts.append("./.")
            data_lines.append("\t".join(fixed + gts))

        out_vcf = _write_bgzipped_vcf(out_dir, chrom, header, data_lines)
        print(f"  chr{chrom}: {len(data_lines):,} variants ({len(sample_ids)} samples) -> {out_vcf}")

    _print_michigan_instructions(out_dir)
    return out_dir


def _print_michigan_instructions(out_dir: Path):
    """Print upload instructions."""
    print(f"\nFiles ready in: {out_dir}/")
    print(f"  Upload only the chr*.vcf.gz files (ignore .tbi)")
    print()
    print("Michigan Imputation Server instructions:")
    print("  1. Go to https://imputationserver.sph.umich.edu/")
    print("  2. Run > Genotype Imputation (Minimac4)")
    print("  3. Settings:")
    print("     - Reference Panel: 1000g-phase-3-v5 (GRCh37/hg19)")
    print("     - Build: GRCh37/hg19")
    print("     - r2 Filter: 0.3")
    print("     - Phasing: Eagle v2.4")
    print("     - Population: Mixed/ALL")


def run_michigan(mode: str, sample_filter: str | None):
    """Prepare VCFs for Michigan Imputation Server upload."""
    samples = get_samples()
    if not samples:
        print("No converted VCF samples found in data/vcf/. Run convert_to_vcf.py first.")
        sys.exit(1)

    if sample_filter:
        samples = [s for s in samples if sample_filter in s]

    if not samples:
        print(f"No samples matching '{sample_filter}'")
        sys.exit(1)

    if mode == "dup":
        # Duplicate each sample 5 times (separate upload per sample)
        for sample_id in samples:
            prep_michigan_dup(sample_id, n_copies=5)
    elif mode == "multi":
        # Merge all matched samples + duplicate first 2 to reach 5
        if len(samples) < 5:
            print(f"Only {len(samples)} samples, padding with duplicates to reach 5...")
            padded = list(samples)
            i = 0
            while len(padded) < 5:
                padded.append(samples[i % len(samples)])
                i += 1
            # Give duplicates unique names for the header, but same data
            prep_michigan_multi_padded(samples, padded)
        else:
            prep_michigan_multi(samples)


def prep_michigan_multi_padded(real_samples: list[str], padded_order: list[str]):
    """Merge samples with padding (duplicated samples get suffixed names)."""
    # Build unique column names
    seen = {}
    col_names = []
    for s in padded_order:
        count = seen.get(s, 0)
        seen[s] = count + 1
        col_names.append(s if count == 0 else f"{s}_dup{count}")

    out_dir = MICHIGAN_DIR / "multi_all"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Michigan prep (multi-sample, padded to 5)")
    for name, source in zip(col_names, padded_order):
        tag = " (duplicate)" if name != source else ""
        print(f"  - {name}{tag}")
    print(f"{'='*60}")

    for chrom in AUTOSOMES:
        # Load unique samples
        sample_data = {}  # sample_id -> {(chrom_name, pos): fields}
        ref_header = None

        for sid in real_samples:
            input_vcf = VCF_DIR / sid / f"{chrom}.vcf"
            if not input_vcf.exists():
                continue
            header_lines, variants = _read_vcf_variants(input_vcf)
            if ref_header is None:
                ref_header = header_lines
            sample_data[sid] = variants

        if ref_header is None:
            print(f"  chr{chrom}: no input VCFs found, skipping")
            continue

        # Collect union of all variant positions
        all_keys = set()
        for variants in sample_data.values():
            all_keys.update(variants.keys())

        # Build header
        new_header_lines = []
        for line in ref_header:
            if line.startswith("#CHROM"):
                cols = line.split("\t")[:9]
                new_header_lines.append("\t".join(cols + col_names))
            else:
                new_header_lines.append(line)
        header = "\n".join(new_header_lines) + "\n"

        # Build data lines
        sorted_keys = sorted(all_keys, key=lambda k: int(k[1]))
        data_lines = []
        for key in sorted_keys:
            # Find a sample that has this variant for fixed columns
            ref_fields = None
            for sid in real_samples:
                if sid in sample_data and key in sample_data[sid]:
                    ref_fields = sample_data[sid][key]
                    break
            fixed = ref_fields[:9]

            gts = []
            for source_sid in padded_order:
                if source_sid in sample_data and key in sample_data[source_sid]:
                    gts.append(sample_data[source_sid][key][9])
                else:
                    gts.append("./.")
            data_lines.append("\t".join(fixed + gts))

        out_vcf = _write_bgzipped_vcf(out_dir, chrom, header, data_lines)
        print(f"  chr{chrom}: {len(data_lines):,} variants ({len(col_names)} columns) -> {out_vcf}")

    _print_michigan_instructions(out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Imputation pipeline: local Beagle or Michigan server prep"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Beagle subcommand
    beagle_parser = subparsers.add_parser("beagle", help="Run local Beagle imputation")
    beagle_parser.add_argument("--sample", help="Filter samples by substring match")
    beagle_parser.add_argument("--chrom", help="Run only this chromosome (1-22 or X)")
    beagle_parser.add_argument("--parallel", type=int, default=1,
                               help="Number of chromosomes to impute in parallel")

    # Michigan subcommand
    michigan_parser = subparsers.add_parser("michigan", help="Prepare files for Michigan server")
    michigan_parser.add_argument("--sample", help="Filter samples by substring match")
    michigan_parser.add_argument("--merge", choices=["dup", "multi"], default="dup",
                                 help="'dup': duplicate each sample 5x (default). "
                                      "'multi': merge all samples into one VCF (padded to 5).")

    args = parser.parse_args()

    if args.mode == "beagle":
        chrom = args.chrom
        if chrom and chrom.isdigit():
            chrom = int(chrom)
        run_beagle(args.sample, chrom, args.parallel)
    elif args.mode == "michigan":
        run_michigan(args.merge, args.sample)


if __name__ == "__main__":
    main()
