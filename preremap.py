"""Pre-remap all non-build-37 genomes and save to disk.

Phase 1: Parallel scan to determine build (no network I/O)
Phase 2: Sequential remap of build-36 genomes (downloads chain files)
"""

import warnings
from multiprocessing import Pool
from pathlib import Path

from snps import SNPs

GENOMES_DIR = Path("/Users/dawndrain/Code/genetics_regression/genomes")
REMAPPED_DIR = Path("data/genomes_b37")
N_WORKERS = 8


def scan_build(genome_path_str: str) -> tuple[str, int | None]:
    """Determine the build of a genome file. Returns (filename, build_or_None)."""
    warnings.filterwarnings("ignore")
    try:
        s = SNPs(genome_path_str)
        return (Path(genome_path_str).name, s.build)
    except Exception:
        return (Path(genome_path_str).name, None)


def main():
    REMAPPED_DIR.mkdir(parents=True, exist_ok=True)

    genome_files = sorted(f for f in GENOMES_DIR.iterdir()
                          if f.is_file() and "exome" not in f.name.lower()
                          and f.stat().st_size < 50_000_000)

    # Skip already processed
    already_done = set()
    for p in REMAPPED_DIR.iterdir():
        if p.is_file() or p.is_symlink():
            already_done.add(p.name)

    todo = [f for f in genome_files if f.name not in already_done]
    print(f"{len(genome_files)} genomes total, {len(already_done)} already done, {len(todo)} to process", flush=True)

    if not todo:
        print("Nothing to do!")
        return

    # Phase 1: Parallel scan for build
    print(f"\n=== Phase 1: Scan builds ({N_WORKERS} workers) ===", flush=True)
    paths = [str(f) for f in todo]

    build_37 = []
    need_remap = []
    skipped = 0
    done = 0

    with Pool(N_WORKERS) as pool:
        for name, build in pool.imap_unordered(scan_build, paths):
            done += 1
            if build == 37:
                build_37.append(name)
            elif build and build != 37:
                need_remap.append(name)
            else:
                skipped += 1
            if done % 100 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} scanned "
                      f"(b37={len(build_37)}, remap={len(need_remap)}, skip={skipped})", flush=True)

    print(f"\n  Build 37: {len(build_37)} (symlink)")
    print(f"  Need remap: {len(need_remap)}")
    print(f"  Skipped: {skipped}")

    # Symlink build-37 files
    for name in build_37:
        out_path = REMAPPED_DIR / name
        if not out_path.exists():
            out_path.symlink_to((GENOMES_DIR / name).resolve())

    # Phase 2: Sequential remap
    if need_remap:
        print(f"\n=== Phase 2: Remap {len(need_remap)} genomes ===", flush=True)
        warnings.filterwarnings("ignore")
        for i, name in enumerate(need_remap):
            out_path = REMAPPED_DIR / name
            if out_path.exists():
                continue
            print(f"  [{i+1}/{len(need_remap)}] {name}...", end=" ", flush=True)
            try:
                s = SNPs(str(GENOMES_DIR / name))
                s.remap(37)
                s.snps.to_csv(str(out_path))
                print(f"done ({s.count} SNPs)")
            except Exception as e:
                print(f"ERROR: {e}")

    total = sum(1 for f in REMAPPED_DIR.iterdir() if f.is_file() or f.is_symlink())
    print(f"\nDone! {total} genomes ready in {REMAPPED_DIR}/")


if __name__ == "__main__":
    main()
