"""Drop duplicate-position rows from 1KG Phase 3 VCFs so Beagle accepts
them. Multi-allelic sites in 1KG are split into adjacent rows at the
same position; Beagle 5.5 rejects any reference with repeated positions.
This keeps the first row at each position (the most common ALT) and
drops the rest — fine for imputation since the array input only ever
has biallelic calls anyway.

Output goes to data/1kg_dedup/ as bgzipped+tabix-indexed VCFs.
"""

import gzip
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

KG_DIR = Path("data/1kg")
OUT_DIR = Path("data/1kg_dedup")


def dedup_one(chrom: str):
    src = next(KG_DIR.glob(f"ALL.chr{chrom}.*.vcf.gz"), None)
    if src is None:
        return chrom, 0, 0, "no source"
    out = OUT_DIR / f"chr{chrom}.vcf.gz"
    if out.exists() and out.stat().st_size > 1_000_000:
        return chrom, 0, 0, "exists"
    n_in = n_out = 0
    last_pos = -1
    bgz = subprocess.Popen(
        ["bgzip", "-c"], stdin=subprocess.PIPE, stdout=open(out, "wb")
    )
    assert bgz.stdin is not None
    with gzip.open(src, "rt") as f:
        for line in f:
            if line.startswith("#"):
                bgz.stdin.write(line.encode())
                continue
            n_in += 1
            t1 = line.find("\t")
            t2 = line.find("\t", t1 + 1)
            pos = int(line[t1 + 1 : t2])
            if pos == last_pos:
                continue
            last_pos = pos
            bgz.stdin.write(line.encode())
            n_out += 1
    bgz.stdin.close()
    bgz.wait()
    subprocess.run(["tabix", "-f", "-p", "vcf", str(out)], check=True)
    return chrom, n_in, n_out, "ok"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chroms = sys.argv[1:] or [str(c) for c in range(1, 23)]
    n_proc = min(len(chroms), int(os.environ.get("COO_CPUS") or os.cpu_count() or 8))
    with mp.get_context("fork").Pool(n_proc) as pool:
        for chrom, n_in, n_out, status in pool.imap_unordered(dedup_one, chroms):
            dropped = n_in - n_out
            print(
                f"chr{chrom:>2}: {n_in:>10,} → {n_out:>10,} "
                f"(dropped {dropped:>7,}, {dropped / max(n_in, 1):.2%}) "
                f"[{status}]",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
