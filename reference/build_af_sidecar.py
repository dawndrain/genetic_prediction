"""Build the allele-frequency sidecar used for partial-overlap normalization.

Most PGS Catalog weight files carry no allele-frequency column, so when a
genome only covers part of a score's panel the scoring code can't compute the
matched subset's expected mean/variance and has to fall back on the
random-missingness (f·μ, σ·√f) approximation — which is badly biased for
array data (chip content is not a random subset of any panel).

This script makes that information available without requiring users to
download 1000 Genomes themselves: it streams the local 1KG Phase 3 VCFs once
and writes per-super-population ALT allele frequencies for every GRCh37
position referenced by any weight file on disk, plus the rsID where a weight
file provides one.

Output: data/weight_snp_af_grch37.tsv.gz with columns
    rsid  chrom  pos  ref  alt  AFR  AMR  EAS  EUR  SAS
Ship it alongside the weight files (e.g. as a GitHub release asset fetched by
`genepred fetch-weights`); genepred.scoring picks it up automatically from
the data dir or genepred/resources/.

Requires: data/1kg/ALL.chr*.vcf.gz (reference/onekg/download_1kg.sh) and
data/pgs_scoring_files/*.txt.gz (genepred fetch-weights).
"""

from __future__ import annotations

import gzip
import multiprocessing as mp
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genepred.catalog import list_weight_files, read_header  # noqa: E402
from genepred.paths import data_dir, kg_dir, open_maybe_gz  # noqa: E402

OUT = data_dir() / "weight_snp_af_grch37.tsv.gz"
POPS = ("AFR", "AMR", "EAS", "EUR", "SAS")
_AF_RE = {p: re.compile(rf"{p}_AF=([0-9.eE+-]+)") for p in POPS}


def collect_targets() -> tuple[dict[str, set[int]], dict[tuple[str, int], str]]:
    """GRCh37 positions referenced by any weight file, plus pos -> rsID."""
    want: dict[str, set[int]] = defaultdict(set)
    pos2rs: dict[tuple[str, int], str] = {}
    for _, wf in list_weight_files():
        if read_header(wf).get("genome_build", "GRCh37") not in (
            "GRCh37", "hg19", "NCBI37",
        ):
            continue
        with open_maybe_gz(wf) as f:
            header = next((ln for ln in f if not ln.startswith("#")), None)
            if header is None:
                continue
            cols = {c: i for i, c in enumerate(header.rstrip("\n").split("\t"))}
            i_chr, i_pos = cols.get("chr_name"), cols.get("chr_position")
            i_rs = cols.get("hm_rsID", cols.get("rsID", cols.get("rsid")))
            if i_chr is None or i_pos is None:
                continue
            for line in f:
                r = line.rstrip("\n").split("\t")
                try:
                    chrom, pos = r[i_chr].lstrip("chr"), int(r[i_pos])
                except (ValueError, IndexError):
                    continue
                want[chrom].add(pos)
                if i_rs is not None and i_rs < len(r) and r[i_rs].startswith("rs"):
                    pos2rs[(chrom, pos)] = r[i_rs]
    return want, pos2rs


def stream_chrom(args) -> list[str]:
    chrom, positions, pos2rs = args
    vcf = next(kg_dir().glob(f"ALL.chr{chrom}.*.vcf.gz"), None)
    rows: list[str] = []
    if vcf is None or not positions:
        return rows
    with gzip.open(vcf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            t1 = line.find("\t")
            t2 = line.find("\t", t1 + 1)
            try:
                pos = int(line[t1 + 1 : t2])
            except ValueError:
                continue
            if pos not in positions:
                continue
            r = line.split("\t", 8)
            ref, alt, info = r[3].upper(), r[4].upper(), r[7]
            if len(ref) != 1 or len(alt) != 1 or "," in alt:
                continue
            afs = []
            for p in POPS:
                m = _AF_RE[p].search(info)
                if m is None:
                    break
                afs.append(f"{float(m.group(1)):.4f}")
            if len(afs) != len(POPS):
                continue
            rsid = pos2rs.get((chrom, pos), ".")
            rows.append(
                f"{rsid}\t{chrom}\t{pos}\t{ref}\t{alt}\t" + "\t".join(afs) + "\n"
            )
    return rows


def main() -> None:
    want, pos2rs = collect_targets()
    n_target = sum(len(v) for v in want.values())
    print(f"{n_target:,} target positions across {len(want)} chromosomes",
          file=sys.stderr)
    jobs = [
        (chrom, positions, {k: v for k, v in pos2rs.items() if k[0] == chrom})
        for chrom, positions in sorted(want.items())
    ]
    n_written = 0
    with gzip.open(OUT, "wt", compresslevel=9) as out, \
            mp.get_context("fork").Pool(min(8, len(jobs))) as pool:
        out.write("rsid\tchrom\tpos\tref\talt\t" + "\t".join(POPS) + "\n")
        for rows in pool.imap(stream_chrom, jobs):
            out.writelines(rows)
            n_written += len(rows)
            print(f"  ... {n_written:,} rows", file=sys.stderr)
    print(f"wrote {OUT} ({n_written:,} SNPs, "
          f"{OUT.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
