"""Local imputation with Beagle 5 against 1KG Phase 3 (GRCh37).

The fully-automated path. Inputs are conformed per-chromosome VCFs (REF/
ALT matching the 1KG panel — Beagle drops anything else); output is
data/imputed/<name>/chr*.vcf.gz with phased GTs and DS dosages.

The 1KG reference must be position-deduped first (Beagle 5.5 rejects
the multi-allelic same-position rows on chr12/14/17). reference/
dedup_1kg.py does this — equivalent to `bcftools norm -d both` but
without the bcftools dependency.

Wall-clock: ~3–4 min genome-wide on 64 cores (8 chroms in parallel ×
8 threads each); ~25 min on a 4-core laptop with parallel=2.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from genepred.io import (
    bgzf_compress,
    load_genotype_by_chrom,
    parse_chroms,
    write_conformed_vcf,
)
from genepred.paths import data_dir, find_tool, kg_dir, open_maybe_gz, tools_dir

BEAGLE_URL = "https://faculty.washington.edu/browning/beagle/beagle.27Feb25.75f.jar"
GENETIC_MAP_URL = (
    "https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.GRCh37.map.zip"
)


def setup() -> tuple[Path, Path]:
    """Ensure beagle.jar and the GRCh37 genetic maps are present.
    Returns (jar_path, maps_dir)."""
    td = tools_dir()
    td.mkdir(parents=True, exist_ok=True)
    jar = td / "beagle.jar"
    if not jar.exists():
        print(f"  fetching Beagle → {jar}", file=sys.stderr)
        urllib.request.urlretrieve(BEAGLE_URL, jar)
    maps = td / "genetic_maps_GRCh37"
    if not (maps / "plink.chr1.GRCh37.map").exists():
        maps.mkdir(parents=True, exist_ok=True)
        zip_path = td / "plink.GRCh37.map.zip"
        print(f"  fetching GRCh37 genetic maps → {maps}", file=sys.stderr)
        urllib.request.urlretrieve(GENETIC_MAP_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(maps)
    return jar, maps


def _ref_panel(chrom: str) -> Path | None:
    """Find the 1KG Phase3 VCF for one chromosome. Prefers the
    position-deduped copy (raw 1KG has multi-allelic sites split into
    adjacent same-position rows on chr12/14/17, which Beagle 5.5
    rejects). Run `reference/onekg/dedup_1kg.py` once to produce these."""
    dedup = data_dir() / "1kg_dedup" / f"chr{chrom}.vcf.gz"
    if dedup.exists():
        return dedup
    return next(kg_dir().glob(f"ALL.chr{chrom}.*.vcf.gz"), None)


def _require_ref_panels(chrom_list: list[str]) -> None:
    """Fail fast with the exact commands to run if any requested
    chromosome is missing a Beagle-compatible reference panel."""
    missing = [c for c in chrom_list if _ref_panel(c) is None]
    if not missing:
        return
    kg = kg_dir()
    dedup = data_dir() / "1kg_dedup"
    have_raw = any(list(kg.glob(f"ALL.chr{c}.*.vcf.gz")) for c in missing)
    if have_raw:
        hint = (
            "Raw 1KG VCFs are present but not deduped "
            "(Beagle 5.5 rejects multi-allelic dup-position rows).\n"
            "  python reference/onekg/dedup_1kg.py"
        )
    else:
        hint = (
            "Download the 1KG Phase 3 reference first (~15 GB), then dedup:\n"
            "  ./reference/onekg/download_1kg.sh\n"
            "  python reference/onekg/dedup_1kg.py"
        )
    raise FileNotFoundError(
        f"Beagle reference panel not found for chr{', chr'.join(missing)}.\n"
        f"Searched: {dedup}/  and  {kg}/\n\n{hint}"
    )


def _diploidize_x(input_vcf: Path) -> Path:
    """Beagle wants consistent ploidy; convert haploid X (0,1,.) to
    homozygous diploid (0/0, 1/1, ./.)."""
    out = input_vcf.with_suffix(".dip.vcf")
    with open(input_vcf) as r, open(out, "w") as w:
        for line in r:
            if line.startswith("#"):
                w.write(line)
                continue
            f = line.rstrip("\n").split("\t")
            if f[9] in ("0", "1"):
                f[9] = f"{f[9]}/{f[9]}"
            elif f[9] == ".":
                f[9] = "./."
            w.write("\t".join(f) + "\n")
    return out


def _impute_chrom(
    java: str,
    jar: Path,
    maps: Path,
    in_vcf: Path,
    out_prefix: Path,
    chrom: str,
    threads: int,
    heap_gb: int,
) -> dict:
    out_file = Path(f"{out_prefix}.vcf.gz")
    if out_file.exists():
        return {"chrom": chrom, "status": "cached", "secs": 0.0}
    t0 = time.monotonic()
    ref = _ref_panel(chrom)
    if ref is None:
        return {"chrom": chrom, "status": "skip", "reason": "no reference panel"}
    if not in_vcf.exists():
        return {"chrom": chrom, "status": "skip", "reason": "no input"}
    if chrom == "X":
        in_vcf = _diploidize_x(in_vcf)
    gmap = maps / f"plink.chr{chrom}.GRCh37.map"
    cmd = [
        java,
        f"-Xmx{heap_gb}g",
        "-jar",
        str(jar),
        f"gt={in_vcf}",
        f"ref={ref}",
        f"out={out_prefix}",
        f"chrom={chrom}",
        f"nthreads={threads}",
        "gp=true",
    ]
    if gmap.exists():
        cmd.append(f"map={gmap}")
    log = out_prefix.with_suffix(".log")
    with open(log, "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not out_file.exists():
        return {"chrom": chrom, "status": "error", "rc": rc, "log": str(log)}
    try:
        subprocess.run(
            [find_tool("tabix"), "-f", "-p", "vcf", str(out_file)],
            check=False,
            capture_output=True,
        )
    except FileNotFoundError:
        pass
    return {
        "chrom": chrom,
        "status": "ok",
        "out": str(out_file),
        "secs": time.monotonic() - t0,
    }


# --------------------------------------------------------- rsID annotation
# Beagle drops marker IDs and the 1KG Phase 3 panel VCFs don't carry rsIDs,
# so the imputed output can only be matched by (chrom, pos). That silently
# zeroes out every PGS weight file that has rsIDs but no GRCh37 positions.
# Re-annotate the output from what we do know: the chip's own rsIDs (typed
# sites, via the conformed input VCFs) and the GRCh37-build weight files on
# disk, each of which pairs rsIDs with GRCh37 positions.

_ANNOTATED_MARKER = "##genepred_rsid_annotated=1"


def _grch37_rsid_map(
    in_dir: Path | None, weights_dir: Path | None = None
) -> dict[tuple[str, int], str]:
    """(chrom, pos) -> rsID on GRCh37, from conformed chip input + weight
    files whose native build is GRCh37."""
    from genepred.catalog import list_weight_files, read_header

    pos2rs: dict[tuple[str, int], str] = {}
    try:
        weight_files = list_weight_files(weights_dir)
    except OSError:
        weight_files = []
    for _, wf in weight_files:
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
            if i_chr is None or i_pos is None or i_rs is None:
                continue
            for line in f:
                r = line.rstrip("\n").split("\t")
                if i_rs >= len(r) or not r[i_rs].startswith("rs"):
                    continue
                try:
                    pos2rs[(r[i_chr].lstrip("chr"), int(r[i_pos]))] = r[i_rs]
                except (ValueError, IndexError):
                    continue
    # The chip's own rsIDs are authoritative for typed sites — apply last.
    for vcf in sorted(in_dir.glob("chr*.vcf")) if in_dir else []:
        with open(vcf) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                r = line.split("\t", 4)
                if len(r) > 3 and r[2].startswith("rs"):
                    try:
                        pos2rs[(r[0].lstrip("chr"), int(r[1]))] = r[2]
                    except ValueError:
                        continue
    return pos2rs


def _annotate_chrom(vcf_gz: Path, pos2rs: dict[tuple[str, int], str]) -> int:
    """Fill missing IDs in one imputed chr*.vcf.gz in place (BGZF output).
    Returns the number of sites annotated; -1 if already annotated."""
    out_lines: list[str] = []
    n_named = 0
    with open_maybe_gz(vcf_gz) as f:
        for line in f:
            if line.startswith("#"):
                if line.rstrip("\n") == _ANNOTATED_MARKER:
                    return -1
                if line.startswith("#CHROM"):
                    out_lines.append(_ANNOTATED_MARKER + "\n")
                out_lines.append(line)
                continue
            r = line.split("\t", 3)
            if len(r) > 3 and (not r[2] or r[2] == "."):
                rs = pos2rs.get((r[0].lstrip("chr"), int(r[1])))
                if rs:
                    r[2] = rs
                    n_named += 1
                    line = "\t".join(r)
            out_lines.append(line)
    tmp = vcf_gz.with_suffix(".rsid.tmp")
    bgzf_compress("".join(out_lines).encode(), tmp)
    tmp.rename(vcf_gz)
    # Any tabix index built before the rewrite no longer matches the file.
    Path(str(vcf_gz) + ".tbi").unlink(missing_ok=True)
    return n_named


def annotate_rsids(
    out_dir: Path,
    in_dir: Path | None = None,
    chroms: str = "1-22",
    weights_dir: Path | None = None,
    parallel: int = 4,
) -> int:
    """Backfill rsIDs into Beagle output VCFs under `out_dir`. Idempotent
    (annotated files carry a header marker and are skipped). Returns the
    total number of sites annotated."""
    targets = [
        out_dir / f"chr{c}.vcf.gz"
        for c in parse_chroms(chroms)
        if (out_dir / f"chr{c}.vcf.gz").exists()
    ]
    if not targets:
        return 0
    pos2rs = _grch37_rsid_map(in_dir, weights_dir)
    if not pos2rs:
        return 0
    total = 0
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        for n in ex.map(lambda p: _annotate_chrom(p, pos2rs), targets):
            if n > 0:
                total += n
    return total


def impute(
    genotype_path,
    *,
    name: str | None = None,
    chroms: str = "1-22",
    parallel: int = 8,
    threads_per_chrom: int | None = None,
    heap_gb: int | None = None,
) -> Path:
    """Impute a single genome with Beagle, parallelized by chromosome.

    Returns the output directory containing chr*.vcf.gz.
    """
    if heap_gb is None:
        try:
            avail_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        except (ValueError, OSError):
            avail_gb = 8
        heap_gb = max(2, min(6, int(avail_gb * 0.6 / parallel)))
    genotype_path = Path(genotype_path)
    name = name or genotype_path.stem.split(".")[0]
    out_dir = data_dir() / "imputed" / name
    in_dir = out_dir / "input"
    out_dir.mkdir(parents=True, exist_ok=True)

    chrom_list = parse_chroms(chroms)
    _require_ref_panels(chrom_list)
    if not (in_dir / f"chr{chrom_list[-1]}.vcf").exists():
        print(f"[beagle] conforming {genotype_path} → per-chrom VCFs", file=sys.stderr)
        by_chrom = load_genotype_by_chrom(genotype_path)
        write_conformed_vcf(by_chrom, in_dir, chrom_list)

    jar, maps = setup()
    java = find_tool("java", "jdk-21.0.5+11-jre/bin/java", "jre/bin/java")
    nproc = os.cpu_count() or 8
    if threads_per_chrom is None:
        threads_per_chrom = max(1, nproc // parallel)
    print(
        f"[beagle] {parallel} chroms × {threads_per_chrom} threads "
        f"(java={java}, ref={kg_dir()})",
        file=sys.stderr,
    )

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futs = {
            ex.submit(
                _impute_chrom,
                java,
                jar,
                maps,
                in_dir / f"chr{c}.vcf",
                out_dir / f"chr{c}",
                c,
                threads_per_chrom,
                heap_gb,
            ): c
            for c in chrom_list
        }
        t_start = time.monotonic()
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            tag = "ok" if r["status"] in ("ok", "cached") else r["status"]
            secs = r.get("secs", 0.0)
            extra = (
                f"  {secs:5.1f}s"
                if tag in ("ok", "cached")
                else f"  ({r.get('reason') or r.get('log')})"
            )
            print(f"  chr{r['chrom']:>2}: {tag}{extra}", file=sys.stderr)
        print(
            f"[beagle] {len(results)} chroms in "
            f"{time.monotonic() - t_start:.0f}s wall-clock",
            file=sys.stderr,
        )

    failed = [r for r in results if r["status"] == "error"]
    if failed:
        raise RuntimeError(
            f"{len(failed)} chromosome(s) failed: "
            + ", ".join(f"chr{r['chrom']} (see {r['log']})" for r in failed)
        )

    # Beagle output has no marker IDs (the 1KG panel doesn't carry them);
    # restore rsIDs so rsID-only PGS weight files can still match.
    n_named = annotate_rsids(out_dir, in_dir, chroms, parallel=min(parallel, 4))
    if n_named:
        print(f"[beagle] restored rsIDs at {n_named:,} sites", file=sys.stderr)
    return out_dir


def concat(out_dir: Path, chroms: str = "1-22") -> Path:
    """bcftools concat the per-chromosome outputs into all.vcf.gz."""
    out = out_dir / "all.vcf.gz"
    if out.exists():
        return out
    files = [
        str(out_dir / f"chr{c}.vcf.gz")
        for c in parse_chroms(chroms)
        if (out_dir / f"chr{c}.vcf.gz").exists()
    ]
    bcftools = find_tool("bcftools")
    subprocess.run([bcftools, "concat", "-Oz", "-o", str(out), *files], check=True)
    subprocess.run([find_tool("tabix"), "-p", "vcf", str(out)], check=False)
    return out
