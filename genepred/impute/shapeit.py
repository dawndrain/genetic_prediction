"""Pedigree-aware phasing with SHAPEIT5.

Used here to phase a *parent* (the embryo's mother or father) from
their own parents' genotypes — i.e. trio phasing — which drives the
parental switch-error rate to ≈0 and makes the per-embryo HMM in
``examples/embryo.py`` work without the joint machinery.

SHAPEIT5 applies the Mendelian rule at every het where at least one
grandparent is homozygous (~80 % of hets), then fills the remaining
sites from the population reference panel under the Li-Stephens
model with the trio-resolved sites held fixed as a scaffold. Output
is a phased BCF with the parent's haplotypes ordered by
parent-of-origin.

The binary is not vendored; fetch a static build from
https://odelaneau.github.io/shapeit5/ and drop it under tools/, or
``conda install -c bioconda shapeit5``. The 1KG phase-3 main release
is unrelated-only, so a real-data test additionally needs the
``*_related_samples`` VCFs (download_1kg.sh has a commented stanza).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from genepred.paths import data_dir, find_tool, kg_dir


def _bin() -> str:
    return find_tool(
        "SHAPEIT5_phase_common",
        "phase_common",
        "shapeit5/phase_common_static",
    )


def write_ped(out: Path, trios: list[tuple[str, str, str]]) -> Path:
    """SHAPEIT5's --pedigree format is one line per child with three
    whitespace-separated columns: child father mother (use NA for a
    missing parent). Founders are not listed."""
    with open(out, "w") as f:
        for kid, dad, mom in trios:
            f.write(f"{kid} {dad or 'NA'} {mom or 'NA'}\n")
    return out


def phase_trio(
    input_vcf: Path,
    chrom: str,
    pedigree: Path,
    *,
    reference: Path | None = None,
    genetic_map: Path | None = None,
    out_dir: Path | None = None,
    threads: int = 8,
) -> Path:
    """Run SHAPEIT5 phase_common with --pedigree on one chromosome.

    `input_vcf` must contain the child *and* both grandparents (and
    may contain anything else). Returns the path to the phased BCF."""
    out_dir = out_dir or data_dir() / "shapeit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"chr{chrom}.phased.bcf"
    cmd = [
        _bin(),
        "--input", str(input_vcf),
        "--pedigree", str(pedigree),
        "--region", str(chrom),
        "--output", str(out),
        "--thread", str(threads),
    ]
    if reference is not None:
        cmd += ["--reference", str(reference)]
    if genetic_map is not None:
        cmd += ["--map", str(genetic_map)]
    print(f"[shapeit] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)
    return out


def kg_related_vcf(chrom: str) -> Path:
    """Path to the 1KG related-samples VCF for this chromosome
    (contains the trio children that the main release omits)."""
    p = (
        kg_dir()
        / f"ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5_related_samples"
        f".20130502.genotypes.vcf.gz"
    )
    if not p.exists():
        raise FileNotFoundError(
            f"{p}\n"
            f"The 1KG phase-3 main release is unrelated-only. Trio "
            f"members live in the related-samples VCFs:\n"
            f"  https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
            f"supporting/related_samples_vcf/\n"
            f"Uncomment the related-samples block in "
            f"reference/onekg/download_1kg.sh, or fetch chr{chrom} directly."
        )
    return p
