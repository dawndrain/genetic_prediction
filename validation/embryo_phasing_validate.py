"""Real-data validation of the parental phasing-error model.

Picks a 1KG trio whose child is in the `related_samples` release and
whose parents are in the main release, runs SHAPEIT5 with
``--pedigree`` to get a Mendel-consistent child, and counts how often
the child's transmitted haplotype switches between the parent's two
1KG-published haps. Each switch is either a parental phasing switch
error or one of the ~1 meiotic crossovers per chromosome — so this is
a direct measurement of the SER of the 1KG phase that
``genepred embryo-demo`` was treating as ground truth.

Requires ``tools/shapeit5/phase_common_static`` (see docs/PHASING.md). All
other tooling (bgzip, tabix, BCF read) falls back to
``genepred.htslib_lite`` if bcftools/pysam aren't installed.

Reproduces the "Real-data check" table in docs/PHASING.md:

    python validation/embryo_phasing_validate.py --trio PUR --chrom 22
"""

from __future__ import annotations

import argparse
import gzip
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

from genepred import embryo as E
from genepred.htslib_lite import bgzip_tabix_vcf, read_bcf_gt
from genepred.impute import shapeit
from genepred.paths import kg_dir

TRIOS = {
    "PUR": ("HG00733", "HG00731", "HG00732"),
    "YRI": ("NA19240", "NA19239", "NA19238"),
    "KHV": ("HG02024", "HG02026", "HG02025"),
    "CHS": ("HG00702", "HG00656", "HG00657"),
    "MXL": ("NA19675", "NA19679", "NA19678"),
}

REL_VCF_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/"
    "related_samples_vcf/ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_"
    "v5_related_samples.20130502.genotypes.vcf.gz"
)


def fetch_related_vcf(chrom: str) -> Path:
    p = kg_dir() / Path(REL_VCF_URL.format(chrom=chrom)).name
    if not p.exists():
        print(f"[validate] downloading {p.name} …", file=sys.stderr)
        urllib.request.urlretrieve(REL_VCF_URL.format(chrom=chrom), p)
        urllib.request.urlretrieve(REL_VCF_URL.format(chrom=chrom) + ".tbi", str(p) + ".tbi")
    return p


def build_trio_vcf(chrom: str, kid: str, dad: str, mom: str) -> Path:
    """Merge the two parents (main VCF) and the child (related VCF)
    into one 3-sample unphased VCF, then bgzip+tabix it."""
    main = next(kg_dir().glob(f"ALL.chr{chrom}.phase3_*v5b.*.genotypes.vcf.gz"))
    rel = fetch_related_vcf(chrom)
    out_plain = kg_dir() / f"trio_{kid}.chr{chrom}.vcf"
    out = kg_dir() / f"trio_{kid}.chr{chrom}.vcf.gz"
    if out.exists() and Path(str(out) + ".tbi").exists():
        return out

    print(f"[validate] building {out.name} …", file=sys.stderr)
    kid_gt: dict[str, str] = {}
    with gzip.open(rel, "rt") as f:
        kc = -1
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                kc = line.rstrip().split("\t").index(kid)
                continue
            r = line.rstrip().split("\t")
            kid_gt[r[1]] = r[kc]

    with gzip.open(main, "rt") as f, open(out_plain, "w") as o:
        dc = mc = -1
        for line in f:
            if line.startswith("##"):
                o.write(line)
                continue
            if line.startswith("#CHROM"):
                hdr = line.rstrip().split("\t")
                dc, mc = hdr.index(dad), hdr.index(mom)
                o.write("\t".join(hdr[:9] + [dad, mom, kid]) + "\n")
                continue
            r = line.rstrip().split("\t")
            if len(r[3]) != 1 or len(r[4]) != 1 or "," in r[4]:
                continue
            kg = kid_gt.get(r[1])
            if kg is None or len(kg) < 3 or kg[0] == ".":
                continue
            o.write(
                "\t".join(
                    r[:8]
                    + [
                        "GT",
                        r[dc][:3].replace("|", "/"),
                        r[mc][:3].replace("|", "/"),
                        kg[:3].replace("|", "/"),
                    ]
                )
                + "\n"
            )

    bgzip_tabix_vcf(str(out_plain), str(out))
    out_plain.unlink()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trio", choices=list(TRIOS), default="PUR")
    ap.add_argument("--chrom", default="22")
    ap.add_argument(
        "--skip-shapeit",
        action="store_true",
        help="reuse an existing phased BCF instead of re-running SHAPEIT5",
    )
    args = ap.parse_args()

    kid, dad, mom = TRIOS[args.trio]
    print(f"[validate] trio {args.trio}: {kid} ← {dad} × {mom}", file=sys.stderr)

    par = E.load_parents_cached(args.chrom, dad, mom)
    M = len(par.pos)
    pos2i = {int(p): i for i, p in enumerate(par.pos)}

    bcf = kg_dir() / f"trio_{kid}.chr{args.chrom}.phased.bcf"
    if not args.skip_shapeit or not bcf.exists():
        trio_vcf = build_trio_vcf(args.chrom, kid, dad, mom)
        ped = shapeit.write_ped(kg_dir() / f"trio_{kid}.ped", [(kid, dad, mom)])
        ref = next(kg_dir().glob(f"ALL.chr{args.chrom}.phase3_*v5b.*.genotypes.vcf.gz"))
        try:
            shapeit.phase_trio(
                trio_vcf, args.chrom, ped, reference=ref, out_dir=kg_dir(),
            )
            (kg_dir() / f"chr{args.chrom}.phased.bcf").rename(bcf)
        except FileNotFoundError as ex:
            print(f"[validate] SHAPEIT5 not found: {ex}", file=sys.stderr)
            sys.exit(2)
        except subprocess.CalledProcessError as ex:
            print(f"[validate] SHAPEIT5 failed: {ex}", file=sys.stderr)
            sys.exit(1)

    kid_h = np.full((2, M), -1, np.int8)
    g = read_bcf_gt(str(bcf))
    _, samples, _ = next(g)
    kc = samples.index(kid)
    for _, pos, _, _, gts in g:
        i = pos2i.get(pos)
        if i is not None:
            kid_h[0, i], kid_h[1, i], _ = gts[kc]
    ok = kid_h[0] >= 0

    print(
        f"\nReal-data SER of 1KG-published phase, chr{args.chrom}, "
        f"trio {args.trio}\n"
        f"(child's transmitted hap vs parent's two 1KG haps; transitions = "
        f"parental switch errors + ~1 meiotic crossover)\n"
    )
    for name, hap, transmitted in (
        (dad, par.pat, kid_h[0]),
        (mom, par.mat, kid_h[1]),
    ):
        het = (hap[0] != hap[1]) & ok
        n_het = int(het.sum())
        which = (transmitted[het] == hap[1, het]).astype(int)
        n_tr = int((np.diff(which) != 0).sum())
        gaps = np.diff(np.flatnonzero(np.diff(which) != 0)) if n_tr > 1 else np.array([0])
        print(
            f"  {name}: {n_het:,} hets, {n_tr} transitions → "
            f"SER ≈ {(n_tr - 1) / max(n_het, 1):.3%}  "
            f"(median gap {int(np.median(gaps))} hets, mean {int(gaps.mean())})"
        )


if __name__ == "__main__":
    main()
