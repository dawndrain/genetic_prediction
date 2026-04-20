"""Genotype I/O — loading DTC files and VCFs, conforming to 1KG.

A "genotype" here is a pair of dicts:
    by_rs:  rsid -> (allele1, allele2)
    by_pos: (chrom_str, pos_int) -> (allele1, allele2)
where chrom_str has no "chr" prefix and alleles are single-letter ACGT.
Indels and no-calls are dropped at load time.
"""

from __future__ import annotations

import gzip
import struct
import zlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from genepred.paths import kg_dir, open_maybe_gz

COMPLEMENT = str.maketrans("ACGT", "TGCA")
Genotype = tuple[dict[str, tuple[str, str]], dict[tuple[str, int], tuple[str, str]]]


def chrom_sort_key(chrom: str):
    try:
        return (0, int(chrom))
    except ValueError:
        order = {"X": 1, "Y": 2, "MT": 3}
        return (1, order.get(chrom, 99))


# ---------------------------------------------------------------- loading


def load_genotypes(path) -> Genotype:
    """Auto-detect VCF vs 23andMe-style flat text and load to (by_rs, by_pos)."""
    with open_maybe_gz(path) as f:
        for line in f:
            s = line.lstrip()
            if s.startswith("##fileformat=VCF") or s.startswith("#CHROM"):
                return _load_vcf(path)
            if s and not s.startswith("#"):
                break
    by_rs, by_pos = {}, {}
    with open_maybe_gz(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            line = line.rstrip("\r\n")
            parts = line.split("\t")
            if len(parts) < 4:
                parts = line.split(",")
            if len(parts) < 4:
                parts = line.split()
            if len(parts) < 4 or parts[0] in ("rsid", "rsID"):
                continue
            rsid, chrom, pos, gt = parts[0], parts[1].lstrip("chr"), parts[2], parts[3]
            if len(parts) >= 5 and len(parts[3]) == 1 and len(parts[4]) == 1:
                gt = parts[3] + parts[4]
            if len(gt) != 2 or any(c not in "ACGT" for c in gt):
                continue
            g = (gt[0], gt[1])
            if rsid.startswith("rs"):
                by_rs[rsid] = g
            try:
                by_pos[(chrom, int(pos))] = g
            except ValueError:
                pass
    return by_rs, by_pos


def _load_vcf(path) -> Genotype:
    """Single-sample VCF → (by_rs, by_pos). Uses the first sample column.

    For hard calls (GT) the value is the allele 2-tuple (a1, a2).
    When a continuous dosage (DS) field is present — typical for
    imputed VCFs — the value is a 3-tuple (ref, alt, ds_float) so the
    fractional dosage isn't lost. score_one() handles both forms;
    other consumers that only expect 2-tuples should treat len==3 as
    (round-DS) hard call."""
    by_rs, by_pos = {}, {}
    with open_maybe_gz(path) as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                hdr = line.rstrip().split("\t")
                if len(hdr) < 10:
                    raise ValueError("VCF has no sample columns")
                continue
            r = line.rstrip("\r\n").split("\t")
            if len(r) < 10:
                continue
            chrom, pos, rsid, ref, alt = (r[0].lstrip("chr"), r[1], r[2], r[3], r[4])
            if len(ref) != 1 or len(alt) != 1 or "," in alt:
                continue
            fmt = r[8].split(":")
            samp = r[9].split(":")
            ds = None
            if "DS" in fmt:
                try:
                    ds = float(samp[fmt.index("DS")])
                except (ValueError, IndexError):
                    ds = None
            if "GT" in fmt:
                gt = samp[fmt.index("GT")]
                if len(gt) < 3 or gt[0] not in "01" or gt[2] not in "01":
                    if ds is None:
                        continue
                    g: tuple = (ref, alt, ds)
                else:
                    a1 = alt if gt[0] == "1" else ref
                    a2 = alt if gt[2] == "1" else ref
                    g = (ref, alt, ds) if ds is not None else (a1, a2)
            elif ds is not None:
                g = (ref, alt, ds)
            else:
                continue
            if rsid and rsid != "." and rsid.startswith("rs"):
                by_rs[rsid] = g
            try:
                by_pos[(chrom, int(pos))] = g
            except ValueError:
                pass
    return by_rs, by_pos


def load_genotype_by_chrom(path) -> dict[str, dict[int, tuple[str, str, str]]]:
    """chrom -> {pos: (rsid, a1, a2)} — used for per-chromosome VCF emission."""
    out: dict[str, dict] = defaultdict(dict)
    with open_maybe_gz(path) as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            p = line.split("\t")
            if len(p) < 4:
                p = line.split(",")
            if len(p) < 4:
                p = line.split()
            if len(p) < 4 or p[0] in ("rsid", "rsID"):
                continue
            rsid, chrom, pos, gt = p[0], p[1].lstrip("chr"), p[2], p[3]
            if len(p) >= 5 and len(p[3]) == 1 and len(p[4]) == 1:
                gt = p[3] + p[4]
            if len(gt) != 2 or any(c not in "ACGT" for c in gt):
                continue
            try:
                out[chrom][int(pos)] = (rsid, gt[0], gt[1])
            except ValueError:
                pass
    return out


# -------------------------------------------------- conformed VCF emission


def conform_chrom(chrom: str, sites: dict, kg_vcf: Path, out_fh):
    """Stream the 1KG VCF for one chromosome; for each input site that
    matches a 1KG biallelic SNP, emit a VCF row with REF/ALT from 1KG and
    the genotype oriented accordingly. Returns (n_in, n_ok, n_mismatch,
    n_not_in_ref).

    The conformance rate this reports is essentially the fraction of sites
    Beagle/minimac will accept, since both drop sites whose REF/ALT don't
    match the panel.
    """
    n_in = len(sites)
    n_ok = n_mismatch = 0
    seen = set()
    with gzip.open(kg_vcf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            t1 = line.find("\t")
            t2 = line.find("\t", t1 + 1)
            try:
                pos = int(line[t1 + 1 : t2])
            except ValueError:
                continue
            hit = sites.get(pos)
            if hit is None:
                continue
            seen.add(pos)
            row = line.split("\t", 8)
            ref, alt = row[3], row[4]
            if len(ref) != 1 or len(alt) != 1 or "," in alt:
                n_mismatch += 1
                continue
            rsid, a1, a2 = hit
            if {a1, a2} <= {ref, alt}:
                gt = f"{int(a1 == alt)}/{int(a2 == alt)}"
                out_fh.write(
                    f"{chrom}\t{pos}\t{rsid}\t{ref}\t{alt}\t.\t.\t.\tGT\t{gt}\n"
                )
                n_ok += 1
            else:
                n_mismatch += 1
    return n_in, n_ok, n_mismatch, n_in - len(seen)


def write_conformed_vcf(
    by_chrom: dict,
    out_dir: Path,
    chroms: Iterable[str] | None = None,
    panel_dir: Path | None = None,
) -> list[Path]:
    """Emit per-chromosome VCFs conformed to 1KG REF/ALT under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = panel_dir or kg_dir()
    chroms = list(chroms or [str(c) for c in range(1, 23)])
    written = []
    for chrom in chroms:
        vcf = next(panel_dir.glob(f"ALL.chr{chrom}.*.vcf.gz"), None)
        if vcf is None or chrom not in by_chrom:
            continue
        out_path = out_dir / f"chr{chrom}.vcf"
        with open(out_path, "w") as out_fh:
            out_fh.write("##fileformat=VCFv4.2\n")
            out_fh.write("##source=genepred (REF/ALT from 1KG Phase3)\n")
            out_fh.write(
                '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            )
            out_fh.write(
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n"
            )
            conform_chrom(chrom, by_chrom[chrom], vcf, out_fh)
        written.append(out_path)
    return written


def parse_chroms(spec: str) -> list[str]:
    """'22' → ['22']; '1-22' → ['1',...,'22']; '1,5,22' → ['1','5','22']."""
    if "-" in spec:
        a, b = spec.split("-")
        return [str(c) for c in range(int(a), int(b) + 1)]
    return spec.split(",")


# Pure-Python BGZF (bgzip) writer — fallback when the htslib `bgzip`
# binary isn't on PATH. BGZF is concatenated gzip blocks ≤64KB with a
# 'BC' extra subfield holding the compressed block size (SAMv1 §4.1).
# Tabix indexing still requires the binary; this just produces a file
# Michigan/Beagle will accept.
_BGZF_EOF = bytes.fromhex(
    "1f8b08040000000000ff0600424302001b0003000000000000000000"
)


def bgzf_compress(data: bytes, out_path: Path, block_size: int = 65280) -> Path:
    with open(out_path, "wb") as o:
        for i in range(0, len(data), block_size):
            chunk = data[i : i + block_size]
            comp = zlib.compress(chunk, 6)[2:-4]  # raw deflate
            bsize = 12 + 6 + len(comp) + 8 - 1
            o.write(
                b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff"
                + struct.pack("<H", 6)
                + b"BC"
                + struct.pack("<HH", 2, bsize)
                + comp
                + struct.pack("<II", zlib.crc32(chunk) & 0xFFFFFFFF, len(chunk))
            )
        o.write(_BGZF_EOF)
    return out_path
