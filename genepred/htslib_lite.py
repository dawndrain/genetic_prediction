"""Pure-Python fallbacks for the htslib bits we need when
bcftools/tabix/pysam aren't installed.

Three pieces, each the minimum to feed a single-chromosome VCF to
SHAPEIT5 and read its BCF output back:

  bgzf_compress(data)       one BGZF block
  bgzip_tabix_vcf(in, out)  bgzip a single-chrom VCF and write a TBI
  read_bcf_gt(path)         iterate (chrom, pos, ref, alt, [(a0,a1,phased)…])

Use real htslib (bcftools/tabix or pysam) when available — these
exist so the embryo demos run on a bare Python install.
"""

from __future__ import annotations

import gzip
import struct
import zlib
from collections.abc import Iterator

_BGZF_EOF = bytes.fromhex(
    "1f8b08040000000000ff0600424302001b0003000000000000000000"
)
_TBI_SHIFT = 14  # 16 kb linear-index tiles


def bgzf_compress(data: bytes) -> bytes:
    """One BGZF block. len(data) must be ≤ 64 KiB."""
    co = zlib.compressobj(6, zlib.DEFLATED, -15)
    comp = co.compress(data) + co.flush()
    bsize = len(comp) + 25
    assert bsize <= 0xFFFF, "BGZF block exceeds 64 KiB"
    hdr = (
        b"\x1f\x8b\x08\x04" + b"\x00" * 4 + b"\x00\xff"
        + struct.pack("<H", 6) + b"BC" + struct.pack("<HH", 2, bsize)
    )
    return (
        hdr + comp
        + struct.pack("<II", zlib.crc32(data) & 0xFFFFFFFF, len(data) & 0xFFFFFFFF)
    )


def _bgzf_multi(data: bytes, blocksize: int = 60000) -> bytes:
    return b"".join(
        bgzf_compress(data[i : i + blocksize]) for i in range(0, len(data), blocksize)
    )


def bgzip_tabix_vcf(inp: str, outp: str, blocksize: int = 60000) -> None:
    """bgzip a plain-text single-chromosome VCF and write a minimal TBI.

    The index has one top-level bin covering the whole file plus a
    16 kb linear index — sufficient for region queries on one
    chromosome and accepted by SHAPEIT5/htslib."""
    linear: list[int] = []
    chrom: str | None = None
    coffset = 0
    first_voff: int | None = None
    last_voff = 0

    with open(inp, "rb") as f, open(outp, "wb") as o:
        buf, eof = b"", False
        while not eof or buf:
            if not eof:
                chunk = f.read(1 << 20)
                if not chunk:
                    eof = True
                buf += chunk
            cut = buf.rfind(b"\n") + 1 if not eof else len(buf)
            data, buf = buf[:cut], buf[cut:]
            i = 0
            while i < len(data):
                seg = data[i : i + blocksize]
                bnl = seg.rfind(b"\n")
                seg = data[i : i + bnl + 1] if bnl >= 0 else seg
                i += len(seg)
                voff_block = coffset << 16
                u = 0
                for line in seg.split(b"\n")[:-1]:
                    voff = voff_block | u
                    u += len(line) + 1
                    if line.startswith(b"#"):
                        continue
                    cols = line.split(b"\t", 3)
                    if chrom is None:
                        chrom = cols[0].decode()
                        first_voff = voff
                    tile = (int(cols[1]) - 1) >> _TBI_SHIFT
                    while len(linear) <= tile:
                        linear.append(voff)
                last_voff = voff_block | u
                blk = bgzf_compress(seg)
                o.write(blk)
                coffset += len(blk)
        o.write(_BGZF_EOF)

    assert chrom is not None and first_voff is not None
    name = chrom.encode() + b"\x00"
    tbi = bytearray(b"TBI\x01")
    tbi += struct.pack("<7i", 1, 2, 1, 2, 0, ord("#"), 0)
    tbi += struct.pack("<i", len(name)) + name
    tbi += struct.pack("<i", 1)  # n_bin
    tbi += struct.pack("<Ii", 0, 1) + struct.pack("<QQ", first_voff, last_voff)
    tbi += struct.pack("<i", len(linear))
    for ioff in linear:
        tbi += struct.pack("<Q", ioff)
    with open(outp + ".tbi", "wb") as t:
        t.write(_bgzf_multi(bytes(tbi)) + _BGZF_EOF)


# ---------------------------------------------------------------- BCF


def _read_typed(buf: memoryview, off: int):
    tb = buf[off]
    off += 1
    n, t = tb >> 4, tb & 0x0F
    if n == 15:
        nv, off = _read_typed(buf, off)
        n = nv[0] if isinstance(nv, list) else nv
    if t == 0:
        return None, off
    sz = {1: 1, 2: 2, 3: 4, 5: 4, 7: 1}[t]
    raw = bytes(buf[off : off + n * sz])
    off += n * sz
    if t == 7:
        return raw.decode("ascii"), off
    fmt = {1: "b", 2: "h", 3: "i", 5: "f"}[t]
    return list(struct.unpack(f"<{n}{fmt}", raw)), off


def read_bcf_gt(
    path: str,
) -> Iterator[tuple[int, int, str, str | None, list[tuple[int, int, bool]]]]:
    """Yield (chrom_idx, pos, ref, alt, [(a0, a1, phased) per sample])
    from a BCF v2.2 file. The first record yielded is the special
    tuple ("__header__", samples, contigs)."""
    f = gzip.open(path, "rb")
    magic = f.read(5)
    assert magic[:3] == b"BCF", f"not a BCF: {magic!r}"
    (l_text,) = struct.unpack("<I", f.read(4))
    header = f.read(l_text).decode("ascii")
    samples: list[str] = []
    contigs: list[str] = []
    for line in header.splitlines():
        if line.startswith("#CHROM"):
            samples = line.split("\t")[9:]
        elif line.startswith("##contig=<ID="):
            contigs.append(line.split("ID=", 1)[1].split(",")[0].rstrip(">"))
    yield ("__header__", samples, contigs)  # type: ignore[misc]

    while True:
        h = f.read(8)
        if len(h) < 8:
            return
        l_shared, l_indiv = struct.unpack("<II", h)
        shared = memoryview(f.read(l_shared))
        indiv = memoryview(f.read(l_indiv))
        chrom_i, pos0, _ = struct.unpack_from("<iii", shared, 0)
        _, n_allele = struct.unpack_from("<HH", shared, 16)
        ns_nf = struct.unpack_from("<I", shared, 20)[0]
        n_sample = ns_nf & 0xFFFFFF
        off = 24
        _, off = _read_typed(shared, off)  # ID
        alleles: list[str] = []
        for _ in range(n_allele):
            a, off = _read_typed(shared, off)
            alleles.append(a)
        # FORMAT/GT is the first field in the indiv block
        ioff = 0
        _, ioff = _read_typed(indiv, ioff)
        tb = indiv[ioff]
        ioff += 1
        n, t = tb >> 4, tb & 0x0F
        sz = {1: 1, 2: 2, 3: 4}[t]
        fc = "b" if t == 1 else "h" if t == 2 else "i"
        gts: list[tuple[int, int, bool]] = []
        for s in range(n_sample):
            raw = struct.unpack_from(f"<{n}{fc}", indiv, ioff + s * n * sz)
            a0 = (raw[0] >> 1) - 1 if raw[0] > 0 else -1
            a1 = (raw[1] >> 1) - 1 if len(raw) > 1 and raw[1] > 0 else -1
            phased = bool(len(raw) > 1 and (raw[1] & 1))
            gts.append((a0, a1, phased))
        yield (chrom_i, pos0 + 1, alleles[0], alleles[1] if n_allele > 1 else None, gts)
