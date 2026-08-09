import gzip
import shutil

import pytest

from genepred.impute import beagle
from genepred.io import bgzf_compress

needs_htslib = pytest.mark.skipif(
    shutil.which("bcftools") is None or shutil.which("tabix") is None,
    reason="bcftools/tabix not installed",
)

_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=1>\n"
    "##contig=<ID=2>\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
)


def _write_chrom_vcf(out_dir, chrom):
    body = _HEADER + f"{chrom}\t100\trs{chrom}\tG\tA\t.\t.\t.\tGT\t0/1\n"
    bgzf_compress(body.encode(), out_dir / f"chr{chrom}.vcf.gz")


@needs_htslib
def test_concat_indexes_and_joins(tmp_path):
    """concat() must index the per-chrom files (rsID annotation drops the
    .tbi), join them, and index the result (issue #3)."""
    _write_chrom_vcf(tmp_path, "1")
    _write_chrom_vcf(tmp_path, "2")

    out = beagle.concat(tmp_path, chroms="1-2")

    assert out == tmp_path / "all.vcf.gz"
    assert (tmp_path / "all.vcf.gz.tbi").exists()
    assert (tmp_path / "chr1.vcf.gz.tbi").exists()
    with gzip.open(out, "rt") as f:
        data = [ln for ln in f if not ln.startswith("#")]
    assert [ln.split("\t")[0] for ln in data] == ["1", "2"]


@needs_htslib
def test_concat_is_cached_but_not_stale(tmp_path):
    _write_chrom_vcf(tmp_path, "1")
    out = beagle.concat(tmp_path, chroms="1")
    mtime = out.stat().st_mtime
    assert beagle.concat(tmp_path, chroms="1").stat().st_mtime == mtime

    # rewriting a per-chrom input (as rsID annotation does) must retrigger
    import os
    import time

    time.sleep(0.01)
    _write_chrom_vcf(tmp_path, "1")
    os.utime(tmp_path / "chr1.vcf.gz")
    beagle.concat(tmp_path, chroms="1")
    assert out.stat().st_mtime >= (tmp_path / "chr1.vcf.gz").stat().st_mtime


def test_concat_without_inputs_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        beagle.concat(tmp_path, chroms="1-22")
