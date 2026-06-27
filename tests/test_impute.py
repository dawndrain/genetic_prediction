"""Offline tests for the Beagle post-processing helpers."""

import gzip

from genepred.impute.beagle import annotate_rsids
from genepred.paths import open_maybe_gz


def _write_weight_file(path):
    rows = [
        "#pgs_id=TEST",
        "#genome_build=GRCh37",
        "rsID\tchr_name\tchr_position\teffect_allele\tother_allele\teffect_weight",
        "rs100\t22\t300\tA\tG\t0.1",
        "rs101\t22\t500\tC\tT\t-0.2",
    ]
    with gzip.open(path, "wt") as f:
        f.write("\n".join(rows) + "\n")


def _write_imputed_vcf(path):
    rows = [
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE",
        "22\t200\t.\tA\tG\t.\tPASS\tDR2=0.99\tGT:DS\t0|1:1.0",
        "22\t300\t.\tA\tG\t.\tPASS\tDR2=0.80;IMP\tGT:DS\t0|0:0.1",
        "22\t400\t.\tC\tT\t.\tPASS\tDR2=0.70;IMP\tGT:DS\t1|1:1.9",
    ]
    with gzip.open(path, "wt") as f:
        f.write("\n".join(rows) + "\n")


def test_annotate_rsids_backfills_ids(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    _write_weight_file(weights / "TEST_hmPOS_GRCh38.txt.gz")

    in_dir = tmp_path / "input"
    in_dir.mkdir()
    (in_dir / "chr22.vcf").write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n"
        "22\t200\trs99\tA\tG\t.\t.\t.\tGT\t0/1\n"
    )

    out_dir = tmp_path / "imputed"
    out_dir.mkdir()
    _write_imputed_vcf(out_dir / "chr22.vcf.gz")

    n = annotate_rsids(out_dir, in_dir, chroms="22", weights_dir=weights)
    assert n == 2  # typed site from chip + rs100 from the weight file

    ids = {}
    with open_maybe_gz(out_dir / "chr22.vcf.gz") as f:
        text = f.read()
    assert "##genepred_rsid_annotated=1" in text
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        c = line.split("\t")
        ids[int(c[1])] = c[2]
    assert ids == {200: "rs99", 300: "rs100", 400: "."}

    # idempotent: a second pass skips the already-annotated file
    assert annotate_rsids(out_dir, in_dir, chroms="22", weights_dir=weights) == 0
