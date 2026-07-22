import gzip
import textwrap
import pytest

from genepred.impute import michigan


def _write_23andme(path):
    path.write_text(
        textwrap.dedent(
            """
            # rsid\tchromosome\tposition\tgenotype
            rs2\t22\t2000\tCC
            rs1\t22\t1000\tTA
            """
        ).lstrip()
    )


def test_prepare_raises_without_1kg_panel(tmp_path, monkeypatch):
    monkeypatch.setenv("GENEPRED_DATA", str(tmp_path / "data"))
    genome = tmp_path / "genome.txt"
    _write_23andme(genome)

    with pytest.raises(FileNotFoundError, match="1KG reference panel"):
        michigan.prepare([genome], tmp_path / "out", chroms="22")


def test_prepare_succeeds_when_panel_present_no_duplicates(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("GENEPRED_DATA", str(data_dir))
    kg = data_dir / "1kg"
    kg.mkdir(parents=True)
    panel = kg / "ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
    with gzip.open(panel, "wt") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        f.write("22\t1000\trs1\tA\tT\t.\t.\t.\n")
        f.write("22\t2000\trs2\tC\tG\t.\t.\t.\n")
        f.write("22\t2000\trs2\tC\tA\t.\t.\t.\n")

    genome = tmp_path / "genome.txt"
    _write_23andme(genome)

    files = michigan.prepare([genome], tmp_path / "out", chroms="22")
    assert [f.name for f in files] == ["chr22.vcf.gz"]
    with gzip.open(files[0], "rt") as f:
        lines = [ln.rstrip("\n") for ln in f]
    header = next(ln for ln in lines if ln.startswith("#CHROM"))
    assert header.split("\t")[9:] == ["genome", "genome_dup2", "genome_dup3", "genome_dup4", "genome_dup5",]
    data_lines = [ln for ln in lines if not ln.startswith("#")]
    assert data_lines == [
        "22\t1000\trs1\tA\tT\t.\t.\t.\tGT\t1/0\t1/0\t1/0\t1/0\t1/0",
        "22\t2000\trs2\tC\tG\t.\t.\t.\tGT\t0/0\t0/0\t0/0\t0/0\t0/0",
    ]
