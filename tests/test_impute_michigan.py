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


def test_prepare_preserves_recoverable_multiallelic_snp(tmp_path, monkeypatch):
    """Preserve multi-allelic SNP sites by casting them to bi-allelic ones
    if the 1KF reference matches the ref,alt combination
    """
    data_dir = tmp_path / "data"
    monkeypatch.setenv("GENEPRED_DATA", str(data_dir))
    kg = data_dir / "1kg"
    kg.mkdir(parents=True)
    panel = kg / "ALL.chr1.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
    with gzip.open(panel, "wt") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        f.write("1\t100\trs1\tG\tC,T\t.\t.\t.\n")
        f.write("1\t200\trs2\tA\tG,AG\t.\t.\t.\n")
        f.write("1\t300\trs3\tA\tC,G\t.\t.\t.\n")
        f.write("1\t400\trs4\tA\tC,TT\t.\t.\t.\n")
        f.write("1\t500\trs5\tAC\tA\t.\t.\t.\n")
        f.write("1\t600\trs6\tG\tA,T\t.\t.\t.\n")

    genome = tmp_path / "genome.txt"
    genome.write_text(
        "# rsid\tchromosome\tposition\tgenotype\n"
        "rs1\t1\t100\tGT\n"  # het for a valid alt
        "rs2\t1\t200\tGG\n"  # hom for the SNP alt, not the indel
        "rs3\t1\t300\tAA\n"  # hom-ref, we keep the first reference alt
        "rs4\t1\t400\tTT\n"  # T is not a valid alt
        "rs5\t1\t500\tAA\n"  # dels are dropped
        "rs6\t1\t600\tAT\n"  # too many alts
    )

    files = michigan.prepare([genome], tmp_path / "out", chroms="1")
    with gzip.open(files[0], "rt") as f:
        data_lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#")]

    assert data_lines == [
        "1\t100\trs1\tG\tT\t.\t.\t.\tGT\t" + "\t".join(["0/1"] * 5),
        "1\t200\trs2\tA\tG\t.\t.\t.\tGT\t" + "\t".join(["1/1"] * 5),
        "1\t300\trs3\tA\tC\t.\t.\t.\tGT\t" + "\t".join(["0/0"] * 5),
    ]
    # position 400 must not appear at all -- "T" isn't a 1KG alternate there
    assert not any(ln.startswith("1\t400\t") for ln in data_lines)


def test_prepare_drops_multiallelic_site_with_conflicting_samples(tmp_path, monkeypatch):
    """Drops multiallelic sites for conflicting sample alts"""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("GENEPRED_DATA", str(data_dir))
    kg = data_dir / "1kg"
    kg.mkdir(parents=True)
    panel = kg / "ALL.chr1.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
    with gzip.open(panel, "wt") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        f.write("1\t100\trs1\tG\tC,T\t.\t.\t.\n")
        f.write("1\t200\trs2\tA\tC,G\t.\t.\t.\n")

    genome_a = tmp_path / "personA.txt"
    genome_a.write_text(
        "# rsid\tchromosome\tposition\tgenotype\n"
        "rs1\t1\t100\tGC\n"
        "rs2\t1\t200\tGA\n"
    )
    genome_b = tmp_path / "personB.txt"
    genome_b.write_text("# rsid\tchromosome\tposition\tgenotype\n"
                        "rs1\t1\t100\tGT\n"  # different alt on the same site
                        "rs2\t1\t200\tAG\n"  # same alt
                        )

    files = michigan.prepare([genome_a, genome_b], tmp_path / "out", chroms="1")
    with gzip.open(files[0], "rt") as f:
        data_lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#")]
    assert data_lines == ["1\t200\trs2\tA\tG\t.\t.\t.\tGT\t" + "\t".join(["1/0\t0/1"] * 2) + "\t1/0"]
