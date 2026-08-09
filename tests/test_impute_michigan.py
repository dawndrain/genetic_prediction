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


def _write_padded_panel(data_dir):
    """Panel with 8 named samples so prepare() can pad with real columns."""
    kg = data_dir / "1kg"
    kg.mkdir(parents=True)
    panel = kg / "ALL.chr1.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
    names = "\t".join(f"HG{i:05d}" for i in range(8))
    with gzip.open(panel, "wt") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{names}\n")
        f.write(
            "1\t100\trs1\tG\tT\t.\t.\t.\tGT\t"
            "0|0\t0|1\t1|1\t0|0\t0|1\t1|1\t0|0\t0|1\n"
        )
        # multiallelic: samples carrying the dropped alt (code 2) go missing
        f.write(
            "1\t200\trs2\tA\tC,G\t.\t.\t.\tGT\t"
            "0|0\t1|1\t0|2\t0|0\t2|2\t0|1\t0|0\t1|1\n"
        )


def test_prepare_pads_with_real_1kg_samples(tmp_path, monkeypatch):
    """A single-genome upload gets padded with real 1KG sample columns so
    sites carry variation — the servers drop monomorphic sites (issue #5)."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("GENEPRED_DATA", str(data_dir))
    _write_padded_panel(data_dir)

    genome = tmp_path / "genome.txt"
    genome.write_text(
        "# rsid\tchromosome\tposition\tgenotype\n"
        "rs1\t1\t100\tGG\n"  # hom-ref: variation must come from the pads
        "rs2\t1\t200\tAC\n"
    )

    files = michigan.prepare([genome], tmp_path / "out", chroms="1")
    with gzip.open(files[0], "rt") as f:
        lines = [ln.rstrip("\n") for ln in f]
    header = next(ln for ln in lines if ln.startswith("#CHROM"))
    # 4 pads strided across the 8-sample panel: indices 0, 2, 5, 7
    assert header.split("\t")[9:] == ["genome", "HG00000", "HG00002", "HG00005", "HG00007"]
    data = [ln for ln in lines if not ln.startswith("#")]
    assert data == [
        "1\t100\trs1\tG\tT\t.\t.\t.\tGT\t0/0\t0/0\t1/1\t1/1\t0/1",
        # ALT kept is C (code 1); HG00005 is 0|1 -> 0/1, HG00002 carries
        # the dropped G alt (code 2) -> ./.
        "1\t200\trs2\tA\tC\t.\t.\t.\tGT\t0/1\t0/0\t./.\t0/1\t1/1",
    ]


def test_prepare_min_samples_pads_to_topmed_minimum(tmp_path, monkeypatch):
    """TOPMed requires 20 samples (issue #4); prepare honors min_samples,
    topping up with user duplicates when the panel runs out of columns."""
    assert michigan.SERVERS["topmed"]["min_samples"] == 20
    assert michigan.SERVERS["michigan"]["min_samples"] >= 5

    data_dir = tmp_path / "data"
    monkeypatch.setenv("GENEPRED_DATA", str(data_dir))
    _write_padded_panel(data_dir)

    genome = tmp_path / "genome.txt"
    genome.write_text("# rsid\tchromosome\tposition\tgenotype\nrs1\t1\t100\tGT\n")

    files = michigan.prepare([genome], tmp_path / "out", chroms="1", min_samples=20)
    with gzip.open(files[0], "rt") as f:
        lines = [ln.rstrip("\n") for ln in f]
    header = next(ln for ln in lines if ln.startswith("#CHROM"))
    cols = header.split("\t")[9:]
    assert len(cols) == 20  # 1 user + 8 panel samples + 11 duplicates
    assert cols[1:9] == [f"HG{i:05d}" for i in range(8)]
    assert all(c.startswith("genome_dup") for c in cols[9:])
    data = [ln for ln in lines if not ln.startswith("#")]
    assert len(data[0].split("\t")) == 9 + 20
