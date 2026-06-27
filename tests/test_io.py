import textwrap
from pathlib import Path

from genepred import io


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content).lstrip())
    return p


def test_load_23andme_text(tmp_path):
    p = _write(
        tmp_path,
        "g.txt",
        """
        # 23andMe raw data
        # rsid\tchromosome\tposition\tgenotype
        rs1\t1\t1000\tAG
        rs2\t1\t2000\tCC
        rs3\t1\t3000\t--
        i999\t1\t4000\tTT
        rs4\tX\t5000\tA
        """,
    )
    by_rs, by_pos = io.load_genotypes(p)
    assert by_rs["rs1"] == ("A", "G")
    assert by_rs["rs2"] == ("C", "C")
    assert "rs3" not in by_rs  # no-call dropped
    assert by_rs["rs4"] == ("A", "A")  # hemizygous X/MT kept as homozygous
    assert ("1", 4000) in by_pos  # i-id keeps position
    assert "i999" not in by_rs


def test_load_ancestrydna_csv(tmp_path):
    p = _write(
        tmp_path,
        "g.csv",
        """
        rsid,chromosome,position,allele1,allele2
        rs10,1,1000,A,G
        rs11,1,2000,T,T
        """,
    )
    by_rs, _ = io.load_genotypes(p)
    assert by_rs["rs10"] == ("A", "G")
    assert by_rs["rs11"] == ("T", "T")


def test_load_vcf(tmp_path):
    p = _write(
        tmp_path,
        "g.vcf",
        """
        ##fileformat=VCFv4.2
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
        1\t1000\trs20\tA\tG\t.\t.\t.\tGT\t0/1
        1\t2000\trs21\tC\tT\t.\t.\t.\tGT\t1|1
        1\t3000\trs22\tA\tAT\t.\t.\t.\tGT\t0/1
        """,
    )
    by_rs, by_pos = io.load_genotypes(p)
    assert by_rs["rs20"] == ("A", "G")
    assert by_rs["rs21"] == ("T", "T")
    assert "rs22" not in by_rs  # indel dropped
    assert ("1", 1000) in by_pos


def test_crlf_line_endings(tmp_path):
    p = tmp_path / "g.txt"
    p.write_bytes(b"# hdr\r\nrs30\t1\t100\tAC\r\nrs31\t1\t200\tGG\r\n")
    by_rs, _ = io.load_genotypes(p)
    assert by_rs["rs30"] == ("A", "C")
    assert by_rs["rs31"] == ("G", "G")


def test_hemizygous_single_letter_calls_load(tmp_path):
    # Male X / MT calls in 23andMe exports are single letters; they should
    # load as homozygous so G6PD / MT-RNR1 screening sees them.
    f = tmp_path / "dtc.txt"
    f.write_text(
        "# comment\n"
        "rsid\tchromosome\tposition\tgenotype\n"
        "rs1050828\tX\t153762634\tT\n"
        "rs267606617\tMT\t1555\tA\n"
        "rs4244285\t10\t96541616\tGA\n"
        "rs9999999\t1\t1000\t--\n"
    )
    from genepred.io import load_genotypes

    by_rs, by_pos = load_genotypes(f)
    assert by_rs["rs1050828"] == ("T", "T")
    assert by_rs["rs267606617"] == ("A", "A")
    assert by_rs["rs4244285"] == ("G", "A")
    assert "rs9999999" not in by_rs
