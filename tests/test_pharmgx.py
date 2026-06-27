"""Offline tests for pharmgx + rare_variants.

The GOLDEN_ALLELES table records GRCh37 plus-strand (ref, alt) verified
against the 1000 Genomes Phase 3 VCFs — guarding against the cDNA-strand /
wrong-position entries that previously made every genome a TPMT poor
metabolizer and a TYK2 P1104A carrier.
"""

from genepred.io import hard_call
from genepred.pharmgx import (
    CPIC_BY_GENE,
    CPIC_GENES,
    NOT_ASSAYED,
    call_diplotype,
    report_pgx,
    untyped_alleles,
)
from genepred.rare_variants import LPA_RISK, PROTECTIVE, check_genome

# rsid -> (GRCh37 ref, alt) as in 1KG Phase 3; alt is the variant-defining allele.
GOLDEN_ALLELES = {
    # CPIC tag SNPs
    "rs4244285": ("G", "A"),    # CYP2C19*2
    "rs12248560": ("C", "T"),   # CYP2C19*17
    "rs1799853": ("C", "T"),    # CYP2C9*2
    "rs1057910": ("A", "C"),    # CYP2C9*3
    "rs9923231": ("C", "T"),    # VKORC1 -1639A
    "rs1800462": ("C", "G"),    # TPMT*2  (was listed in cDNA orientation)
    "rs116855232": ("C", "T"),  # NUDT15*3
    "rs3918290": ("C", "T"),    # DPYD*2A
    "rs55886062": ("A", "C"),   # DPYD*13 (was listed in cDNA orientation)
    "rs67376798": ("T", "A"),   # DPYD c.2846A>T
    "rs56038477": ("C", "T"),   # DPYD HapB3
    "rs4149056": ("T", "C"),    # SLCO1B1*5
    "rs4148323": ("G", "A"),    # UGT1A1*6
    "rs2395029": ("T", "G"),    # HLA-B*57:01 tag
    # curated rare / protective / LPA variants
    "rs11591147": ("G", "T"),   # PCSK9 R46L
    "rs116843064": ("G", "A"),  # ANGPTL4 E40K
    "rs11209026": ("G", "A"),   # IL23R R381Q
    "rs35667974": ("T", "C"),   # IFIH1 I923V (was at the wrong position)
    "rs34536443": ("G", "C"),   # TYK2 P1104A (was listed in cDNA orientation)
    "rs601338": ("G", "A"),     # FUT2 W154X
    "rs7412": ("C", "T"),       # APOE e2
    "rs9536314": ("T", "G"),    # KL-VS
    "rs10455872": ("A", "G"),   # LPA
    "rs3798220": ("T", "C"),    # LPA I4399M
}
GOLDEN_POSITIONS = {
    "rs35667974": ("2", 163124637),
    "rs116855232": ("13", 48619855),
}


def _all_table_snps():
    for g in CPIC_GENES:
        for snps in g.star_alleles.values():
            for rsid, chrom, pos, ref, alt in snps:
                yield rsid, chrom, pos, ref, alt
    for v in PROTECTIVE + LPA_RISK:
        if v.rsid and v.ref and v.alt:
            yield v.rsid, v.chrom, v.pos_grch37, v.ref, v.alt


def test_tables_match_grch37_plus_strand():
    seen = set()
    for rsid, chrom, pos, ref, alt in _all_table_snps():
        if rsid in GOLDEN_ALLELES:
            seen.add(rsid)
            assert (ref, alt) == GOLDEN_ALLELES[rsid], (
                f"{rsid}: table {ref}>{alt}, GRCh37 plus strand is "
                f"{GOLDEN_ALLELES[rsid][0]}>{GOLDEN_ALLELES[rsid][1]}"
            )
        if rsid in GOLDEN_POSITIONS:
            assert (chrom, pos) == GOLDEN_POSITIONS[rsid], (
                f"{rsid}: table at {chrom}:{pos}, expected "
                f"{GOLDEN_POSITIONS[rsid][0]}:{GOLDEN_POSITIONS[rsid][1]}"
            )
    assert len(seen) >= 20  # the golden table actually covers the modules


def _ref_genotypes():
    return {rsid: (ref, ref) for rsid, _, _, ref, _ in _all_table_snps()}


def test_reference_genome_is_unremarkable():
    """Hom-ref at every site -> normal diplotypes everywhere, no rare hits.
    (Catches major/minor swaps: a swapped entry makes 'reference' people
    carriers.)"""
    geno = _ref_genotypes()
    for g in CPIC_GENES:
        diplo, pheno = call_diplotype(g, geno)
        assert pheno.startswith(("Normal", "Negative")), (
            f"{g.gene}: hom-ref genome called {diplo} / {pheno}"
        )
    by_pos = {
        (chrom, pos): (ref, ref)
        for _, chrom, pos, ref, _ in _all_table_snps()
        if pos is not None
    }
    assert check_genome(by_pos, PROTECTIVE + LPA_RISK) == []


def test_het_genome_calls_variant_alleles():
    geno = {rsid: (ref, alt) for rsid, _, _, ref, alt in _all_table_snps()}
    # CYP2C19 *2/*17 specifically — CPIC classifies it as IM, not normal
    diplo, pheno = call_diplotype(
        CPIC_BY_GENE["CYP2C19"],
        {"rs4244285": ("G", "A"), "rs12248560": ("C", "T")},
    )
    assert set(diplo.split("/")) == {"*17", "*2"}
    assert pheno == "Intermediate metabolizer"
    diplo, pheno = call_diplotype(CPIC_BY_GENE["SLCO1B1"], geno)
    assert diplo == "*1/*5" and pheno == "Decreased function"
    by_pos = {
        (chrom, pos): (ref, alt)
        for _, chrom, pos, ref, alt in _all_table_snps()
        if pos is not None
    }
    hits = {v.gene: n for v, n in check_genome(by_pos, PROTECTIVE + LPA_RISK)}
    assert hits.get("PCSK9") == 1 and hits.get("TYK2") == 1


def test_cyp2d6_decreased_plus_none_is_intermediate():
    # *4/*41 — activity score 0.5 — CPIC calls this Intermediate, not Poor.
    geno = {"rs3892097": ("C", "T"), "rs28371725": ("C", "T")}
    diplo, pheno = call_diplotype(CPIC_BY_GENE["CYP2D6"], geno)
    assert set(diplo.split("/")) == {"*4", "*41"}
    assert pheno == "Intermediate metabolizer"


def test_dosage_tuples_are_hard_called():
    assert hard_call(("G", "A", 0.04)) == ("G", "G")
    assert hard_call(("G", "A", 1.2)) == ("G", "A")
    assert hard_call(("G", "A", 1.8)) == ("A", "A")
    # near-zero dosage must not create a *2 call or a rare-variant hit
    geno = {"rs4244285": ("G", "A", 0.02), "rs12248560": ("C", "C")}
    diplo, _ = call_diplotype(CPIC_BY_GENE["CYP2C19"], geno)
    assert diplo == "*1/*1"
    assert check_genome({("1", 55505647): ("G", "T", 0.05)}, PROTECTIVE) == []
    assert [
        (v.gene, n)
        for v, n in check_genome({("1", 55505647): ("G", "T", 0.95)}, PROTECTIVE)
    ] == [("PCSK9", 1)]


def test_not_assayed_reporting():
    assert call_diplotype(CPIC_BY_GENE["G6PD"], {}) == ("n/a", NOT_ASSAYED)
    # partial coverage: present tags are used, absent ones are listed
    geno = {"rs4244285": ("G", "G")}
    assert untyped_alleles(CPIC_BY_GENE["CYP2C19"], geno) == ["*3", "*17"]
    report = report_pgx(geno)
    assert NOT_ASSAYED in report                # fully untyped genes say so
    assert "Not assayed on this input" in report  # partially-typed note
