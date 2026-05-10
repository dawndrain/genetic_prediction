"""CPIC Level-A pharmacogenomics — drug-metabolism star alleles.

Authoritative sources:
  - CPIC guidelines: https://cpicpgx.org/guidelines/
  - PharmGKB allele definitions: https://www.pharmgkb.org/
  - PharmVar (CYP nomenclature): https://www.pharmvar.org/

This module covers the ~12 highest-impact genes with CPIC Level-A
("actionable — change prescribing") evidence. Each gene's common
star alleles are defined by their tag SNPs; ``call_diplotype`` maps
an individual's genotypes to a diplotype and metabolizer phenotype.

Caveats — these need dedicated assays, not just SNPs:
  - *CYP2D6* has whole-gene deletions (*5) and duplications (*1xN)
    that arrays miss; the diplotype call here is incomplete without
    a CNV assay.
  - Full HLA typing (4-digit resolution) needs sequence or
    imputation (HIBAG, HLA*LA); only the *57:01 / *15:02 tag SNPs
    are included here as a screen.
  - *MT-RNR1* is mitochondrial — homoplasmy/heteroplasmy matters.

For anything beyond a first-pass report, use a clinical PGx panel
(GeneSight, OneOme RightMed, Invitae PGx) or PharmCAT
(https://pharmcat.org/), which implements the full PharmVar
definitions including structural variants.
"""

from __future__ import annotations

from dataclasses import dataclass

# (rsid, chrom, pos_grch37 | None, ref, alt) — alt is the star-allele-defining allele.
SNP = tuple[str, str, int | None, str, str]


@dataclass(frozen=True)
class PgxGene:
    gene: str
    star_alleles: dict[str, list[SNP]]
    """*1 is the reference (no defining SNPs). Other alleles are
    called when *all* their defining SNPs are present on a haplotype;
    here we approximate with unphased genotypes (any-copy)."""
    function: dict[str, str]
    """Star allele → activity ('normal', 'decreased', 'none',
    'increased', 'uncertain')."""
    phenotype_map: dict[tuple[str, str], str]
    """(func_a, func_b) sorted → metabolizer phenotype. Falls back to
    DEFAULT_PHENOTYPE_MAP if a pair is missing."""
    drugs: list[str]
    cpic_url: str
    notes: str = ""
    default_phenotype_map: bool = True


DEFAULT_PHENOTYPE_MAP: dict[tuple[str, str], str] = {
    ("none", "none"): "Poor metabolizer",
    ("decreased", "none"): "Poor metabolizer",
    ("decreased", "decreased"): "Intermediate metabolizer",
    ("none", "normal"): "Intermediate metabolizer",
    ("decreased", "normal"): "Intermediate metabolizer",
    ("normal", "normal"): "Normal metabolizer",
    ("increased", "normal"): "Rapid metabolizer",
    ("increased", "increased"): "Ultrarapid metabolizer",
    ("increased", "none"): "Normal metabolizer",
    ("decreased", "increased"): "Normal metabolizer",
}


CPIC_GENES: list[PgxGene] = [
    PgxGene(
        gene="CYP2C19",
        star_alleles={
            "*1": [],
            "*2": [("rs4244285", "10", 96541616, "G", "A")],
            "*3": [("rs4986893", "10", 96540410, "G", "A")],
            "*17": [("rs12248560", "10", 96521657, "C", "T")],
        },
        function={"*1": "normal", "*2": "none", "*3": "none", "*17": "increased"},
        phenotype_map={},
        drugs=[
            "clopidogrel", "voriconazole", "citalopram", "escitalopram",
            "sertraline", "amitriptyline", "omeprazole", "pantoprazole",
        ],
        cpic_url="https://cpicpgx.org/guidelines/guideline-for-clopidogrel-and-cyp2c19/",
    ),
    PgxGene(
        gene="CYP2D6",
        star_alleles={
            "*1": [],
            "*2": [("rs16947", "22", 42523943, "G", "A"),
                   ("rs1135840", "22", 42522613, "G", "C")],
            "*4": [("rs3892097", "22", 42524947, "C", "T")],
            "*10": [("rs1065852", "22", 42526694, "G", "A")],
            "*41": [("rs28371725", "22", 42523805, "C", "T")],
            # *5 (whole-gene deletion) and *1xN/*2xN (duplications) need a CNV assay.
        },
        function={
            "*1": "normal", "*2": "normal", "*4": "none",
            "*10": "decreased", "*41": "decreased",
        },
        phenotype_map={},
        drugs=[
            "codeine", "tramadol", "tamoxifen", "atomoxetine",
            "ondansetron", "paroxetine", "fluoxetine", "venlafaxine",
            "risperidone", "aripiprazole",
        ],
        cpic_url="https://cpicpgx.org/guidelines/guideline-for-codeine-and-cyp2d6/",
        notes="CNV (*5 deletion, *1xN/*2xN duplications) is common "
        "(~5–10%) and not callable from SNPs; activity-score "
        "phenotype here is incomplete without it.",
    ),
    PgxGene(
        gene="CYP2C9",
        star_alleles={
            "*1": [],
            "*2": [("rs1799853", "10", 96702047, "C", "T")],
            "*3": [("rs1057910", "10", 96741053, "A", "C")],
        },
        function={"*1": "normal", "*2": "decreased", "*3": "none"},
        phenotype_map={},
        drugs=["warfarin", "phenytoin", "celecoxib", "siponimod"],
        cpic_url="https://cpicpgx.org/guidelines/guideline-for-warfarin-and-cyp2c9-vkorc1/",
    ),
    PgxGene(
        gene="VKORC1",
        star_alleles={
            "ref": [],
            "-1639A": [("rs9923231", "16", 31107689, "C", "T")],
        },
        function={"ref": "normal", "-1639A": "decreased"},
        phenotype_map={
            ("normal", "normal"): "Normal warfarin sensitivity",
            ("decreased", "normal"): "Increased warfarin sensitivity",
            ("decreased", "decreased"): "High warfarin sensitivity",
        },
        drugs=["warfarin"],
        cpic_url="https://cpicpgx.org/guidelines/guideline-for-warfarin-and-cyp2c9-vkorc1/",
        default_phenotype_map=False,
    ),
    PgxGene(
        gene="TPMT",
        star_alleles={
            "*1": [],
            "*2": [("rs1800462", "6", 18143955, "G", "C")],
            "*3B": [("rs1800460", "6", 18139228, "C", "T")],
            "*3C": [("rs1142345", "6", 18130918, "T", "C")],
            # *3A = *3B + *3C in cis; resolved by call_diplotype if both present.
        },
        function={"*1": "normal", "*2": "none", "*3A": "none",
                  "*3B": "none", "*3C": "none"},
        phenotype_map={},
        drugs=["azathioprine", "mercaptopurine", "thioguanine"],
        cpic_url="https://cpicpgx.org/guidelines/guideline-for-thiopurines-and-tpmt/",
    ),
    PgxGene(
        gene="NUDT15",
        star_alleles={
            "*1": [],
            "*3": [("rs116855232", "13", 48611918, "C", "T")],
        },
        function={"*1": "normal", "*3": "none"},
        phenotype_map={},
        drugs=["azathioprine", "mercaptopurine", "thioguanine"],
        cpic_url="https://cpicpgx.org/guidelines/guideline-for-thiopurines-and-tpmt/",
        notes="*3 MAF ~10% in EAS, rare in EUR/AFR.",
    ),
    PgxGene(
        gene="DPYD",
        star_alleles={
            "*1": [],
            "*2A": [("rs3918290", "1", 97915614, "C", "T")],
            "*13": [("rs55886062", "1", 97981343, "T", "G")],
            "c.2846A>T": [("rs67376798", "1", 97547947, "T", "A")],
            "HapB3": [("rs56038477", "1", 98039419, "C", "T")],
        },
        function={
            "*1": "normal", "*2A": "none", "*13": "none",
            "c.2846A>T": "decreased", "HapB3": "decreased",
        },
        phenotype_map={},
        drugs=["fluorouracil", "capecitabine"],
        cpic_url="https://cpicpgx.org/guidelines/guideline-for-fluoropyrimidines-and-dpyd/",
    ),
    PgxGene(
        gene="SLCO1B1",
        star_alleles={
            "*1": [],
            "*5": [("rs4149056", "12", 21331549, "T", "C")],
        },
        function={"*1": "normal", "*5": "decreased"},
        phenotype_map={
            ("normal", "normal"): "Normal function",
            ("decreased", "normal"): "Decreased function",
            ("decreased", "decreased"): "Poor function",
        },
        drugs=["simvastatin", "atorvastatin", "rosuvastatin"],
        cpic_url="https://cpicpgx.org/guidelines/cpic-guideline-for-statins/",
        default_phenotype_map=False,
        notes="*15 = *5 + *1B (rs2306283); functionally equivalent to *5.",
    ),
    PgxGene(
        gene="UGT1A1",
        star_alleles={
            "*1": [],
            "*28": [("rs3064744", "2", 234668879, "(TA)6", "(TA)7")],
            "*6": [("rs4148323", "2", 234669144, "G", "A")],
        },
        function={"*1": "normal", "*28": "decreased", "*6": "decreased"},
        phenotype_map={},
        drugs=["irinotecan", "atazanavir", "belinostat"],
        cpic_url="https://cpicpgx.org/guidelines/guideline-for-atazanavir-and-ugt1a1/",
        notes="*28 is a TA-repeat promoter variant; some arrays use "
        "rs8175347/rs887829 (C>T) as proxy. *6 MAF ~15% in EAS.",
    ),
    PgxGene(
        gene="G6PD",
        star_alleles={
            "B (wt)": [],
            "A-": [("rs1050828", "X", 153762634, "C", "T"),
                   ("rs1050829", "X", 153763492, "T", "C")],
            "Med": [("rs5030868", "X", 153762340, "C", "T")],
        },
        function={"B (wt)": "normal", "A-": "decreased", "Med": "none"},
        phenotype_map={
            ("normal", "normal"): "Normal",
            ("decreased", "normal"): "Variable (carrier female)",
            ("decreased", "decreased"): "Deficient",
            ("none", "normal"): "Variable (carrier female)",
            ("none", "none"): "Deficient",
            ("decreased", "none"): "Deficient",
        },
        drugs=["rasburicase", "primaquine", "tafenoquine", "dapsone",
               "pegloticase"],
        cpic_url="https://cpicpgx.org/guidelines/cpic-guideline-for-rasburicase-and-g6pd/",
        default_phenotype_map=False,
        notes="X-linked — males hemizygous. >180 named variants "
        "(WHO classes); only the two commonest are listed here.",
    ),
    PgxGene(
        gene="HLA-B",
        star_alleles={
            "neg": [],
            "*57:01": [("rs2395029", "6", 31431780, "T", "G")],
            "*15:02": [("rs3909184", "6", None, "G", "C")],  # TODO: confirm best tag
        },
        function={"neg": "normal", "*57:01": "risk", "*15:02": "risk"},
        phenotype_map={
            ("normal", "normal"): "Negative",
            ("normal", "risk"): "Positive (carrier)",
            ("risk", "risk"): "Positive (carrier)",
        },
        drugs=["abacavir (*57:01)", "carbamazepine/phenytoin (*15:02)",
               "allopurinol (*58:01 — not tagged here)"],
        cpic_url="https://cpicpgx.org/guidelines/guideline-for-abacavir-and-hla-b/",
        default_phenotype_map=False,
        notes="Tag-SNP screening only — rs2395029 (HCP5) tags *57:01 at "
        "r²≈1 in EUR; *15:02 tagging is population-dependent. Confirm "
        "by sequence-based HLA typing before withholding a drug.",
    ),
    PgxGene(
        gene="MT-RNR1",
        star_alleles={
            "wt": [],
            "1555A>G": [("rs267606617", "MT", 1555, "A", "G")],
            "1494C>T": [("rs267606619", "MT", 1494, "C", "T")],
        },
        function={"wt": "normal", "1555A>G": "risk", "1494C>T": "risk"},
        phenotype_map={
            ("normal", "normal"): "Normal risk",
            ("normal", "risk"): "Increased aminoglycoside ototoxicity risk",
            ("risk", "risk"): "Increased aminoglycoside ototoxicity risk",
        },
        drugs=["gentamicin", "tobramycin", "amikacin", "streptomycin"],
        cpic_url="https://cpicpgx.org/guidelines/cpic-guideline-for-aminoglycosides-and-mt-rnr1/",
        default_phenotype_map=False,
        notes="Mitochondrial — effectively haploid; heteroplasmy level "
        "modulates penetrance.",
    ),
]

CPIC_BY_GENE: dict[str, PgxGene] = {g.gene: g for g in CPIC_GENES}


# ----------------------------------------------------------------- calling


def _has_allele(genotypes: dict[str, tuple[str, str]], snps: list[SNP]) -> int:
    """How many copies (0/1/2) of this star allele the unphased
    genotypes support — the min over defining SNPs of the count of
    the defining allele."""
    if not snps:
        return 0
    counts: list[int] = []
    for rsid, _, _, _, alt in snps:
        gt = genotypes.get(rsid)
        if gt is None:
            return 0
        counts.append(sum(1 for a in gt if a == alt))
    return min(counts)


def call_diplotype(
    gene: PgxGene, genotypes: dict[str, tuple[str, str]]
) -> tuple[str, str]:
    """Best-effort diplotype call from unphased genotypes.

    Counts copies of each non-reference star allele, fills the
    remainder with the reference allele, and maps to a metabolizer
    phenotype via the gene's (or the default) function table. This
    is a screening-grade call — phase, structural variants, and
    rare alleles are not handled. Returns (diplotype, phenotype)."""
    ref = next(iter(gene.star_alleles))
    haps: list[str] = []
    for star, snps in gene.star_alleles.items():
        if star == ref:
            continue
        n = _has_allele(genotypes, snps)
        haps.extend([star] * min(n, 2))
    # TPMT *3A = *3B + *3C in cis; collapse if both present.
    if gene.gene == "TPMT" and "*3B" in haps and "*3C" in haps:
        haps.remove("*3B")
        haps.remove("*3C")
        haps.append("*3A")
    haps = haps[:2]
    while len(haps) < 2:
        haps.append(ref)
    haps.sort()
    diplo = f"{haps[0]}/{haps[1]}"

    fa = gene.function.get(haps[0], "uncertain")
    fb = gene.function.get(haps[1], "uncertain")
    key = tuple(sorted((fa, fb)))
    pmap = gene.phenotype_map if not gene.default_phenotype_map else {
        **DEFAULT_PHENOTYPE_MAP, **gene.phenotype_map,
    }
    phenotype = pmap.get(key, "Indeterminate")  # type: ignore[arg-type]
    return diplo, phenotype


def report_pgx(genotypes: dict[str, tuple[str, str]]) -> str:
    """Run all CPIC genes against an rsid → (allele1, allele2)
    genotype map and emit a text table."""
    rows = []
    for g in CPIC_GENES:
        diplo, pheno = call_diplotype(g, genotypes)
        drugs = ", ".join(g.drugs[:4]) + (" …" if len(g.drugs) > 4 else "")
        rows.append((g.gene, diplo, pheno, drugs, g.notes))
    w_gene = max(8, max(len(r[0]) for r in rows))
    w_dip = max(10, max(len(r[1]) for r in rows))
    w_phen = max(20, max(len(r[2]) for r in rows))
    out = [
        "CPIC Level-A pharmacogenomics (screening-grade — see module "
        "docstring for caveats)",
        "",
        f"{'gene':<{w_gene}}  {'diplotype':<{w_dip}}  "
        f"{'phenotype':<{w_phen}}  affected drugs",
        "-" * (w_gene + w_dip + w_phen + 30),
    ]
    for gene, diplo, pheno, drugs, notes in rows:
        out.append(
            f"{gene:<{w_gene}}  {diplo:<{w_dip}}  {pheno:<{w_phen}}  {drugs}"
        )
        if notes:
            out.append(f"{'':<{w_gene}}  {'':>{w_dip}}  ↳ {notes}")
    out.append("")
    out.append("Sources: cpicpgx.org/guidelines · pharmgkb.org · pharmvar.org")
    return "\n".join(out)
