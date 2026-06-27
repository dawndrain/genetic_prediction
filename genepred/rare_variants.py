"""Curated high-impact rare variants that polygenic scores can't see.

Two halves:

  PROTECTIVE — a hand-curated list of ~30 large-effect protective
      alleles (mostly LoF) discovered in family studies or biobank
      pLoF burden scans. There is no authoritative registry for
      these; this list is best-effort and will drift. For a
      reproducible systematic source, filter genebass.org (UKB 450k
      exomes) to pLoF burden / protective direction / p < 5e-8.

  LPA_RISK — the two common *LPA* tag SNPs for high Lp(a), a
      ~90 %-heritable causal CHD/aortic-stenosis risk factor that
      standard lipid panels and most CHD PGS miss. The KIV2 CNV is
      the underlying causal element and needs sequencing/qPCR.

  PATHOGENIC_GENES — the ACMG SF v3.2 secondary-findings gene list
      (~81 genes; Miller et al., Genet Med 2023). This is *not* a
      variant list — each gene has many P/LP alleles. Use
      ``clinvar_lookup()`` below for per-variant calls, filtered to
      review_status ≥ "criteria provided, multiple submitters" and
      clinical_significance ∈ {Pathogenic, Likely pathogenic}.

  CARRIER_GENES — the high-frequency subset (~110 genes) of the
      ACMG/ACOG tier-3 expanded carrier-screening panel (Gregg et al.
      2021). Recessive; only relevant when *both* parents carry P/LP
      in the same gene (or the mother for X-linked). Full commercial
      panels run to ~500 genes.

Pharmacogenomics (CPIC Level-A) is in the sibling module
``genepred.pharmgx``.

Effect sizes and `qaly_estimate` are population-relative comparisons
of het carriers vs non-carriers — valid for the embryo-selection use
case (one parent carries it, half the embryos inherit) without
extrapolation. None where the literature doesn't support a number.

Use with the embryo-imputation machinery in genepred.embryo:

    carried = check_parents(par, PROTECTIVE)
    _, _, _, wp, wm = joint_recover(par, biopsies, ...)
    per_embryo = flag_embryos(carried, wp, wm, par)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from genepred.embryo import Parents
from genepred.io import hard_call
from genepred.paths import data_dir


@dataclass(frozen=True)
class RareVariant:
    gene: str
    rsid: str | None
    hgvs: str | None
    chrom: str
    pos_grch37: int | None
    ref: str | None
    alt: str | None
    direction: str  # "protective" | "pathogenic" | "risk"
    trait: str
    effect_description: str
    evidence_tier: int  # 1=drug-validated/ACMG, 2=well-replicated, 3=thin
    source: str
    qaly_estimate: float | None = None
    notes: str = ""


# ----------------------------------------------------------------- protective

# GRCh37 coordinates are filled where the canonical rsID is unambiguous;
# entries with pos_grch37=None need lookup before check_parents() can
# match them — TODO: resolve via dbSNP/Ensembl REST.
PROTECTIVE: list[RareVariant] = [
    # --- lipid / cardio LoF (tier 1: drug-validated) ---------------------
    RareVariant(
        "PCSK9", "rs11591147", "p.R46L", "1", 55505647, "G", "T",
        "protective", "coronary_heart_disease",
        "LDL ↓~15%, CHD ↓~30% per allele", 1,
        "Cohen 2006 NEJM; basis of evolocumab/alirocumab",
        qaly_estimate=0.30,
        notes="MAF ~1.5% EUR; the rarer Y142X (rs67608943) and "
        "C679X (rs28362286, AFR ~1%) are full LoF with larger effect.",
    ),
    RareVariant(
        "APOC3", "rs76353203", "p.R19X", "11", 116701353, "C", "T",
        "protective", "coronary_heart_disease",
        "TG ↓~40%, CHD ↓~40%", 1,
        "Jørgensen 2014 NEJM; TG-lowering ASOs (volanesorsen)",
        qaly_estimate=0.30,
        notes="Other APOC3 LoF (IVS2+1G>A rs138326449, A43T) similar.",
    ),
    RareVariant(
        "ANGPTL3", None, "LoF (multiple)", "1", None, None, None,
        "protective", "coronary_heart_disease",
        "LDL+TG ↓, CHD ↓~40%", 1,
        "Dewey 2017 NEJM; basis of evinacumab",
        qaly_estimate=0.25,
    ),
    RareVariant(
        "ANGPTL4", "rs116843064", "p.E40K", "19", 8429323, "G", "A",
        "protective", "coronary_heart_disease",
        "TG ↓, CHD ↓~50% (hom)", 1,
        "Dewey 2016 NEJM; Stitziel 2016",
        qaly_estimate=0.15,
        notes="MAF ~2% EUR. Hom carriers in mice show lymphadenopathy.",
    ),
    RareVariant(
        "ASGR1", "rs186021206", "del12", "17", None, None, None,
        "protective", "coronary_heart_disease",
        "non-HDL ↓, CHD ↓~34%", 1,
        "Nioi 2016 NEJM (deCODE)",
        qaly_estimate=0.20,
        notes="12bp intronic deletion → frameshift; ~1% Iceland.",
    ),
    RareVariant(
        "NPC1L1", None, "LoF (multiple)", "7", None, None, None,
        "protective", "coronary_heart_disease",
        "LDL ↓~12 mg/dL, CHD ↓~50%", 1,
        "Stitziel 2014 NEJM; ezetimibe target",
        qaly_estimate=0.20,
    ),
    RareVariant(
        "CETP", None, "LoF (multiple)", "16", None, None, None,
        "protective", "coronary_heart_disease",
        "HDL ↑; longevity association in Ashkenazi centenarians", 2,
        "Barzilai 2003 JAMA; CETP-inhibitor trials mostly negative",
        qaly_estimate=None,
        notes="HDL-raising per se may not be causal for CHD.",
    ),
    # --- metabolic --------------------------------------------------------
    RareVariant(
        "SLC30A8", None, "LoF (multiple)", "8", None, None, None,
        "protective", "type2_diabetes",
        "T2D risk ↓~65%", 1,
        "Flannick 2014 Nat Genet",
        qaly_estimate=0.30,
    ),
    RareVariant(
        "GPR75", None, "LoF (multiple)", "2", None, None, None,
        "protective", "obesity",
        "BMI ↓~1.8 kg/m², obesity OR ~0.5", 1,
        "Akbari 2021 Science (Regeneron/UKB 640k exomes)",
        qaly_estimate=0.15,
    ),
    RareVariant(
        "HSD17B13", "rs72613567", "splice (T>TA)", "4", 88231392, "T", "TA",
        "protective", "liver_disease",
        "NAFLD/NASH/cirrhosis ↓~30–70%", 1,
        "Abul-Husn 2018 NEJM (Regeneron); Alnylam ALN-HSD",
        qaly_estimate=0.10,
        notes="MAF ~25% EUR — common enough that some PGS partly capture it.",
    ),
    # --- autoimmune / inflammatory ---------------------------------------
    RareVariant(
        "IL23R", "rs11209026", "p.R381Q", "1", 67705958, "G", "A",
        "protective", "inflammatory_bowel_disease",
        "Crohn's OR ~0.3, psoriasis/AS protection", 2,
        "Duerr 2006 Science; IL-23 mAbs (risankizumab)",
        qaly_estimate=0.05,
        notes="MAF ~6% EUR.",
    ),
    RareVariant(
        "IFIH1", "rs35667974", "p.I923V", "2", 163124637, "T", "C",
        "protective", "type1_diabetes",
        "T1D OR ~0.5; also psoriasis protection", 2,
        "Nejentsev 2009 Science",
        qaly_estimate=0.05,
        notes="Other IFIH1 rare LoF similar (rs35337543 etc.).",
    ),
    RareVariant(
        "TYK2", "rs34536443", "p.P1104A", "19", 10463118, "G", "C",
        "protective", "autoimmune",
        "RA/SLE/T1D/IBD/MS protection (OR ~0.6–0.8)", 2,
        "Dendrou 2016 Sci Transl Med; deucravacitinib",
        qaly_estimate=0.05,
        notes="MAF ~4% EUR. Trade-off: TB susceptibility (Boisson-Dupuis 2018).",
    ),
    RareVariant(
        "RNF186", "rs41264113", "p.R179X", "1", 20140391, "G", "A",
        "protective", "inflammatory_bowel_disease",
        "UC OR ~0.3", 2,
        "Rivas 2016 Nat Commun",
        qaly_estimate=0.03,
    ),
    RareVariant(
        "CARD9", "rs141992399", "splice c.IVS11+1G>C", "9", None, None, None,
        "protective", "inflammatory_bowel_disease",
        "IBD OR ~0.3", 2,
        "Rivas 2011 Nat Genet; Beaudoin 2013",
        qaly_estimate=0.03,
        notes="Trade-off: fungal-infection susceptibility when biallelic.",
    ),
    # --- short sleep ------------------------------------------------------
    RareVariant(
        "BHLHE41", "rs121912617", "p.P384R", "12", 26272829, "G", "C",
        "protective", "sleep",
        "Familial natural short sleep, ~4–6 h need", 2,
        "He 2009 Science",
        qaly_estimate=None,
        notes="DEC2. Single family + mouse knock-in.",
    ),
    RareVariant(
        "ADRB1", None, "p.A187V", "10", None, None, None,
        "protective", "sleep",
        "Familial natural short sleep", 2,
        "Shi 2019 Neuron",
        qaly_estimate=None,
        notes="Single family + mouse model.",
    ),
    RareVariant(
        "NPSR1", None, "p.Y206H", "7", None, None, None,
        "protective", "sleep",
        "Short sleep + resistance to memory deficit", 3,
        "Xing 2019 Sci Transl Med",
        qaly_estimate=None,
        notes="Distinct from common rs324981 (N107I).",
    ),
    # --- musculoskeletal --------------------------------------------------
    RareVariant(
        "MSTN", None, "LoF (multiple)", "2", None, None, None,
        "protective", "muscle_mass",
        "Muscle hypertrophy (myostatin null)", 3,
        "Schuelke 2004 NEJM (single child, hom)",
        qaly_estimate=None,
        notes="Tendon fragility in animal models.",
    ),
    RareVariant(
        "LRP5", None, "p.G171V (and other HBM GoF)", "11", None, None, None,
        "protective", "bone_density",
        "High bone mass, fracture resistance", 2,
        "Boyden 2002 NEJM; Little 2002",
        qaly_estimate=0.05,
        notes="Romosozumab targets the same WNT axis.",
    ),
    # --- pain / anxiety ---------------------------------------------------
    RareVariant(
        "FAAH-OUT", None, "~8 kb microdeletion + FAAH rs324420", "1", None, None, None,
        "protective", "pain",
        "Hypoalgesia, low anxiety, fast wound healing", 3,
        "Habib 2019 Br J Anaesth (n=1, Jo Cameron)",
        qaly_estimate=None,
        notes="rs324420 (P129T) alone is common with modest effect.",
    ),
    # --- infection resistance --------------------------------------------
    RareVariant(
        "CCR5", "rs333", "Δ32 (32 bp del)", "3", 46414944, None, None,
        "protective", "hiv",
        "HIV-1 R5-tropic resistance (hom)", 2,
        "Liu 1996 Cell; Samson 1996",
        qaly_estimate=None,
        notes="Het: slower progression. Possible WNV/flu susceptibility.",
    ),
    RareVariant(
        "FUT2", "rs601338", "p.W154X", "19", 49206674, "G", "A",
        "protective", "infection",
        "Norovirus GII.4 + some rotavirus resistance (hom non-secretor)", 2,
        "Lindesmith 2003 Nat Med",
        qaly_estimate=None,
        notes="MAF ~45% EUR — common; on most arrays.",
    ),
    # --- neuroprotection / longevity -------------------------------------
    RareVariant(
        "APP", "rs63750847", "p.A673T", "21", 27269932, "C", "T",
        "protective", "alzheimers",
        "AD OR ~0.2; also age-related cognitive decline ↓", 1,
        "Jonsson 2012 Nature (deCODE Iceland)",
        qaly_estimate=0.30,
        notes="MAF ~0.5% Iceland, ~0.02% elsewhere. Reduces β-amyloidogenic "
        "cleavage ~40% — direct mechanistic validation of the amyloid "
        "hypothesis. On Church's protective-allele list.",
    ),
    RareVariant(
        "APOE", "rs7412", "ε2 (p.R176C)", "19", 45412079, "C", "T",
        "protective", "alzheimers",
        "Alzheimer's OR ~0.6 vs ε3", 2,
        "Corder 1994; covered by AD PGS",
        qaly_estimate=0.10,
        notes="MAF ~8%. Slight ↑ type-III hyperlipoproteinaemia risk.",
    ),
    RareVariant(
        "KL", "rs9536314", "KL-VS (p.F352V)", "13", 33628138, "T", "G",
        "protective", "cognition",
        "Het: longevity + reported cognition benefit; hom: detrimental", 3,
        "Dubal 2014 Cell Rep; replication mixed",
        qaly_estimate=None,
        notes="MAF ~16%. Heterozygote-advantage pattern.",
    ),
    RareVariant(
        "SERPINE1", "rs6092", "PAI-1 null (frameshift)", "7", None, None, None,
        "protective", "longevity",
        "Lower PAI-1; longevity in Berne Amish", 3,
        "Khan 2017 Sci Adv (n≈177 carriers)",
        qaly_estimate=None,
        notes="rsID here is a tag SNP, not the null itself.",
    ),
    RareVariant(
        "EPOR", None, "p.W439X (and other C-terminal truncations)", "19", None, None, None,
        "protective", "endurance",
        "Primary familial erythrocytosis; ↑Hct, ↑VO₂max", 3,
        "de la Chapelle 1993 PNAS (Mäntyranta family)",
        qaly_estimate=None,
        notes="Family-private; thrombosis risk plausible.",
    ),
]


# -------------------------------------------------------------------- Lp(a)
#
# Lp(a) is ~90 % heritable and a causal, independent CHD/AVS risk factor
# that standard lipid panels and most CHD PGS miss. The underlying causal
# element is the KIV2 copy-number repeat in *LPA* (fewer repeats → higher
# Lp(a)); the two SNPs below jointly tag ~40 % of high-Lp(a) variance and
# are on most arrays. Direct KIV2 assay needs sequencing or qPCR.
LPA_RISK: list[RareVariant] = [
    RareVariant(
        "LPA", "rs10455872", None, "6", 161010118, "A", "G",
        "risk", "coronary_heart_disease",
        "Tags low-KIV2 / high-Lp(a); CHD OR ~1.5–1.7 per allele, "
        "aortic stenosis OR ~1.6", 1,
        "Clarke 2009 NEJM; Kamstrup 2009 JAMA; Thanassoulis 2013 NEJM",
        qaly_estimate=-0.15,
        notes="MAF ~7% EUR. Basis of pelacarsen / olpasiran (Lp(a)-ASOs).",
    ),
    RareVariant(
        "LPA", "rs3798220", "p.I4399M", "6", 160961137, "T", "C",
        "risk", "coronary_heart_disease",
        "Tags high-Lp(a); CHD OR ~1.9 per allele", 1,
        "Clarke 2009 NEJM",
        qaly_estimate=-0.15,
        notes="MAF ~2% EUR. Largely independent of rs10455872; together "
        "they explain ~36–40% of Lp(a) variance — the rest is the "
        "untagged KIV2 CNV.",
    ),
]


# -------------------------------------------------------------- pathogenic

# ACMG SF v3.2 (Miller et al., Genet Med 2023; PMID 37347242). This is the
# *gene* list — each has many P/LP alleles. For variant-level calls, query
# ClinVar at runtime; do not hardcode variants.
_ACMG_SF_V3_2: list[tuple[str, str, str]] = [
    # Hereditary cancer
    ("APC", "Familial adenomatous polyposis", "AD"),
    ("BMPR1A", "Juvenile polyposis", "AD"),
    ("BRCA1", "Hereditary breast/ovarian cancer", "AD"),
    ("BRCA2", "Hereditary breast/ovarian cancer", "AD"),
    ("MAX", "Hereditary paraganglioma-pheochromocytoma", "AD"),
    ("MEN1", "Multiple endocrine neoplasia 1", "AD"),
    ("MLH1", "Lynch syndrome", "AD"),
    ("MSH2", "Lynch syndrome", "AD"),
    ("MSH6", "Lynch syndrome", "AD"),
    ("MUTYH", "MUTYH-associated polyposis", "AR"),
    ("NF2", "Neurofibromatosis 2", "AD"),
    ("PALB2", "Hereditary breast cancer", "AD"),
    ("PMS2", "Lynch syndrome", "AD"),
    ("PTEN", "PTEN hamartoma tumour syndrome", "AD"),
    ("RB1", "Retinoblastoma", "AD"),
    ("RET", "MEN2 / FMTC", "AD"),
    ("SDHAF2", "Hereditary PGL/PCC", "AD"),
    ("SDHB", "Hereditary PGL/PCC", "AD"),
    ("SDHC", "Hereditary PGL/PCC", "AD"),
    ("SDHD", "Hereditary PGL/PCC", "AD"),
    ("SMAD4", "Juvenile polyposis / HHT", "AD"),
    ("STK11", "Peutz-Jeghers syndrome", "AD"),
    ("TMEM127", "Hereditary PGL/PCC", "AD"),
    ("TP53", "Li-Fraumeni syndrome", "AD"),
    ("TSC1", "Tuberous sclerosis", "AD"),
    ("TSC2", "Tuberous sclerosis", "AD"),
    ("VHL", "Von Hippel-Lindau", "AD"),
    ("WT1", "WT1-related Wilms tumour", "AD"),
    # Cardiovascular — cardiomyopathy
    ("ACTC1", "Cardiomyopathy (HCM/DCM)", "AD"),
    ("BAG3", "Dilated cardiomyopathy", "AD"),
    ("DES", "Cardiomyopathy", "AD"),
    ("DSC2", "ARVC", "AD"),
    ("DSG2", "ARVC", "AD"),
    ("DSP", "ARVC / DCM", "AD"),
    ("FLNC", "Cardiomyopathy", "AD"),
    ("LMNA", "DCM / laminopathy", "AD"),
    ("MYBPC3", "Hypertrophic cardiomyopathy", "AD"),
    ("MYH7", "HCM / DCM", "AD"),
    ("MYL2", "HCM", "AD"),
    ("MYL3", "HCM", "AD"),
    ("PKP2", "ARVC", "AD"),
    ("PRKAG2", "HCM / WPW", "AD"),
    ("RBM20", "DCM", "AD"),
    ("TMEM43", "ARVC", "AD"),
    ("TNNI3", "HCM / DCM", "AD"),
    ("TNNT2", "HCM / DCM", "AD"),
    ("TPM1", "HCM / DCM", "AD"),
    ("TTN", "DCM (truncating only)", "AD"),
    # Cardiovascular — arrhythmia
    ("CALM1", "CPVT / LQTS", "AD"),
    ("CALM2", "CPVT / LQTS", "AD"),
    ("CALM3", "CPVT / LQTS", "AD"),
    ("CASQ2", "CPVT", "AR"),
    ("KCNH2", "Long-QT 2", "AD"),
    ("KCNQ1", "Long-QT 1", "AD"),
    ("RYR2", "CPVT", "AD"),
    ("SCN5A", "LQT3 / Brugada", "AD"),
    ("TRDN", "CPVT / LQTS", "AR"),
    # Cardiovascular — aortopathy / vascular
    ("ACTA2", "Familial thoracic aortic aneurysm", "AD"),
    ("ACVRL1", "Hereditary haemorrhagic telangiectasia", "AD"),
    ("COL3A1", "Vascular Ehlers-Danlos", "AD"),
    ("ENG", "Hereditary haemorrhagic telangiectasia", "AD"),
    ("FBN1", "Marfan syndrome", "AD"),
    ("MYH11", "FTAAD", "AD"),
    ("SMAD3", "Loeys-Dietz", "AD"),
    ("TGFBR1", "Loeys-Dietz", "AD"),
    ("TGFBR2", "Loeys-Dietz", "AD"),
    # Cardiovascular — lipid
    ("APOB", "Familial hypercholesterolaemia", "AD"),
    ("LDLR", "Familial hypercholesterolaemia", "AD"),
    ("PCSK9", "FH (gain-of-function only)", "AD"),
    # Metabolic / inborn errors
    ("BTD", "Biotinidase deficiency", "AR"),
    ("GAA", "Pompe disease", "AR"),
    ("GLA", "Fabry disease", "XL"),
    ("HFE", "Hereditary haemochromatosis (p.C282Y hom)", "AR"),
    ("OTC", "Ornithine transcarbamylase deficiency", "XL"),
    ("TTR", "Hereditary transthyretin amyloidosis", "AD"),
    # Other
    ("ATP7B", "Wilson disease", "AR"),
    ("CACNA1S", "Malignant hyperthermia susceptibility", "AD"),
    ("HNF1A", "MODY3", "AD"),
    ("RPE65", "Leber congenital amaurosis / RP", "AR"),
    ("RYR1", "Malignant hyperthermia susceptibility", "AD"),
]

PATHOGENIC_GENES: dict[str, dict[str, str]] = {
    g: {"condition": cond, "inheritance": inh, "source": "ACMG SF v3.2"}
    for g, cond, inh in _ACMG_SF_V3_2
}


# ------------------------------------------------------- carrier screening
#
# High-carrier-frequency subset of the ACMG/ACOG tier-3 expanded carrier
# panel (Gregg et al., Genet Med 2021; PMID 34285390). Full commercial
# panels (Invitae, Natera, Myriad Foresight) run to ~280–500 genes; this
# is the ~110 with pan-ethnic carrier frequency ≥ ~1/200 or that ACMG
# specifically calls out. Like PATHOGENIC_GENES this is a *gene* list —
# variant-level calls go through clinvar_lookup() below. Carrier status
# only matters for embryo selection when *both* parents carry P/LP in the
# same AR gene (or the mother for XL).
_CARRIER_PANEL: list[tuple[str, str, str]] = [
    # Pan-ethnic high-frequency core (ACMG tier 3 minimum)
    ("CFTR", "Cystic fibrosis", "AR"),
    ("SMN1", "Spinal muscular atrophy", "AR"),
    ("FMR1", "Fragile X syndrome", "XL"),
    ("HBB", "β-thalassaemia / sickle cell", "AR"),
    ("HBA1", "α-thalassaemia", "AR"),
    ("HBA2", "α-thalassaemia", "AR"),
    ("GJB2", "Non-syndromic hearing loss (DFNB1)", "AR"),
    ("PAH", "Phenylketonuria", "AR"),
    ("DMD", "Duchenne / Becker muscular dystrophy", "XL"),
    ("F8", "Haemophilia A", "XL"),
    ("F9", "Haemophilia B", "XL"),
    ("G6PD", "G6PD deficiency", "XL"),
    ("GALT", "Galactosaemia", "AR"),
    ("GBA", "Gaucher disease", "AR"),
    ("ATP7B", "Wilson disease", "AR"),
    ("MEFV", "Familial Mediterranean fever", "AR"),
    ("SLC26A4", "Pendred syndrome / DFNB4", "AR"),
    ("USH2A", "Usher syndrome 2A / RP", "AR"),
    ("ABCA4", "Stargardt disease", "AR"),
    ("CYP21A2", "Congenital adrenal hyperplasia (21-OH)", "AR"),
    ("PKHD1", "ARPKD", "AR"),
    ("MCOLN1", "Mucolipidosis IV", "AR"),
    ("HEXA", "Tay-Sachs disease", "AR"),
    ("HEXB", "Sandhoff disease", "AR"),
    ("ASPA", "Canavan disease", "AR"),
    ("ELP1", "Familial dysautonomia (IKBKAP)", "AR"),
    ("BLM", "Bloom syndrome", "AR"),
    ("FANCC", "Fanconi anaemia C", "AR"),
    ("FANCA", "Fanconi anaemia A", "AR"),
    ("G6PC", "Glycogen storage disease Ia", "AR"),
    ("SLC37A4", "Glycogen storage disease Ib", "AR"),
    ("SMPD1", "Niemann-Pick A/B", "AR"),
    ("NPC1", "Niemann-Pick C1", "AR"),
    ("IDUA", "MPS I (Hurler)", "AR"),
    ("IDS", "MPS II (Hunter)", "XL"),
    ("GAA", "Pompe disease", "AR"),
    ("GLA", "Fabry disease", "XL"),
    ("GALC", "Krabbe disease", "AR"),
    ("ARSA", "Metachromatic leukodystrophy", "AR"),
    ("ABCD1", "X-linked adrenoleukodystrophy", "XL"),
    # Metabolic / newborn-screen
    ("ACADM", "MCAD deficiency", "AR"),
    ("ACADVL", "VLCAD deficiency", "AR"),
    ("HADHA", "LCHAD deficiency", "AR"),
    ("BTD", "Biotinidase deficiency", "AR"),
    ("MMUT", "Methylmalonic acidaemia (mut)", "AR"),
    ("MMAA", "Methylmalonic acidaemia (cblA)", "AR"),
    ("MMAB", "Methylmalonic acidaemia (cblB)", "AR"),
    ("PCCA", "Propionic acidaemia", "AR"),
    ("PCCB", "Propionic acidaemia", "AR"),
    ("IVD", "Isovaleric acidaemia", "AR"),
    ("GCDH", "Glutaric acidaemia I", "AR"),
    ("BCKDHA", "Maple syrup urine disease", "AR"),
    ("BCKDHB", "Maple syrup urine disease", "AR"),
    ("DBT", "Maple syrup urine disease", "AR"),
    ("ASL", "Argininosuccinic aciduria", "AR"),
    ("ASS1", "Citrullinaemia I", "AR"),
    ("CBS", "Homocystinuria", "AR"),
    ("FAH", "Tyrosinaemia I", "AR"),
    ("MTHFR", "Severe MTHFR deficiency", "AR"),
    ("ALDOB", "Hereditary fructose intolerance", "AR"),
    ("SLC22A5", "Primary carnitine deficiency", "AR"),
    ("CPT1A", "CPT1A deficiency", "AR"),
    ("CPT2", "CPT2 deficiency", "AR"),
    # Endocrine / renal / skeletal
    ("NPHS1", "Congenital nephrotic syndrome (Finnish)", "AR"),
    ("NPHS2", "Steroid-resistant nephrotic syndrome", "AR"),
    ("AGXT", "Primary hyperoxaluria 1", "AR"),
    ("GRHPR", "Primary hyperoxaluria 2", "AR"),
    ("PEX1", "Zellweger spectrum", "AR"),
    ("PEX6", "Zellweger spectrum", "AR"),
    ("DHCR7", "Smith-Lemli-Opitz", "AR"),
    ("ALPL", "Hypophosphatasia", "AR"),
    ("COL1A1", "Osteogenesis imperfecta (AR forms)", "AR"),
    ("COL1A2", "Osteogenesis imperfecta", "AR"),
    ("SLC26A2", "Diastrophic dysplasia", "AR"),
    # Haematology / immunology
    ("HFE", "Hereditary haemochromatosis", "AR"),
    ("SERPINA1", "Alpha-1 antitrypsin deficiency", "AR"),
    ("PKLR", "Pyruvate kinase deficiency", "AR"),
    ("ADA", "ADA-SCID", "AR"),
    ("IL2RG", "X-linked SCID", "XL"),
    ("RAG1", "SCID / Omenn", "AR"),
    ("RAG2", "SCID / Omenn", "AR"),
    ("WAS", "Wiskott-Aldrich syndrome", "XL"),
    ("BTK", "X-linked agammaglobulinaemia", "XL"),
    ("CYBB", "Chronic granulomatous disease", "XL"),
    ("ITGB3", "Glanzmann thrombasthenia", "AR"),
    # Neuromuscular / neuro
    ("CAPN3", "LGMD2A", "AR"),
    ("DYSF", "LGMD2B / Miyoshi", "AR"),
    ("SGCA", "LGMD2D", "AR"),
    ("SGCB", "LGMD2E", "AR"),
    ("FKRP", "LGMD2I", "AR"),
    ("EMD", "Emery-Dreifuss MD", "XL"),
    ("NEB", "Nemaline myopathy", "AR"),
    ("RYR1", "Central core / multiminicore (AR forms)", "AR"),
    ("POLG", "POLG-related mitochondrial disease", "AR"),
    ("ATM", "Ataxia-telangiectasia", "AR"),
    ("FXN", "Friedreich ataxia (GAA repeat)", "AR"),
    ("SACS", "ARSACS", "AR"),
    ("AGL", "Glycogen storage disease III", "AR"),
    ("PYGM", "McArdle disease", "AR"),
    # Sensory
    ("MYO7A", "Usher syndrome 1B", "AR"),
    ("CDH23", "Usher syndrome 1D", "AR"),
    ("PCDH15", "Usher syndrome 1F", "AR"),
    ("CLRN1", "Usher syndrome 3A", "AR"),
    ("OTOF", "DFNB9 (auditory neuropathy)", "AR"),
    ("TMC1", "DFNB7/11", "AR"),
    ("MYO15A", "DFNB3", "AR"),
    ("CEP290", "Leber congenital amaurosis 10 / Joubert", "AR"),
    ("RPE65", "Leber congenital amaurosis 2", "AR"),
    ("RPGR", "X-linked retinitis pigmentosa", "XL"),
    ("CHM", "Choroideraemia", "XL"),
    ("CNGB3", "Achromatopsia", "AR"),
    ("TYR", "Oculocutaneous albinism 1", "AR"),
    ("OCA2", "Oculocutaneous albinism 2", "AR"),
    # Skin / connective
    ("COL7A1", "Dystrophic epidermolysis bullosa", "AR"),
    ("LAMA3", "Junctional epidermolysis bullosa", "AR"),
    ("LAMB3", "Junctional epidermolysis bullosa", "AR"),
    ("LAMC2", "Junctional epidermolysis bullosa", "AR"),
    ("ERCC6", "Cockayne syndrome B", "AR"),
    ("ERCC8", "Cockayne syndrome A", "AR"),
    ("NBN", "Nijmegen breakage syndrome", "AR"),
]

CARRIER_GENES: dict[str, dict[str, str]] = {
    g: {"condition": cond, "inheritance": inh, "source": "ACMG carrier screening 2021"}
    for g, cond, inh in _CARRIER_PANEL
}


# -------------------------------------------------------- ClinVar runtime


_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_last_clinvar_call = 0.0


def clinvar_lookup(
    chrom: str, pos: int, ref: str, alt: str, *, build: str = "GRCh37"
) -> dict | None:
    """Query ClinVar for a single variant via NCBI E-utilities.

    Returns ``{"variation_id", "clinical_significance", "review_status",
    "conditions": [...], "title"}`` for the first matching record, or
    ``None`` if not found / on network error (a warning is printed).

    NCBI rate-limits anonymous E-utilities to 3 requests/second; set
    ``NCBI_API_KEY`` to lift that to 10/sec. This function inserts a
    delay to stay under the limit. For bulk lookups, download the
    ClinVar VCF (https://ftp.ncbi.nlm.nih.gov/pub/clinvar/) instead."""
    global _last_clinvar_call
    api_key = os.environ.get("NCBI_API_KEY")
    min_interval = 0.11 if api_key else 0.34
    wait = min_interval - (time.time() - _last_clinvar_call)
    if wait > 0:
        time.sleep(wait)

    chrom = chrom.lstrip("chr")
    assembly = "GRCh37" if build in ("GRCh37", "hg19") else "GRCh38"
    term = (
        f'{chrom}[Chromosome] AND {pos}[Base Position for Assembly {assembly}] '
        f'AND "{ref}>{alt}"'
    )
    params = {"db": "clinvar", "term": term, "retmode": "json", "retmax": "5"}
    if api_key:
        params["api_key"] = api_key
    try:
        with urllib.request.urlopen(
            f"{_EUTILS}/esearch.fcgi?" + urllib.parse.urlencode(params), timeout=15
        ) as r:
            ids = json.load(r).get("esearchresult", {}).get("idlist", [])
        _last_clinvar_call = time.time()
        if not ids:
            return None
        wait = min_interval - (time.time() - _last_clinvar_call)
        if wait > 0:
            time.sleep(wait)
        params2 = {"db": "clinvar", "id": ids[0], "retmode": "json"}
        if api_key:
            params2["api_key"] = api_key
        with urllib.request.urlopen(
            f"{_EUTILS}/esummary.fcgi?" + urllib.parse.urlencode(params2), timeout=15
        ) as r:
            doc = json.load(r)["result"][ids[0]]
        _last_clinvar_call = time.time()
        gsig = doc.get("germline_classification", {}) or doc.get(
            "clinical_significance", {}
        )
        return {
            "variation_id": int(ids[0]),
            "title": doc.get("title", ""),
            "clinical_significance": gsig.get("description", ""),
            "review_status": gsig.get("review_status", ""),
            "conditions": [
                t.get("trait_name", "")
                for ts in doc.get("germline_classification", {}).get("trait_set", [])
                or doc.get("trait_set", [])
                for t in ([ts] if isinstance(ts, dict) else [])
            ],
        }
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as e:
        print(f"[clinvar] lookup failed for {chrom}:{pos}{ref}>{alt}: {e}", file=sys.stderr)
        return None


# ------------------------------------------------------------------ lookup


Carried = tuple[RareVariant, str, int]
"""(variant, "father" | "mother", haplotype_index 0|1)"""


_GNOMAD_CONSTRAINT_URL = (
    "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/"
    "constraint/gnomad.v4.1.constraint_metrics.tsv"
)
_GNOMAD_PLOF_URL = (
    "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/"
    "vcf/exomes/gnomad.exomes.v4.1.sites.chr{chrom}.vcf.bgz"
)


def load_gnomad_constraint(path: str | None = None) -> dict[str, dict[str, float]]:
    """gene → {loeuf, pli, oe_lof, exp_hom_lof, obs_hom_lof} from the
    gnomAD constraint table. Downloads (~5 MB) on first call if no
    local copy."""
    p = Path(path) if path else data_dir() / "gnomad_constraint.tsv"
    if not p.exists():
        print(f"[rare_variants] fetching gnomAD constraint → {p}", file=sys.stderr)
        urllib.request.urlretrieve(_GNOMAD_CONSTRAINT_URL, p)
    out: dict[str, dict[str, float]] = {}
    with open(p) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ix = {c: i for i, c in enumerate(hdr)}
        for line in f:
            r = line.rstrip("\n").split("\t")
            g = r[ix["gene"]]
            try:
                out[g] = {
                    "loeuf": float(r[ix.get("lof.oe_ci.upper", ix.get("oe_lof_upper"))]),
                    "pli": float(r[ix.get("lof.pLI", ix.get("pLI"))]),
                    "oe_lof": float(r[ix.get("lof.oe", ix.get("oe_lof"))]),
                }
            except (ValueError, KeyError, TypeError):
                continue
    return out


def scan_recessive_load(
    lof_calls: list[tuple[str, str, int, str, str]],
    constraint: dict[str, dict[str, float]] | None = None,
    loeuf_max: float = 0.6,
) -> list[dict]:
    """Per-genome recessive-load scan: which heterozygous LoF
    variants does this individual carry in constrained genes?

    `lof_calls` is a list of (gene, chrom, pos, ref, alt) for every
    predicted-LoF variant the individual carries — produced upstream
    by a variant-effect annotator (VEP, snpEff, bcftools csq, or by
    intersecting with the gnomAD pLoF site list at
    `_GNOMAD_PLOF_URL`). This module does not annotate variants
    itself; it ranks the supplied LoF calls by gene constraint.

    Returns one dict per call in a gene with LOEUF ≤ `loeuf_max`,
    sorted by LOEUF (most constrained first). Genes in
    PATHOGENIC_GENES or CARRIER_GENES are flagged so the caller can
    distinguish "known disease gene" from "constrained but
    uncharacterised — would probably be bad homozygous".

    This is *not* yet part of routine clinical carrier screening,
    which sticks to the curated disease-gene panels; it is what
    research-grade WES/WGS reports and some embryo-screening
    providers (e.g. Orchid's whole-genome embryo report) add on top.
    For an embryo workflow the actionable output is the intersection
    with the partner's scan: any gene where *both* parents carry LoF
    is a candidate for deselecting homozygous embryos."""
    if constraint is None:
        constraint = load_gnomad_constraint()
    rows: list[dict] = []
    for gene, chrom, pos, ref, alt in lof_calls:
        c = constraint.get(gene)
        if not c or c["loeuf"] > loeuf_max:
            continue
        rows.append(
            {
                "gene": gene,
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "loeuf": c["loeuf"],
                "pli": c["pli"],
                "known_disease": gene in PATHOGENIC_GENES or gene in CARRIER_GENES,
            }
        )
    rows.sort(key=lambda r: r["loeuf"])
    return rows


def check_genome(
    by_pos: dict[tuple[str, int], tuple], variants: list[RareVariant] | None = None
) -> list[tuple[RareVariant, int]]:
    """Curated variants carried by a single genome.

    `by_pos` is the (chrom, pos) → (a1, a2[, …]) map from
    genepred.io.load_genotypes(). Returns [(variant, n_copies)] for
    every variant with ≥1 copy of the alt allele. Variants without
    a resolved GRCh37 position are skipped — those are gene-level
    LoF aggregates that need a separate scan."""
    variants = variants if variants is not None else PROTECTIVE + LPA_RISK
    hits: list[tuple[RareVariant, int]] = []
    for v in variants:
        if v.pos_grch37 is None or v.alt is None:
            continue
        g = by_pos.get((v.chrom, int(v.pos_grch37)))
        if g is None:
            continue
        # hard_call collapses imputed (ref, alt, dosage) tuples; without it
        # every variant whose site merely exists in an imputed VCF would be
        # counted as carried.
        a1, a2 = hard_call(g)
        n = (a1 == v.alt) + (a2 == v.alt)
        if n > 0:
            hits.append((v, int(n)))
    return hits


def check_parents(
    par: Parents, variants: list[RareVariant] | None = None
) -> list[Carried]:
    """Variants from `variants` (default PROTECTIVE) that either parent
    carries on at least one haplotype, matched by (chrom, pos, ref, alt).

    Variants without a resolved GRCh37 position are skipped. The
    parental allele encoding is 0=REF / 1=ALT, so a carrier on hap h
    means par.{pat,mat}[h, idx] == 1 when the variant's ALT matches
    the site's ALT (or == 0 when reversed)."""
    variants = variants if variants is not None else PROTECTIVE
    pos_to_idx = {int(p): i for i, p in enumerate(par.pos)}
    out: list[Carried] = []
    for v in variants:
        if v.chrom != par.chrom or v.pos_grch37 is None:
            continue
        i = pos_to_idx.get(v.pos_grch37)
        if i is None:
            continue
        if v.ref == par.ref[i] and v.alt == par.alt[i]:
            target = 1
        elif v.ref == par.alt[i] and v.alt == par.ref[i]:
            target = 0
        else:
            continue
        for who, hap in (("father", par.pat), ("mother", par.mat)):
            for h in (0, 1):
                if int(hap[h, i]) == target:
                    out.append((v, who, h))
    return out


def flag_embryos(
    carried: list[Carried],
    wp: np.ndarray,
    wm: np.ndarray,
    par: Parents,
) -> list[dict[str, bool]]:
    """Per-embryo {gene: inherited?} for each carried parental variant.

    `wp`/`wm` are the (E, M) inheritance-path arrays from
    `joint_recover` — wp[e, i] is which paternal haplotype embryo e
    carries at site i (and likewise wm for maternal)."""
    E = wp.shape[0]
    pos_to_idx = {int(p): i for i, p in enumerate(par.pos)}
    out: list[dict[str, bool]] = [dict() for _ in range(E)]
    for v, who, h in carried:
        i = pos_to_idx.get(v.pos_grch37)  # type: ignore[arg-type]
        if i is None:
            continue
        path = wp if who == "father" else wm
        for e in range(E):
            out[e][v.gene] = out[e].get(v.gene, False) or int(path[e, i]) == h
    return out


def summarise(
    carried: list[Carried], flags: list[dict[str, bool]]
) -> str:
    """Compact text block for the embryo report."""
    if not carried:
        return "No curated rare variants carried by either parent."
    lines = ["Rare variants carried by parents:"]
    by_gene: dict[str, RareVariant] = {}
    for v, who, h in carried:
        by_gene.setdefault(v.gene, v)
        lines.append(
            f"  {v.gene:<10} {who} hap{h}  [{v.direction}, tier {v.evidence_tier}] "
            f"{v.effect_description}  ({v.source})"
        )
    lines.append("")
    lines.append("Inheritance per embryo:")
    hdr = "  gene      " + "".join(f" e{e + 1:>2}" for e in range(len(flags)))
    lines.append(hdr)
    for g in by_gene:
        row = "  " + f"{g:<10}" + "".join(
            ("  ✓" if flags[e].get(g) else "  ·") for e in range(len(flags))
        )
        lines.append(row)
    return "\n".join(lines)
