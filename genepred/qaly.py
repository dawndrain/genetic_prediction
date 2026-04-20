"""QALY and cost model for polygenic score predictions.

Given PGS z-scores for a set of traits, estimates absolute disease risk
(via the liability-threshold model), expected QALY impact, and expected
lifetime cost impact relative to the population average. Also provides
`simulate_selection`, a Monte Carlo over correlated sibling-PGS draws
that estimates the expected gain from picking the QALY-best of N embryos.

CLI lives in `genepred.cli` (`genepred qaly` / `genepred select`).
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Trait definitions
# ---------------------------------------------------------------------------


@dataclass
class _PGSTrait:
    # Out-of-sample population R² of the current best PGS in EUR.
    pgs_r2_population: float
    # Within-sibship attenuation: (within-family R²) / (population R²).
    # <1 because population R² is inflated by stratification, assortative
    # mating, and indirect genetic (parental nurture) effects, none of which
    # discriminate between embryos from the same parents. Ratios from
    # Howe et al. 2022 Nat Genet sibling-GWAS where available.
    within_family_ratio: float

    def effective_r2(self, *, within_family: bool) -> float:
        if within_family:
            return self.pgs_r2_population * self.within_family_ratio
        return self.pgs_r2_population

    @property
    def pgs_r2(self) -> float:  # legacy alias for callers that haven't migrated
        return self.effective_r2(within_family=True)


# Cross-ancestry attenuation: average ratio of (R² in target population)
# to (R² in EUR) when scoring with EUR-trained weights. From Martin et al.
# 2019 Nat Genet, Privé et al. 2022 AJHG, Duncan et al. 2019 — averaged
# over ~250 quantitative + disease traits in UKB. Per-trait spread is
# wide (height ~0.20× in AFR, some immune/lipid traits ~0.6×); these
# are population means.  AMR varies most because the 1KG-AMR samples
# span a large EUR-admixture gradient.
ANCESTRY_R2_RATIO: dict[str, float] = {
    "EUR": 1.00,
    "AMR": 0.65,
    "SAS": 0.55,
    "EAS": 0.50,
    "AFR": 0.30,
}


@dataclass
class DiseaseTrait(_PGSTrait):
    """A binary disease trait scored by PGS."""

    name: str
    display_name: str
    # Lifetime prevalence in the general population
    prevalence: float
    # Average QALYs lost over remaining lifetime if you develop the condition
    qaly_loss_if_affected: float
    # Average lifetime treatment cost (USD) if you develop the condition.
    # Kept for backward compatibility; if cost_medical / cost_productivity
    # are set, those are used instead and this becomes their sum.
    lifetime_cost_if_affected: float
    # Direct medical cost (hospitalization, drugs, procedures, long-term
    # care) regardless of payer. Subject to excess healthcare inflation
    # when discounting. None → fall back to lifetime_cost_if_affected.
    cost_medical: float | None = None
    # Indirect cost: lost earnings from morbidity + premature mortality,
    # valued at human-capital (forgone wages). Discounted at the general
    # real rate (no excess inflation).
    cost_productivity: float | None = None
    # Typical age of onset (used for discounting)
    typical_onset_age: float = 50.0
    # SNP-heritability (LDSC). Used to attenuate cross-trait rg to
    # the PGS-level correlation: cor(PGS_X,PGS_Y)≈rg·√(R²_X/h²_X·R²_Y/h²_Y).
    snp_h2: float | None = None
    # Sources / notes for the above numbers
    sources: str = ""

    def costs(self) -> tuple[float, float]:
        """(medical, productivity). Falls back to a 60/40 split of the
        legacy single-figure cost if the split fields aren't set —
        roughly the medical:productivity ratio across published
        cost-of-illness studies for chronic conditions."""
        if self.cost_medical is not None and self.cost_productivity is not None:
            return self.cost_medical, self.cost_productivity
        return (
            0.6 * self.lifetime_cost_if_affected,
            0.4 * self.lifetime_cost_if_affected,
        )

    def cost_personal(self, personal_medical_share: float = 0.12) -> float:
        """Out-of-pocket medical share + own lost income. The default
        0.12 is the US average out-of-pocket fraction of national
        health expenditure (CMS NHE 2023); near-zero in single-payer
        systems."""
        med, prod = self.costs()
        return personal_medical_share * med + prod

    def cost_societal(self) -> float:
        med, prod = self.costs()
        return med + prod


@dataclass
class ContinuousTrait(_PGSTrait):
    """A continuous trait scored by PGS (e.g. height, cognition)."""

    name: str
    display_name: str
    # QALY change per 1 SD increase in the trait
    qaly_per_sd: float
    # Savings per 1 SD increase (positive = good: earnings, avoided costs)
    savings_per_sd: float
    # Age at which the QALY/savings effects are centered (for discounting).
    # For traits with lifelong effects (cognition, height), use a midlife value.
    typical_effect_age: float = 40.0
    snp_h2: float | None = None
    sources: str = ""


# --- Disease traits ---
# Prevalence: lifetime risk for a ~European-ancestry population
# QALY losses: from GBD / CEA registry / published utility studies
# Costs: US lifetime treatment cost estimates

DISEASE_TRAITS: dict[str, DiseaseTrait] = {
    "heart_disease": DiseaseTrait(
        name="heart_disease",
        display_name="Coronary heart disease",
        prevalence=0.36,  # lifetime risk ~36% (sex-avg; Lloyd-Jones 2006 Framingham)
        qaly_loss_if_affected=3.0,  # ~3 QALYs lost on average (acute event + chronic HF)
        lifetime_cost_if_affected=250_000,  # US lifetime cost estimate
        cost_medical=137_500,  # AHA HD&S Stats ~55/45
        cost_productivity=112_500,
        pgs_r2_population=0.06,  # Patel 2023 GPSmult
        within_family_ratio=0.8,
        snp_h2=0.1,
        typical_onset_age=60,
        sources="Framingham lifetime risk; GBD 2019; Dunbar-Rees 2018",
    ),
    "type2_diabetes": DiseaseTrait(
        name="type2_diabetes",
        display_name="Type 2 diabetes",
        prevalence=0.13,  # ~13% lifetime risk
        qaly_loss_if_affected=5.0,  # chronic condition, ~5 QALYs over remaining life
        lifetime_cost_if_affected=200_000,  # ADA lifetime cost estimates
        cost_medical=144_000,  # ADA 2022: $307B direct/$106B indirect ≈74/26
        cost_productivity=56_000,
        pgs_r2_population=0.07,  # Suzuki 2024 DIAMANTE
        within_family_ratio=0.8,
        snp_h2=0.18,
        typical_onset_age=55,
        sources="CDC lifetime risk; Zhuo 2013 ADA cost; Sullivan 2020 QALY",
    ),
    "alzheimers": DiseaseTrait(
        name="alzheimers",
        display_name="Alzheimer's disease",
        prevalence=0.10,  # ~10% lifetime risk (age 65+)
        qaly_loss_if_affected=8.5,  # 8-12yr duration × 0.7-0.9 disability weight
        lifetime_cost_if_affected=350_000,  # Alzheimer's Association 2023
        cost_medical=175_000,  # Alz Assoc: ~50/50 counting unpaid caregiving as productivity
        cost_productivity=175_000,
        pgs_r2_population=0.07,  # APOE e4 dominates; GenoBoost AUROC 0.83 (Ohta 2024, N=67K)
        within_family_ratio=0.9,
        snp_h2=0.2,
        typical_onset_age=75,
        sources="Alzheimer's Association 2023; GBD DALY estimates; Ohta 2024 Nat Commun",
    ),
    "schizophrenia": DiseaseTrait(
        name="schizophrenia",
        display_name="Schizophrenia",
        prevalence=0.007,  # ~0.7% lifetime risk
        qaly_loss_if_affected=15.0,  # early onset, chronic, severe
        lifetime_cost_if_affected=1_500_000,  # lifetime societal cost
        cost_medical=375_000,  # Cloutier 2016: $38B/$117B ≈25/75 (unemployment dominates)
        cost_productivity=1_125_000,
        pgs_r2_population=0.08,  # PGC3
        within_family_ratio=0.75,
        snp_h2=0.24,
        typical_onset_age=22,
        sources="McGrath 2008 epidemiology; Chong 2016 cost; Millier 2014 QALY",
    ),
    "depression": DiseaseTrait(
        name="depression",
        display_name="Major depressive disorder",
        prevalence=0.16,  # ~16% lifetime prevalence
        qaly_loss_if_affected=3.0,  # episodic but recurrent
        lifetime_cost_if_affected=100_000,
        cost_medical=30_000,  # Greenberg 2021: $98B/$228B ≈30/70
        cost_productivity=70_000,
        pgs_r2_population=0.02,
        within_family_ratio=0.6,
        snp_h2=0.08,
        typical_onset_age=30,
        sources="Kessler 2005 NCS-R; Greenberg 2021 cost; Saarni 2007 QALY",
    ),
    "atrial_fibrillation": DiseaseTrait(
        name="atrial_fibrillation",
        display_name="Atrial fibrillation",
        prevalence=0.25,  # ~25% lifetime risk after age 40 (Framingham/Rotterdam)
        qaly_loss_if_affected=1.5,  # chronic but manageable; moderate utility decrement
        lifetime_cost_if_affected=150_000,
        cost_medical=112_500,  # Kim 2011: ablation/anticoag; older onset
        cost_productivity=37_500,
        pgs_r2_population=0.06,  # Weng 2024
        within_family_ratio=0.85,
        snp_h2=0.12,
        typical_onset_age=65,
        sources="Lloyd-Jones 2004 Framingham; Roselli 2018 AF GWAS; Kim 2011 AF cost",
    ),
    "breast_cancer": DiseaseTrait(
        name="breast_cancer",
        display_name="Breast cancer",
        prevalence=0.065,  # ~13% in women, ~6.5% population-average
        qaly_loss_if_affected=3.0,
        lifetime_cost_if_affected=150_000,
        cost_medical=90_000,  # Mariotto 2011 NCI
        cost_productivity=60_000,
        pgs_r2_population=0.07,  # Mavaddat 2019 PRS313; one of best cancer PGS
        within_family_ratio=0.85,
        snp_h2=0.13,
        typical_onset_age=55,
        sources="SEER lifetime risk; Mavaddat 2019 AJHG PRS313; Campbell 2019 cost",
    ),
    "prostate_cancer": DiseaseTrait(
        name="prostate_cancer",
        display_name="Prostate cancer",
        prevalence=0.065,  # ~13% in men, ~6.5% population-average
        qaly_loss_if_affected=2.0,  # many low-grade; treatment side effects
        lifetime_cost_if_affected=120_000,
        cost_medical=86_400,  # Roehrborn 2011: older onset
        cost_productivity=33_600,
        pgs_r2_population=0.09,  # Conti 2021 PRS269
        within_family_ratio=0.85,
        snp_h2=0.2,
        typical_onset_age=65,
        sources="SEER lifetime risk; Conti 2021 Nature Genetics; Wilson 2007 cost",
    ),
    "stroke": DiseaseTrait(
        name="stroke",
        display_name="Stroke",
        prevalence=0.22,  # lifetime risk ~22% (GBD 2016 '1 in 4')
        qaly_loss_if_affected=4.0,  # acute mortality ~15% + severe disability
        lifetime_cost_if_affected=200_000,
        cost_medical=124_000,  # AHA: acute+rehab+LTC
        cost_productivity=76_000,
        pgs_r2_population=0.02,  # GIGASTROKE
        within_family_ratio=0.8,
        snp_h2=0.12,
        typical_onset_age=65,
        sources="Seshadri 2006 Framingham; Mishra 2022; Taylor 1996 cost updated",
    ),
    "colorectal_cancer": DiseaseTrait(
        name="colorectal_cancer",
        display_name="Colorectal cancer",
        prevalence=0.04,  # ~4% lifetime risk (SEER)
        qaly_loss_if_affected=3.5,
        lifetime_cost_if_affected=180_000,
        cost_medical=99_000,  # Yabroff 2011
        cost_productivity=81_000,
        pgs_r2_population=0.04,  # Huyghe 2019, Thomas 2023
        within_family_ratio=0.85,
        snp_h2=0.1,
        typical_onset_age=60,
        sources="SEER lifetime risk; Huyghe 2019 Nat Genet; Mariotto 2011 cost",
    ),
    "bipolar_disorder": DiseaseTrait(
        name="bipolar_disorder",
        display_name="Bipolar disorder",
        prevalence=0.01,  # ~1% bipolar I
        qaly_loss_if_affected=12.0,  # early onset, chronic, ~15yr life expectancy reduction
        lifetime_cost_if_affected=600_000,
        cost_medical=180_000,  # Cloutier 2018: productivity-dominated like SCZ
        cost_productivity=420_000,
        pgs_r2_population=0.04,  # PGC3 Mullins 2021
        within_family_ratio=0.75,
        snp_h2=0.2,
        typical_onset_age=25,
        sources="Merikangas 2007 NCS-R; PGC3 Mullins 2021; Dilsaver 2003 cost",
    ),
    "chronic_kidney_disease": DiseaseTrait(
        name="chronic_kidney_disease",
        display_name="Chronic kidney disease",
        prevalence=0.07,  # ~7% develop CKD stage 3+
        qaly_loss_if_affected=2.0,
        lifetime_cost_if_affected=150_000,
        cost_medical=127_500,  # USRDS: dialysis dominates
        cost_productivity=22_500,
        pgs_r2_population=0.02,  # Khan 2022 Nat Commun
        within_family_ratio=0.8,
        snp_h2=0.07,
        typical_onset_age=60,
        sources="USRDS 2023; Khan 2022 Nat Commun; Honeycutt 2003 CKD cost",
    ),
    "asthma": DiseaseTrait(
        name="asthma",
        display_name="Asthma",
        prevalence=0.08,  # ~8% current prevalence
        qaly_loss_if_affected=1.0,  # mostly mild-moderate
        lifetime_cost_if_affected=50_000,
        cost_medical=24_000,  # Nurmagambetov 2018
        cost_productivity=26_000,
        pgs_r2_population=0.04,  # GBMI Wang 2023
        within_family_ratio=0.8,
        snp_h2=0.12,
        typical_onset_age=10,
        sources="CDC asthma prevalence; Shrine 2023; Nurmagambetov 2018 cost",
    ),
    "inflammatory_bowel_disease": DiseaseTrait(
        name="inflammatory_bowel_disease",
        display_name="Inflammatory bowel disease",
        prevalence=0.008,  # ~0.5% Crohn's + ~0.3% UC
        qaly_loss_if_affected=4.0,  # chronic, relapsing; surgery common
        lifetime_cost_if_affected=400_000,  # biologics ~$30-50k/yr
        cost_medical=260_000,  # Park 2020: biologics; young onset
        cost_productivity=140_000,
        pgs_r2_population=0.06,  # Strong known loci (NOD2, IL23R); snpnet AUROC 0.90 (Tanigawa 2022) but inflated
        within_family_ratio=0.85,
        snp_h2=0.27,
        typical_onset_age=30,
        sources="Ng 2017 Lancet; de Lange 2017 Nat Genet; Park 2020 IBD cost",
    ),
    "adhd": DiseaseTrait(
        name="adhd",
        display_name="ADHD",
        prevalence=0.05,  # ~5% childhood, ~2.5% persistent into adulthood; using ~5% lifetime
        qaly_loss_if_affected=4.0,  # academic failure, accidents, substance abuse, relationships
        lifetime_cost_if_affected=300_000,  # Doshi 2012; treatment + lost productivity
        cost_medical=66_000,  # Doshi 2012: education/employment dominate
        cost_productivity=234_000,
        pgs_r2_population=0.04,  # Demontis 2023 PGC; R² ~3-5% liability
        within_family_ratio=0.6,
        snp_h2=0.22,
        typical_onset_age=8,
        sources="Demontis 2023 Nat Genet; Doshi 2012 JAACAP cost; Daley 2015 QALY",
    ),
    "type1_diabetes": DiseaseTrait(
        name="type1_diabetes",
        display_name="Type 1 diabetes",
        prevalence=0.005,  # ~0.5% lifetime; onset usually childhood/adolescence
        qaly_loss_if_affected=10.0,  # lifelong insulin dependence, complications, reduced lifespan
        lifetime_cost_if_affected=1_000_000,  # insulin + CGM + complications; Tao 2010
        cost_medical=650_000,  # Tao 2010
        cost_productivity=350_000,
        pgs_r2_population=0.15,  # HLA-dominated; Sharp 2019 AUROC 0.96 with 85 SNPs. Genetics-only ~0.10-0.15
        within_family_ratio=0.9,
        snp_h2=0.5,
        typical_onset_age=12,
        sources="Maahs 2010 epidemiology; Sharp 2019 Diabetes Care; Tao 2010 cost",
    ),
    "osteoporosis": DiseaseTrait(
        name="osteoporosis",
        display_name="Osteoporosis",
        prevalence=0.10,  # ~10% lifetime prevalence of osteoporotic fracture
        qaly_loss_if_affected=2.0,  # hip fracture → high mortality + disability; vertebral less severe
        lifetime_cost_if_affected=100_000,  # Burge 2007; hip fracture ~$40k acute + long-term care
        cost_medical=80_000,  # Burge 2007: fracture treatment
        cost_productivity=20_000,
        pgs_r2_population=0.05,  # Morris 2019 BMD GWAS; PGS for BMD, used as proxy for fracture
        within_family_ratio=0.85,
        snp_h2=0.2,
        typical_onset_age=70,
        sources="Morris 2019 Nat Genet; Burge 2007 cost; Peasgood 2009 QALY",
    ),
    "anxiety_disorders": DiseaseTrait(
        name="anxiety_disorders",
        display_name="Anxiety disorders",
        prevalence=0.20,  # ~20% lifetime prevalence (GAD + social + panic + specific)
        qaly_loss_if_affected=2.0,  # chronic, moderate disability; less severe than MDD on average
        lifetime_cost_if_affected=60_000,
        cost_medical=18_000,  # Konnopka 2020
        cost_productivity=42_000,
        pgs_r2_population=0.01,  # Purves 2020; very low R², limited GWAS power
        within_family_ratio=0.6,
        snp_h2=0.05,
        typical_onset_age=20,
        sources="Bandelow 2015 epidemiology; Purves 2020 Nat Genet; Konnopka 2009 cost",
    ),
}

# --- Continuous traits ---

CONTINUOUS_TRAITS: dict[str, ContinuousTrait] = {
    "height": ContinuousTrait(
        name="height",
        display_name="Height",
        # Height has a small but nonzero association with mortality/earnings.
        # Literature suggests ~0.02-0.05 QALY per SD through indirect pathways
        # (earnings, self-esteem, cardiovascular). Very uncertain.
        qaly_per_sd=0.03,
        # Taller people earn slightly more (~$800/inch in some studies).
        # 1 SD height ≈ 2.7 inches → ~$2000/SD in lifetime earnings premium.
        # Modeled as a cost "saving" (higher earning), sign is debatable.
        savings_per_sd=5_000,  # lifetime earnings premium per SD
        # PGS Catalog reports R²=0.717 for PGS001229 but that includes
        # sex/age covariates in snpnet. Genetics-only R² is ~0.16.
        pgs_r2_population=0.42,  # Yengo 2022 GIANT; Howe 2022 ratio ~0.85
        within_family_ratio=0.85,
        snp_h2=0.48,
        typical_effect_age=30,
        sources="Judge & Cable 2004 earnings; NCD-RisC 2016; PGS001229 Tanigawa 2022",
    ),
    "cognitive_ability": ContinuousTrait(
        name="cognitive_ability",
        display_name="Cognitive ability",
        # Higher cognitive ability associated with better health outcomes,
        # higher earnings, longer life. ~0.5-1 QALY per SD is plausible
        # through combined pathways (health literacy, SES, longevity).
        qaly_per_sd=0.5,
        # Lifetime earnings difference per SD of cognitive ability is large.
        # Conservative estimate: ~$100k-300k lifetime.
        savings_per_sd=200_000,
        # PGS001232 reports R²=0.127 on UKB fluid intelligence but that's the
        # full model (PGS + age + sex + PCs). Incremental R² (genetics-only) is
        # 0.050 in white British, 0.028 in non-British European. Using ~0.05.
        pgs_r2_population=0.1,  # EA4-derived; Selzam 2019 within-fam ~50%
        within_family_ratio=0.5,
        snp_h2=0.2,
        typical_effect_age=35,
        sources="Gottfredson 2004; Batty 2007 IQ-mortality; Zagorsky 2007; PGS001232 Tanigawa 2022 (incremental R²)",
    ),
    "bmi": ContinuousTrait(
        name="bmi",
        display_name="BMI",
        # Higher BMI = worse. 1 SD BMI ~ 4.5 kg/m².
        # Total MR estimate: ~1.5 QALY/SD, but much is mediated through
        # T2D (~30%), CVD (~15%), CKD (~5%), cancer (~5%) which are already
        # in the model. Direct effects (joint pain, sleep apnea, mobility,
        # psychosocial) are ~40-50% of total. Using 0.6 as direct-only.
        qaly_per_sd=-0.6,
        # Similarly reduce cost to direct-only (not through diseases in model)
        savings_per_sd=-6_000,
        pgs_r2_population=0.12,  # Howe 2022 ratio ~0.7
        within_family_ratio=0.7,
        snp_h2=0.25,
        typical_effect_age=50,
        sources="Gou 2025 Nat Med; Dixon 2021 MR BMI-QALY; Cawley 2012 obesity cost; Privé 2022",
    ),
    "educational_attainment": ContinuousTrait(
        name="educational_attainment",
        display_name="Educational attainment",
        # Years of schooling (1 SD ≈ 3.5 years in EA4). Heavily mediated
        # by cognition (rg ≈ 0.70) and predicts the same downstream
        # outcomes (income, health-literacy, mortality), so to avoid
        # triple-counting against cognition AND income, only the residual
        # direct-education effect is valued here: roughly the wage
        # premium and health gain from credentialing/network access net
        # of ability. Set to 0 by default — the cognition + income
        # entries already carry the QALY/$ for this pathway. Kept as a
        # trait so the score appears in reports.
        qaly_per_sd=0.0,
        savings_per_sd=0,
        pgs_r2_population=0.13,  # EA4 excl-23andMe (full EA4 ≈0.16)
        within_family_ratio=0.40,  # Howe 2022: EA shrinks ~60% within-family
        sources="Okbay 2022 EA4; Howe 2022; Heckman 2018 (returns to ed)",
    ),
    "income": ContinuousTrait(
        name="income",
        display_name="Household income",
        # UKB measures household income. Hill 2019 GWAS of income.
        # SNP h² of income ~7-10%. rg with cognitive ability is ~0.70,
        # so most genetic effect on income is via cognition + education.
        # Direct health effect (independent of cognition/education):
        # ~0.05-0.1 QALY/SD from healthcare access, housing, nutrition.
        qaly_per_sd=0.05,
        # Savings is the income itself. 1 SD of household income (the
        # trait, log-scale) ≈ 60–80% earnings difference ≈ $1.5–2M
        # lifetime at US median. Cognition's $200k/SD already carries
        # the cognition→earnings pathway; what's left is the income
        # variance not mediated by cognition. At the genetic level
        # rg≈0.70 → ~50% of income's genetic variance is independent;
        # taking √0.5 of the trait effect and being conservative about
        # the lifetime $ figure gives a residual on the order of
        # $300–600k. Using $90k here is deliberately low (≈ the
        # within-family-attenuated, post-cognition-overlap value) but
        # this is among the shakiest parameters in the model.
        savings_per_sd=90_000,
        # PGS R² for income is low — most variance is environmental.
        pgs_r2_population=0.025,  # Hill 2019; Howe 2022 within-family R² even lower
        within_family_ratio=0.35,
        snp_h2=0.05,
        typical_effect_age=40,
        sources="Hill 2019 Nat Commun income GWAS; Howe 2022 Science within-family",
    ),
    "longevity_residual": ContinuousTrait(
        name="longevity_residual",
        display_name="Longevity (residual)",
        # This is the component of the longevity PGS NOT explained by the
        # individual disease/trait PGS already in the model. Computed by
        # regressing longevity PGS on all other trait PGS and taking residuals.
        #
        # Original longevity PGS: R² = 0.015 for lifespan.
        # ~34% of its variance is explained by the individual trait PGS
        # (via genetic correlations with CVD, T2D, BMI, cognition, etc.)
        # Residual R² = 0.015 * 0.66 ≈ 0.01
        #
        # What's in the residual: immune function, cancer types not modeled,
        # vascular aging, telomere biology, DNA repair, rare diseases, etc.
        #
        # qaly_per_sd = 7.0 (12yr SD × 0.6 QALY/marginal yr) — same as
        # full longevity since the residual still predicts *lifespan*.
        qaly_per_sd=7.0,
        savings_per_sd=0,
        # Zeroed: no validated longevity PGS in genepred.catalog, and at
        # qaly_per_sd=7 even R²=0.01 makes this the largest single QALY
        # contributor — a phantom. Restore once a Timmers-style score is
        # added to the curated set and the residualization above is
        # actually computed (not just described).
        pgs_r2_population=0.0,
        within_family_ratio=0.8,
        snp_h2=0.0,
        typical_effect_age=75,
        sources="Timmers 2019 eLife; Deelen 2019; residual after regressing on trait PGS",
    ),
    "subjective_wellbeing": ContinuousTrait(
        name="subjective_wellbeing",
        display_name="Subjective wellbeing",
        # Okbay 2016 GWAS of subjective wellbeing (life satisfaction).
        # SNP h² ~5-10%. Very noisy phenotype.
        # Hard to monetize, but 1 SD of life satisfaction ≈ 0.5-1.0 QALY
        # if you take utility weights seriously (which is circular...).
        qaly_per_sd=0.5,
        savings_per_sd=0,  # no clear cost pathway
        # PGS prediction is very weak — R² ~1-2% at best.
        pgs_r2_population=0.02,  # Okbay 2016; Baselmans 2019
        within_family_ratio=0.55,
        snp_h2=0.05,
        typical_effect_age=40,
        sources="Okbay 2016 Nat Genet; Baselmans 2019; Clark 2018 wellbeing valuation",
    ),
}


# ---------------------------------------------------------------------------
# Genetic correlation matrix (from LD score regression / published rg values)
# ---------------------------------------------------------------------------
# Sources: Bulik-Sullivan 2015 (LD Hub), Zheng 2017, Grotzinger 2019 (GenomicSEM),
# various trait-specific LDSC papers. Values are approximate rg between the
# *genetic components* of each trait pair.
#
# Trait order must match: list(DISEASE_TRAITS) + list(CONTINUOUS_TRAITS)


def get_trait_order() -> list[str]:
    return list(DISEASE_TRAITS.keys()) + list(CONTINUOUS_TRAITS.keys())


# fmt: off
# Approximate genetic correlations (rg) between trait PGS scores.
# Symmetric matrix; diagonal = 1.0.
# Where published rg isn't available, set to 0.
_RG_VALUES = {
    # Metabolic cluster: BMI ↔ cardiometabolic diseases
    ("bmi", "type2_diabetes"):            0.40,
    ("bmi", "heart_disease"):             0.20,
    ("bmi", "stroke"):                    0.15,
    ("bmi", "chronic_kidney_disease"):    0.20,
    ("bmi", "atrial_fibrillation"):       0.15,
    ("bmi", "breast_cancer"):            -0.05,  # weak protective for premenopausal
    ("bmi", "colorectal_cancer"):         0.10,
    ("type2_diabetes", "heart_disease"):  0.30,
    ("type2_diabetes", "stroke"):         0.20,
    ("type2_diabetes", "chronic_kidney_disease"): 0.25,
    ("type2_diabetes", "atrial_fibrillation"):    0.10,
    ("heart_disease", "stroke"):          0.40,
    ("heart_disease", "atrial_fibrillation"):     0.25,
    ("heart_disease", "chronic_kidney_disease"):  0.15,

    # Psychiatric cluster
    ("schizophrenia", "bipolar_disorder"):  0.70,
    ("schizophrenia", "depression"):        0.30,
    ("depression", "bipolar_disorder"):     0.30,
    ("depression", "subjective_wellbeing"): -0.50,
    ("schizophrenia", "cognitive_ability"): -0.20,
    ("depression", "cognitive_ability"):    -0.30,
    ("bipolar_disorder", "cognitive_ability"): 0.05,  # weak/inconsistent

    # Cognition / SES cluster
    ("cognitive_ability", "income"):         0.70,
    ("cognitive_ability", "height"):         0.15,
    ("cognitive_ability", "bmi"):           -0.20,
    ("income", "height"):                    0.10,
    ("income", "bmi"):                      -0.20,
    ("income", "depression"):               -0.30,
    ("income", "subjective_wellbeing"):      0.30,

    # Height ↔ diseases
    ("height", "heart_disease"):            -0.10,
    ("height", "type2_diabetes"):           -0.10,
    ("height", "breast_cancer"):             0.10,
    ("height", "prostate_cancer"):           0.10,

    # Cancers (mostly weakly correlated)
    ("breast_cancer", "prostate_cancer"):    0.05,
    ("breast_cancer", "colorectal_cancer"):  0.05,
    ("prostate_cancer", "colorectal_cancer"): 0.05,

    # longevity_residual: by construction uncorrelated with all other trait PGS
    # (it's the residual from regressing longevity PGS on the others)

    # ADHD
    ("adhd", "depression"):                  0.40,  # strong comorbidity
    ("adhd", "bipolar_disorder"):            0.15,
    ("adhd", "schizophrenia"):               0.10,
    ("adhd", "cognitive_ability"):          -0.20,  # modest negative rg
    ("adhd", "bmi"):                         0.15,
    ("adhd", "income"):                     -0.20,
    ("adhd", "subjective_wellbeing"):       -0.10,
    ("adhd", "anxiety_disorders"):           0.30,

    # Type 1 diabetes (largely distinct genetics from T2D; HLA-dominated)
    ("type1_diabetes", "type2_diabetes"):   -0.05,  # slightly negative, different biology
    ("type1_diabetes", "inflammatory_bowel_disease"): 0.10,  # shared autoimmune loci
    ("type1_diabetes", "asthma"):            0.05,

    # Osteoporosis (BMD)
    ("osteoporosis", "height"):             -0.15,  # shorter → lower BMD
    ("osteoporosis", "bmi"):                -0.10,  # lower BMI → lower BMD

    # Anxiety disorders
    ("anxiety_disorders", "depression"):     0.70,  # very high genetic overlap
    ("anxiety_disorders", "bipolar_disorder"): 0.15,
    ("anxiety_disorders", "schizophrenia"):  0.15,
    ("anxiety_disorders", "subjective_wellbeing"): -0.40,
    ("anxiety_disorders", "cognitive_ability"): -0.15,
    ("anxiety_disorders", "income"):         -0.20,

    # Wellbeing
    ("subjective_wellbeing", "bmi"):        -0.10,
    ("subjective_wellbeing", "heart_disease"): -0.10,

    # Asthma / IBD (immune-mediated, weakly correlated)
    ("asthma", "inflammatory_bowel_disease"): 0.10,
}
# fmt: on


def build_genetic_correlation_matrix() -> np.ndarray:
    """Build the full genetic correlation matrix for all traits."""
    traits = get_trait_order()
    n = len(traits)
    trait_idx = {name: i for i, name in enumerate(traits)}

    corr = np.eye(n)
    for (t1, t2), rg in _RG_VALUES.items():
        if t1 in trait_idx and t2 in trait_idx:
            i, j = trait_idx[t1], trait_idx[t2]
            corr[i, j] = rg
            corr[j, i] = rg

    # Ensure positive semi-definite (clamp tiny negative eigenvalues)
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-6)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalize diagonal to 1
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)

    return corr


# ---------------------------------------------------------------------------
# Discounting
# ---------------------------------------------------------------------------

# US Social Security period life table (2020), simplified.
# P(surviving to age X | alive at birth). Used to discount by mortality.
_SURVIVAL_TABLE = {
    0: 1.000,
    5: 0.994,
    10: 0.993,
    15: 0.992,
    20: 0.990,
    25: 0.987,
    30: 0.984,
    35: 0.980,
    40: 0.975,
    45: 0.967,
    50: 0.955,
    55: 0.936,
    60: 0.906,
    65: 0.862,
    70: 0.798,
    75: 0.706,
    80: 0.581,
    85: 0.424,
    90: 0.255,
    95: 0.108,
}


def survival_probability(age: float) -> float:
    """Interpolate survival probability to a given age."""
    ages = sorted(_SURVIVAL_TABLE.keys())
    if age <= ages[0]:
        return 1.0
    if age >= ages[-1]:
        return _SURVIVAL_TABLE[ages[-1]]
    for i in range(len(ages) - 1):
        if ages[i] <= age <= ages[i + 1]:
            t = (age - ages[i]) / (ages[i + 1] - ages[i])
            return _SURVIVAL_TABLE[ages[i]] * (1 - t) + _SURVIVAL_TABLE[ages[i + 1]] * t
    return 0.0


def discount_factor(age: float, rate: float = 0.0, use_survival: bool = True) -> float:
    """Compute discount factor for a health/cost effect at a given age.

    Combines:
      - Pure time discount: 1/(1+r)^age  (time preference, uncertainty)
      - Survival probability: P(alive at age) (might not live long enough)

    Args:
        age: Age at which the effect occurs.
        rate: Annual pure time preference rate (e.g. 0.01 = 1%).
            Recommended: 0-1% for embryo selection context. Standard health
            economics uses 3%, but that answers a different question (resource
            allocation with a fixed budget today).
        use_survival: Whether to apply survival curve discounting.

    Returns:
        Multiplicative factor in [0, 1] to apply to undiscounted QALYs/costs.
    """
    factor = 1.0
    if rate > 0:
        factor *= 1.0 / (1 + rate) ** age
    if use_survival:
        factor *= survival_probability(age)
    return factor


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------


def liability_threshold_risk(pgs_z: float, prevalence: float, pgs_r2: float) -> float:
    """Convert a PGS z-score to absolute disease risk using the liability threshold model.

    The liability threshold model assumes an underlying continuous liability
    that is normally distributed, with disease occurring above a threshold.
    The PGS captures a fraction (r2) of that liability.

    Args:
        pgs_z: PGS z-score (in PGS SD units, i.e. how many SDs above mean PGS)
        prevalence: population lifetime prevalence of the disease
        pgs_r2: fraction of liability variance explained by the PGS

    Returns:
        Conditional absolute risk P(disease | PGS = pgs_z)
    """
    # Threshold on the liability scale corresponding to the prevalence
    threshold = norm.ppf(1 - prevalence)

    # The PGS shifts the mean of the liability distribution.
    # If PGS explains r2 of liability variance, then a 1-SD shift in PGS
    # corresponds to sqrt(r2) SDs on the liability scale.
    liability_shift = pgs_z * (pgs_r2**0.5)

    # Residual variance after conditioning on PGS
    residual_sd = (1 - pgs_r2) ** 0.5

    # P(liability > threshold | PGS) = P(N(shift, residual²) > threshold)
    return float(1 - norm.cdf((threshold - liability_shift) / residual_sd))


def compute_disease_impact(
    pgs_z: float,
    trait: DiseaseTrait,
    rate: float = 0.0,
    use_survival: bool = False,
    r2: float | None = None,
    rate_money: float | None = None,
) -> dict:
    """Compute QALY and savings impact for a disease trait given a PGS z-score."""
    r2 = trait.pgs_r2 if r2 is None else r2
    baseline_risk = trait.prevalence
    individual_risk = liability_threshold_risk(pgs_z, trait.prevalence, r2)

    risk_difference = individual_risk - baseline_risk

    df_q = discount_factor(trait.typical_onset_age, rate, use_survival)
    df_m = discount_factor(
        trait.typical_onset_age, rate if rate_money is None else rate_money, use_survival
    )

    # Positive = good: lower risk means positive QALY and savings
    qaly_delta = -risk_difference * trait.qaly_loss_if_affected * df_q
    savings = -risk_difference * trait.lifetime_cost_if_affected * df_m

    return {
        "trait": trait.display_name,
        "type": "disease",
        "pgs_z": pgs_z,
        "baseline_risk": baseline_risk,
        "individual_risk": individual_risk,
        "risk_difference": risk_difference,
        "qaly_delta": qaly_delta,
        "savings": savings,
        "discount_factor": df_q,
    }


def compute_continuous_impact(
    pgs_z: float,
    trait: ContinuousTrait,
    rate: float = 0.0,
    use_survival: bool = False,
    r2: float | None = None,
    rate_money: float | None = None,
) -> dict:
    """Compute QALY and savings impact for a continuous trait given a PGS z-score."""
    r2 = trait.pgs_r2 if r2 is None else r2
    trait_sd_shift = pgs_z * (r2**0.5)

    df_q = discount_factor(trait.typical_effect_age, rate, use_survival)
    df_m = discount_factor(
        trait.typical_effect_age, rate if rate_money is None else rate_money, use_survival
    )

    qaly_delta = trait_sd_shift * trait.qaly_per_sd * df_q
    savings = trait_sd_shift * trait.savings_per_sd * df_m

    return {
        "trait": trait.display_name,
        "type": "continuous",
        "pgs_z": pgs_z,
        "trait_sd_shift": trait_sd_shift,
        "qaly_delta": qaly_delta,
        "savings": savings,
        "discount_factor": df_q,
    }


def compute_all(
    scores: dict[str, float],
    rate: float = 0.0,
    rate_money: float | None = None,
    use_survival: bool = True,
    ancestry_ratio: float = 1.0,
) -> dict:
    """Compute QALY and cost impact for all provided trait scores.

    Args:
        scores: mapping of trait name -> PGS z-score
        rate: annual time-discount rate applied to QALYs (0 = none).
        rate_money: separate rate for $ savings; defaults to `rate`.
            Health-economics convention sometimes differs (e.g., UK
            NICE: 1.5% QALY, 3.5% costs).
        use_survival: whether to weight by survival-to-age probability.
        ancestry_ratio: multiply each trait's R² by this. See
            ANCESTRY_R2_RATIO for per-population values when the
            scored genome is non-EUR; default 1.0 = EUR.

    Returns:
        Dict with per-trait results and totals.
    """
    results = []

    for trait_name, pgs_z in scores.items():
        if trait_name in DISEASE_TRAITS:
            t = DISEASE_TRAITS[trait_name]
            results.append(
                compute_disease_impact(
                    pgs_z, t, rate, use_survival,
                    r2=t.pgs_r2 * ancestry_ratio, rate_money=rate_money,
                )
            )
        elif trait_name in CONTINUOUS_TRAITS:
            t = CONTINUOUS_TRAITS[trait_name]
            results.append(
                compute_continuous_impact(
                    pgs_z, t, rate, use_survival,
                    r2=t.pgs_r2 * ancestry_ratio, rate_money=rate_money,
                )
            )
        else:
            print(f"Warning: unknown trait '{trait_name}', skipping")

    total_qaly = sum(r["qaly_delta"] for r in results)
    total_savings = sum(r["savings"] for r in results)

    return {
        "traits": results,
        "total_qaly_delta": total_qaly,
        "total_savings": total_savings,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def format_qaly_results(results: dict) -> str:
    """Format results as a human-readable report."""
    lines = []
    lines.append("=" * 72)
    lines.append("GENOME VALUE CALCULATOR — QALY & Cost Impact Estimates")
    lines.append("=" * 72)
    lines.append("")
    lines.append("All values relative to population average. Positive = good.")
    lines.append("")

    # Disease traits
    disease_results = [r for r in results["traits"] if r["type"] == "disease"]
    if disease_results:
        lines.append("DISEASE RISK")
        lines.append("-" * 72)
        lines.append(
            f"  {'Trait':<25} {'PGS z':>7} {'Base risk':>10} {'Your risk':>10} {'ΔRisk':>8} {'QALYs':>8} {'Savings':>10}"
        )
        lines.append(f"  {'':<25} {'':>7} {'':>10} {'':>10} {'':>8} {'':>8} {'':>10}")
        for r in disease_results:
            lines.append(
                f"  {r['trait']:<25} {r['pgs_z']:>+7.2f} "
                f"{r['baseline_risk']:>9.1%} {r['individual_risk']:>9.1%} "
                f"{r['risk_difference']:>+7.1%} {r['qaly_delta']:>+8.3f} "
                f"${r['savings']:>+9,.0f}"
            )
        lines.append("")

    # Continuous traits
    cont_results = [r for r in results["traits"] if r["type"] == "continuous"]
    if cont_results:
        lines.append("CONTINUOUS TRAITS")
        lines.append("-" * 72)
        lines.append(
            f"  {'Trait':<25} {'PGS z':>7} {'Trait ΔSD':>10} {'QALYs':>8} {'Savings':>10}"
        )
        lines.append(f"  {'':<25} {'':>7} {'':>10} {'':>8} {'':>10}")
        for r in cont_results:
            lines.append(
                f"  {r['trait']:<25} {r['pgs_z']:>+7.2f} "
                f"{r['trait_sd_shift']:>+9.2f} {r['qaly_delta']:>+8.3f} "
                f"${r['savings']:>+9,.0f}"
            )
        lines.append("")

    # Totals
    lines.append("=" * 72)
    lines.append(f"  TOTAL QALYs gained: {results['total_qaly_delta']:>+.3f}")
    lines.append(f"  TOTAL savings:      ${results['total_savings']:>+,.0f}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("NOTE: These are rough estimates assuming trait independence,")
    lines.append("no time discounting, and population-average prevalence rates.")
    lines.append("See docstring for full list of simplifications.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Embryo selection simulation
# ---------------------------------------------------------------------------


def simulate_selection(
    n_embryos: int = 5,
    n_simulations: int = 100_000,
    seed: int = 42,
    use_correlations: bool = True,
    attenuate_rg: bool = True,
    exclude: list[str] | None = None,
    only: list[str] | None = None,
    rate: float = 0.0,
    rate_money: float | None = None,
    use_survival: bool = True,
    ancestry_ratio: float = 1.0,
) -> dict:
    """Simulate selecting the best embryo from n siblings by total QALY.

    Among full siblings, each trait's PGS varies due to Mendelian segregation.
    The within-family variance of PGS is ~0.5 of the population variance
    (half the additive genetic variance segregates within families).

    We draw n_embryos PGS z-scores per trait from a multivariate normal
    with the genetic correlation matrix, scaled by the within-family SD.
    Then compute total QALY for each embryo, pick the best, and compare
    to the sibling average.

    Args:
        use_correlations: If True, draw correlated PGS using the genetic
            correlation matrix. If False, draw independently (old behavior).
    """
    rng = np.random.default_rng(seed)
    rm = rate if rate_money is None else rate_money

    def _df(age: float) -> tuple[float, float]:
        d_q = discount_factor(age, rate, use_survival)
        return d_q, (d_q if rm == rate else discount_factor(age, rm, use_survival))


    all_traits_full = get_trait_order()

    if only:
        active_traits = [t for t in all_traits_full if t in only]
    elif exclude:
        active_traits = [t for t in all_traits_full if t not in exclude]
    else:
        active_traits = all_traits_full

    # We still draw from the full correlation matrix, but only score active traits
    all_traits = all_traits_full
    n_traits = len(all_traits)
    active_set = set(active_traits)

    # Within-family SD = sqrt(0.5) of population SD (which is 1 for z-scores)
    sibling_sd = np.sqrt(0.5)

    if use_correlations:
        # Covariance matrix for within-family PGS draws:
        # Cov = sibling_sd² * genetic_correlation_matrix
        rg = build_genetic_correlation_matrix()
        if attenuate_rg:
            a = np.ones(n_traits)
            for i, t in enumerate(all_traits):
                tr = DISEASE_TRAITS.get(t) or CONTINUOUS_TRAITS.get(t)
                if tr and tr.snp_h2 and tr.snp_h2 > 0:
                    a[i] = min(1.0, (tr.pgs_r2 / tr.snp_h2) ** 0.5)
            rg = rg * np.outer(a, a)
            np.fill_diagonal(rg, 1.0)
        cov = sibling_sd**2 * rg

        # Draw correlated PGS z-scores: shape (n_simulations * n_embryos, n_traits)
        flat_draws = rng.multivariate_normal(
            np.zeros(n_traits), cov, size=n_simulations * n_embryos
        )
        pgs_draws = flat_draws.reshape(n_simulations, n_embryos, n_traits)
    else:
        pgs_draws = rng.normal(0, sibling_sd, (n_simulations, n_embryos, n_traits))

    # Compute QALY contribution for each embryo for each trait
    # We vectorize this by computing QALYs for all draws at once
    qaly_matrix = np.zeros((n_simulations, n_embryos))

    for t_idx, trait_name in enumerate(all_traits):
        if trait_name not in active_set:
            continue
        z = pgs_draws[:, :, t_idx]  # shape (n_sims, n_embryos)

        if trait_name in DISEASE_TRAITS:
            trait = DISEASE_TRAITS[trait_name]
            r2 = trait.pgs_r2 * ancestry_ratio
            df_q, df_m = _df(trait.typical_onset_age)
            threshold = norm.ppf(1 - trait.prevalence)
            liability_shift = z * (r2**0.5)
            residual_sd = (1 - r2) ** 0.5
            risk = 1 - norm.cdf((threshold - liability_shift) / residual_sd)
            risk_diff = risk - trait.prevalence
            qaly_contrib = -risk_diff * trait.qaly_loss_if_affected * df_q
        else:
            trait = CONTINUOUS_TRAITS[trait_name]
            r2 = trait.pgs_r2 * ancestry_ratio
            df_q, df_m = _df(trait.typical_effect_age)
            trait_sd_shift = z * (r2**0.5)
            qaly_contrib = trait_sd_shift * trait.qaly_per_sd * df_q

        qaly_matrix += qaly_contrib

    # For each simulation: best embryo vs mean embryo
    best_qaly = np.max(qaly_matrix, axis=1)
    mean_qaly = np.mean(qaly_matrix, axis=1)
    gain = best_qaly - mean_qaly

    # Also compute savings matrix for the selected embryo
    savings_matrix = np.zeros((n_simulations, n_embryos))
    for t_idx, trait_name in enumerate(all_traits):
        if trait_name not in active_set:
            continue
        z = pgs_draws[:, :, t_idx]
        if trait_name in DISEASE_TRAITS:
            trait = DISEASE_TRAITS[trait_name]
            r2 = trait.pgs_r2 * ancestry_ratio
            df_q, df_m = _df(trait.typical_onset_age)
            threshold = norm.ppf(1 - trait.prevalence)
            liability_shift = z * (r2**0.5)
            residual_sd = (1 - r2) ** 0.5
            risk = 1 - norm.cdf((threshold - liability_shift) / residual_sd)
            risk_diff = risk - trait.prevalence
            savings_contrib = -risk_diff * trait.lifetime_cost_if_affected * df_m
        else:
            trait = CONTINUOUS_TRAITS[trait_name]
            r2 = trait.pgs_r2 * ancestry_ratio
            df_q, df_m = _df(trait.typical_effect_age)
            trait_sd_shift = z * (r2**0.5)
            savings_contrib = trait_sd_shift * trait.savings_per_sd * df_m
        savings_matrix += savings_contrib

    # Select same embryo (by QALY) for savings calculation
    best_idx = np.argmax(qaly_matrix, axis=1)
    best_savings = savings_matrix[np.arange(n_simulations), best_idx]
    mean_savings = np.mean(savings_matrix, axis=1)
    savings_gain = best_savings - mean_savings

    # Per-trait breakdown: expected z-score of selected embryo
    per_trait_z = {}
    for t_idx, trait_name in enumerate(all_traits):
        if trait_name not in active_set:
            continue
        selected_z = pgs_draws[np.arange(n_simulations), best_idx, t_idx]
        per_trait_z[trait_name] = float(np.mean(selected_z))

    # Per-trait single-trait selection: what if you only selected on this one trait?
    per_trait_solo = {}
    for t_idx, trait_name in enumerate(all_traits):
        if trait_name not in active_set:
            continue
        trait_z = pgs_draws[:, :, t_idx]  # (n_sims, n_embryos)

        if trait_name in DISEASE_TRAITS:
            trait = DISEASE_TRAITS[trait_name]
            df_q, df_m = _df(trait.typical_onset_age)
            # For diseases, lower PGS = lower risk = better, so select min
            best_solo_idx = np.argmin(trait_z, axis=1)
            selected_z = trait_z[np.arange(n_simulations), best_solo_idx]

            threshold = norm.ppf(1 - trait.prevalence)
            r2 = trait.pgs_r2 * ancestry_ratio
            liability_shift = selected_z * (r2**0.5)
            residual_sd_val = (1 - r2) ** 0.5
            risk = 1 - norm.cdf((threshold - liability_shift) / residual_sd_val)
            risk_diff = risk - trait.prevalence
            solo_qaly = float(np.mean(-risk_diff * trait.qaly_loss_if_affected * df_q))
            solo_savings = float(
                np.mean(-risk_diff * trait.lifetime_cost_if_affected * df_m)
            )
        else:
            trait = CONTINUOUS_TRAITS[trait_name]
            df_q, df_m = _df(trait.typical_effect_age)
            # Direction depends on sign of qaly_per_sd (e.g. BMI: lower is better)
            if trait.qaly_per_sd >= 0:
                best_solo_idx = np.argmax(trait_z, axis=1)
            else:
                best_solo_idx = np.argmin(trait_z, axis=1)
            selected_z = trait_z[np.arange(n_simulations), best_solo_idx]

            r2 = trait.pgs_r2 * ancestry_ratio
            trait_sd_shift = selected_z * (r2**0.5)
            solo_qaly = float(np.mean(trait_sd_shift * trait.qaly_per_sd * df_q))
            solo_savings = float(np.mean(trait_sd_shift * trait.savings_per_sd * df_m))

        per_trait_solo[trait_name] = {
            "mean_selected_z": float(np.mean(selected_z)),
            "qaly_gain": solo_qaly,
            "savings_gain": solo_savings,
        }

    return {
        "n_embryos": n_embryos,
        "n_simulations": n_simulations,
        "use_correlations": use_correlations,
        "rate": rate,
        "use_survival": use_survival,
        "qaly_gain_mean": float(np.mean(gain)),
        "qaly_gain_median": float(np.median(gain)),
        "qaly_gain_p10": float(np.percentile(gain, 10)),
        "qaly_gain_p90": float(np.percentile(gain, 90)),
        "savings_gain_mean": float(np.mean(savings_gain)),
        "savings_gain_median": float(np.median(savings_gain)),
        "per_trait_selected_z": per_trait_z,
        "per_trait_solo": per_trait_solo,
    }


def format_selection_results(results: dict) -> str:
    """Format embryo selection simulation results."""
    lines = []
    corr_label = (
        "with genetic correlations"
        if results.get("use_correlations", False)
        else "independent traits"
    )
    discount_parts = []
    if results.get("rate", 0) > 0:
        discount_parts.append(f"{results['rate']:.0%}/yr time discount")
    if results.get("use_survival", False):
        discount_parts.append("survival-adjusted")
    discount_label = ", ".join(discount_parts) if discount_parts else "no discounting"
    lines.append("=" * 72)
    lines.append(f"EMBRYO SELECTION: Best of {results['n_embryos']} siblings")
    lines.append(
        f"({results['n_simulations']:,} simulations, {corr_label}, {discount_label})"
    )
    lines.append("=" * 72)
    lines.append("")
    lines.append("COMBINED SELECTION (pick embryo with best total QALY):")
    lines.append("")
    lines.append(
        f"  QALYs gained:  {results['qaly_gain_mean']:>+.3f}  (median {results['qaly_gain_median']:>+.3f},"
        f"  10th-90th: {results['qaly_gain_p10']:>+.3f} to {results['qaly_gain_p90']:>+.3f})"
    )
    lines.append(
        f"  Savings:       ${results['savings_gain_mean']:>+,.0f}  (median ${results['savings_gain_median']:>+,.0f})"
    )
    lines.append("")
    lines.append("  PGS z-score of selected embryo (among siblings):")
    lines.append(f"    {'Trait':<25} {'Selected z':>10}")
    for trait_name, z in results["per_trait_selected_z"].items():
        display = DISEASE_TRAITS.get(trait_name) or CONTINUOUS_TRAITS[trait_name]
        lines.append(f"    {display.display_name:<25} {z:>+10.3f}")

    lines.append("")
    lines.append("-" * 72)
    lines.append("SINGLE-TRAIT SELECTION (what if you only selected for one trait?):")
    lines.append("")
    lines.append(f"  {'Trait':<25} {'Sel. z':>8} {'QALYs':>8} {'Savings':>10}")
    for trait_name, data in results["per_trait_solo"].items():
        display = DISEASE_TRAITS.get(trait_name) or CONTINUOUS_TRAITS[trait_name]
        lines.append(
            f"  {display.display_name:<25} {data['mean_selected_z']:>+8.3f}"
            f" {data['qaly_gain']:>+8.3f} ${data['savings_gain']:>+9,.0f}"
        )

    lines.append("")
    lines.append("All gains are relative to average sibling (not population).")
    lines.append("Positive = good. Assumes average parents, independent traits.")
    return "\n".join(lines)