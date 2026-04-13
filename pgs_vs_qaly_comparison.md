# PGS Catalog vs qaly_calculator.py: R² Comparison

Comparison of the `pgs_r2` values currently hardcoded in `qaly_calculator.py` against the best available scores in the PGS Catalog (searched April 2026).

Important: the catalog reports evaluation-set metrics (often AUROC/C-index, not R²), which aren't directly comparable to the liability-scale R² used in `qaly_calculator.py`. I convert where possible and flag caveats.

## Disease Traits

| Trait | qaly_calculator R² | Best PGS | Catalog Metric | Approx Liability R² | Notes |
|-------|-------------------|----------|----------------|---------------------|-------|
| **Heart disease** | 0.04 | PGS000013 (Khera 2018) | AUROC 0.921 | ~0.10-0.15 | qaly_calculator underestimates; 0.04 is closer to old C+T scores. But AUROC 0.92 likely includes covariates or favorable cohort. Realistic genetics-only R² probably 0.05-0.08 |
| **Type 2 diabetes** | 0.03 | PGS000014 (Khera 2018) | C-index 0.869 | ~0.06-0.10 | Similar story — 0.03 is conservative, probably 0.04-0.06 is fairer |
| **Alzheimer's** | 0.03 | PGS004285 (Ohta 2024) | AUROC 0.834 | ~0.06-0.10 | Dominated by APOE. 0.03 is too low if APOE is included. With APOE, genetics-only R² is ~0.06-0.08 |
| **Schizophrenia** | 0.05 | PGS000135 (Zheutlin 2019) | AUROC 0.74 | ~0.05-0.07 | 0.05 looks about right. AUROC ranges 0.60-0.74 across cohorts |
| **Depression** | 0.02 | PGS004885 (Jermy 2024) | C-index 0.62 | ~0.01-0.02 | 0.02 is reasonable, possibly slightly optimistic |
| **Atrial fibrillation** | 0.04 | PGS005168 (Roselli 2025) | C-index 0.872 | ~0.08-0.12 | 0.04 may be low. AF has relatively high heritability (~22% SNP h²). Newer PRS-CS scores are strong. But C-index 0.87 with covariates inflates this. Probably 0.04-0.06 genetics-only |
| **Breast cancer** | 0.08 | PGS000007 (Mavaddat 2018) | AUROC 0.780 | ~0.08-0.10 | 0.08 looks right — this is the well-validated PRS313. Catalog top entry (C-index=632) is a data entry error |
| **Prostate cancer** | 0.07 | PGS000662 (Conti 2021) | AUROC 0.961 | ~0.10-0.15 | AUROC 0.96-0.97 is inflated (snpnet UKB self-report eval). Realistic genetics-only is ~0.07-0.10. 0.07 is reasonable |
| **Stroke** | 0.015 | PGS002725 (Mishra 2022) | AUROC 0.765 | ~0.02-0.04 | 0.015 is slightly low. Stroke PGS has improved with MEGASTROKE. ~0.02-0.03 is fairer |
| **Colorectal cancer** | 0.03 | PGS003433 (Briggs 2022) | R² 28.5% (=0.285?) | unclear | R² = 28.5 in catalog is almost certainly a percentage on the observed scale, not liability. If Nagelkerke R², the liability R² is much lower. 0.03 is plausible for genetics-only |
| **Bipolar disorder** | 0.04 | PGS002786 (Gui 2022) | no eval metrics | — | No evaluation metrics deposited. 0.04 is consistent with PGC3 literature (~3-5% liability R²) |
| **Chronic kidney disease** | 0.02 | PGS005090 (Jones 2024) | AUROC 0.900 | ~0.06-0.10 | AUROC 0.90 almost certainly includes eGFR/clinical covariates. Genetics-only probably 0.02-0.03. 0.02 is reasonable |
| **Asthma** | 0.02 | PGS002248 (Kothalawala 2022) | AUROC 0.850 | ~0.04-0.06 | AUROC 0.85 with 105 SNPs feels high — likely includes age/sex covariates or favorable case definition. 0.02 is conservative but defensible |
| **IBD** | 0.04 | PGS001307 (Tanigawa 2022) | AUROC 0.896 | ~0.07-0.10 | snpnet with 809 SNPs, UKB eval. IBD has strong genetics (NOD2 etc). 0.04 might be slightly low — 0.05-0.06 may be fairer |
| **ADHD** | 0.04 | PGS002746 (Lahey 2022) | no eval metrics | — | No evaluation metrics deposited. Demontis 2023 PGC GWAS suggests ~3-5% liability R². 0.04 is consistent |
| **Type 1 diabetes** | 0.06 | PGS000024 (Sharp 2019) | AUROC 0.960 | ~0.15-0.25 | T1D is HLA-dominated (like APOE for Alzheimer's). AUROC 0.96 with 85 SNPs is largely HLA. qaly_calculator's 0.06 may be low — realistic genetics-only R² with HLA is ~0.10-0.15 |
| **Osteoporosis** | 0.05 | PGS001273 (Tanigawa 2022) | AUROC 0.858 | ~0.05-0.08 | snpnet with 316 SNPs, UKB eval. 0.05 is reasonable |
| **Anxiety disorders** | 0.01 | PGS005393 (Bugiga 2024) | R² 0.087 | ~0.01-0.02 | R² 0.087 is observed scale, probably with covariates. Liability R² for anxiety is very low. 0.01 is about right |

## Continuous Traits

| Trait | qaly_calculator R² | Best PGS | Catalog Metric | Notes |
|-------|-------------------|----------|----------------|-------|
| **Height** | 0.16 | PGS003895 (Tanigawa 2023) | R² 0.729 | qaly_calculator notes this: "PGS Catalog reports R²=0.717 for PGS001229 but that includes sex/age covariates in snpnet. Genetics-only R² is ~0.16." The 0.16 is the GWAS-based estimate of SNP heritability captured by PGS without covariates. This is correct for the liability-threshold/QALY calculation |
| **Cognitive ability** | 0.127 | PGS001232 (Tanigawa 2022) | R² 0.127 | Taken directly from catalog. But same caveat applies — this may include age/sex covariates in snpnet. Pure genetics-only R² might be ~0.05-0.07. This could be overstated |
| **BMI** | 0.08 | PGS005199 (Smit 2025) | R² 13.2% (=0.132) | Catalog value is on percentage scale. PRS-CSx cross-ancestry, so this is likely the best-case EUR evaluation. 0.08 is reasonable for genetics-only, maybe slightly low — 0.08-0.12 is the range |
| **Income** | 0.03 | no eval metrics | — | 3 scores deposited, none with evaluations. 0.03 is a rough estimate consistent with Hill 2019 |
| **Longevity** | 0.01 | no catalog entry | — | No "longevity" trait in catalog (only "life span" with 2 unevaluated scores). 0.01 is consistent with Timmers 2019 |
| **Subjective wellbeing** | 0.02 | PGS001090 (Tanigawa 2022) | AUROC 0.654 (N=264) | Very weak, tiny eval sample. 0.02 is a guess but in the right ballpark |

## Summary

Most of the `pgs_r2` values in `qaly_calculator.py` are reasonable and tend toward conservative, which is appropriate for a QALY calculator (you don't want to overstate predictive power when making real decisions).

The values I'd consider adjusting:

| Trait | Current R² | Suggested R² | Reason |
|-------|-----------|-------------|--------|
| **Type 1 diabetes** | 0.06 | 0.10-0.12 | HLA region gives T1D unusually high genetic predictability, similar to APOE for Alzheimer's |
| **Alzheimer's** | 0.03 | 0.06-0.08 | With APOE e4 included, discrimination is substantially better than 0.03 |
| **Cognitive ability** | 0.127 | 0.05-0.07 | Current value likely includes covariate adjustment in snpnet; genetics-only is lower |
| **IBD** | 0.04 | 0.05-0.06 | Strong known loci (NOD2, IL23R) give IBD above-average PGS performance |

Everything else is within a reasonable range given the uncertainty. The hardest numbers to pin down are the ones where catalog metrics are AUROC/C-index with unknown covariate adjustment — the AUROC-to-liability-R² conversion is sensitive to assumptions about prevalence and whether age/sex are baked into the evaluation.

### Traits with no or minimal catalog data

These traits have essentially no usable evaluation data in the PGS Catalog:
- **Bipolar disorder** (3 scores, 0 with metrics)
- **ADHD** (2 scores, 0 with metrics)
- **Income** (3 scores, 0 with metrics)
- **Longevity** (2 scores under "life span", 0 with metrics)
- **Subjective wellbeing** (tiny eval, unreliable)

For these, the `qaly_calculator.py` values are literature-based estimates that can't be validated against the catalog. They seem reasonable based on published GWAS/PGS papers but carry more uncertainty.
