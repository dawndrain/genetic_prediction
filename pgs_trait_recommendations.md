# PGS Score Recommendations by Trait

Scores sourced from the [PGS Catalog](https://www.pgscatalog.org/) REST API, April 2026. Ranked by best reported evaluation metric (R² for continuous traits, AUROC/C-index for binary traits). All scores evaluated in European-ancestry cohorts unless noted.

## Height

**Recommendation: PGS003895** (Tanigawa Y, 2023) — R² = 0.729, 62,419 variants, snpnet

Height is the most well-studied PGS trait and the top R²-based scores are tightly clustered. PGS003895 is a minor update to PGS001229 (which this project already uses) from the same group, gaining ~1 percentage point of variance explained with more variants.

| Rank | PGS ID | Metric | Value | Variants | Method | Publication | Eval Cohort (N) |
|------|--------|--------|-------|----------|--------|-------------|-----------------|
| 1 | PGS003895 | R² | 0.729 | 62,419 | snpnet | Tanigawa Y (2023) | UKB (2,885) |
| 2 | PGS001229 | R² | 0.717 | 51,209 | snpnet | Tanigawa Y (2022) | UKB (67,298) |
| 3 | PGS004212 | R² | 0.712 | 27,779 | LASSO | Raben TG (2023) | UKB (45,334) |
| 4 | PGS004214 | R² | 0.712 | 23,686 | LASSO | Raben TG (2023) | UKB (45,334) |
| * | PGS000758 | AUROC | 0.861 | 33,938 | LASSO on P+T | Lu T (2021) | ALSPAC (941) |

PGS000758 tops the raw ranking because it reports AUROC (binary tall/short classification) rather than R², so it's not directly comparable. Among R²-based scores, PGS003895 edges out PGS001229, but note the much smaller evaluation sample (2,885 vs 67,298) — the improvement may not be robust. PGS001229 is a safe choice with the largest validation cohort.

**Practical pick: stick with PGS001229** unless you want to test PGS003895 for a marginal gain. The difference is small and PGS001229 has stronger validation.

---

## Coronary Artery Disease (Heart Disease)

**Recommendation: PGS000013** (Khera AV, 2018) — AUROC = 0.921, 6,630,150 variants, LDpred

This is from the landmark Khera et al. 2018 Nature Genetics paper that demonstrated genome-wide PRS can identify individuals at risk equivalent to monogenic mutations. It remains one of the most widely validated CAD scores.

| Rank | PGS ID | Metric | Value | Variants | Method | Publication | Eval Cohort (N) |
|------|--------|--------|-------|----------|--------|-------------|-----------------|
| 1 | PGS000013 | AUROC | 0.921 | 6,630,150 | LDpred | Khera AV (2018) | MGI (90,053) |
| 2 | PGS001780 | AUROC | 0.913 | 1,090,048 | PRS-CS | Tamlander M (2022) | FinnGen (291,720) |
| 3 | PGS000018 | AUROC | 0.890 | 1,745,179 | metaGRS | Inouye M (2018) | MHI (3,309) |
| 4 | PGS000200 | C-index | 0.859 | 28 | GWAS-sig SNPs | Tikkanen E (2013) | FINRISK (24,124) |
| 5 | PGS000329 | C-index | 0.832 | 6,423,165 | LDpred | Mars N (2020) | FINRISK (20,165) |

PGS001780 is a strong alternative — validated in FinnGen's massive cohort (292K) with nearly equivalent AUROC, using fewer variants (1.1M vs 6.6M) which makes it more practical to compute. PGS000013's 6.6M variants will require substantial imputation coverage.

PGS000200 is notable: only 28 GWAS-significant SNPs yet achieves C-index 0.859. If you want a quick-and-dirty score that doesn't need imputation, this is interesting, though it measures something slightly different (long-term risk prediction with covariates).

**Practical pick: PGS001780** may be the better balance of performance vs. computational cost. If you can handle 6.6M variants, PGS000013 is the gold standard.

---

## Type 2 Diabetes

**Recommendation: PGS000014** (Khera AV, 2018) — C-index = 0.869, 6,917,436 variants, LDpred

| Rank | PGS ID | Metric | Value | Variants | Method | Publication | Eval Cohort (N) |
|------|--------|--------|-------|----------|--------|-------------|-----------------|
| 1* | PGS000031 | C-index | 0.906 | 62 | GWAS-sig SNPs | Vassy JL (2014) | FOS (3,471) |
| 2 | PGS004859 | AUROC | 0.880 | 1,108,235 | PRS-CS | Deutsch AJ (2023) | AllofUs (323) |
| 3 | PGS000014 | C-index | 0.869 | 6,917,436 | LDpred | Khera AV (2018) | UKB (68,229) |
| 4 | PGS005363 | AUROC | 0.860 | 946,015 | PRS-CSx | Huerta-Chagoya A (2025) | SP2 (1,660) |
| 5 | PGS005364 | AUROC | 0.860 | 980,079 | PRS-CSx | Huerta-Chagoya A (2025) | SP2 (1,660) |

PGS000031 tops the ranking with C-index 0.906 but uses only 62 GWAS-significant SNPs — this almost certainly includes clinical covariates (age, BMI, family history) in the C-index calculation, inflating the metric relative to genetics-only scores.

PGS004859 (Deutsch AJ, 2023) reports AUROC 0.88 with PRS-CS but was evaluated on only 323 people in AllofUs — too small to trust.

PGS000014 is from the same Khera 2018 paper as the CAD score, validated in 68K UKB participants. It's the most robust genome-wide score here.

The Huerta-Chagoya 2025 scores (PGS005363/5364) use PRS-CSx, which is designed for cross-ancestry portability — worth considering if your subjects are non-European.

**Practical pick: PGS000014** for European ancestry. PGS005363 for multi-ancestry applications.

---

## Alzheimer's Disease

**Recommendation: PGS004285** (Ohta R, 2024) — AUROC = 0.834, 20 variants, GenoBoost

| Rank | PGS ID | Metric | Value | Variants | Method | Publication | Eval Cohort (N) |
|------|--------|--------|-------|----------|--------|-------------|-----------------|
| 1* | PGS000945 | AUROC | 0.985 | 26 | snpnet | Tanigawa Y (2022) | UKB (6,497) |
| 2* | PGS001348 | AUROC | 0.967 | 15 | snpnet | Tanigawa Y (2022) | UKB (6,497) |
| 3* | PGS001349 | AUROC | 0.962 | 6 | snpnet | Tanigawa Y (2022) | UKB (6,497) |
| 4 | PGS004285 | AUROC | 0.834 | 20 | GenoBoost | Ohta R (2024) | UKB (67,428) |
| 5 | PGS004286 | AUROC | 0.832 | 10 | GenoBoost | Ohta R (2024) | UKB (67,428) |

The Tanigawa scores (PGS000945/1348/1349) report implausibly high AUROCs (0.96-0.99) with very few SNPs. These are almost certainly driven by APOE e4 variants and evaluated in a way that inflates discrimination (e.g., prevalent cases in UKB, which skews toward late-onset diagnosed cases). An AUROC of 0.985 for any polygenic score would be extraordinary — treat with skepticism.

The GenoBoost scores (Ohta R, 2024) report more realistic AUROCs (~0.83) evaluated in a much larger sample (67K). GenoBoost is a boosting method that models dominance effects, which matters for APOE (the e4 allele has a partially dominant risk effect).

Alzheimer's PGS is unusual in that a single locus (APOE) dominates. All these scores will be mostly measuring APOE status, with small contributions from other loci.

**Practical pick: PGS004285.** Most realistic evaluation, largest validation sample, and the method handles APOE's dominance pattern.

---

## Schizophrenia

**Recommendation: PGS000135** (Zheutlin AB, 2019) — AUROC = 0.74, 972,439 variants, PRS-CS

| Rank | PGS ID | Metric | Value | Variants | Method | Publication | Eval Cohort (N) |
|------|--------|--------|-------|----------|--------|-------------|-----------------|
| 1 | PGS000135 | AUROC | 0.740 | 972,439 | PRS-CS | Zheutlin AB (2019) | BioMe (9,569) |
| 2 | PGS000136 | AUROC | 0.640 | 833,502 | PRS-CS | Zheutlin AB (2019) | PHB (18,461) |
| 3 | PGS000133 | AUROC | 0.600 | 604,645 | PRS-CS | Zheutlin AB (2019) | BioVU (33,694) |
| 4 | PGS000134 | AUROC | 0.600 | 830,589 | PRS-CS | Zheutlin AB (2019) | MyCode (44,436) |

All four evaluated scores are from the same paper (Zheutlin AB, 2019) using PRS-CS, just evaluated in different biobank cohorts. The variation in AUROC (0.60-0.74) reflects differences in case ascertainment and cohort composition, not method differences.

PGS000135 has the highest AUROC, evaluated in BioMe (Mount Sinai's biobank). The gap between 0.74 (BioMe) and 0.60 (BioVU/MyCode) is striking — BioMe's more diverse population and potentially different phenotyping may contribute. Only 5 schizophrenia scores exist in the catalog total, which is surprisingly few given the extensive GWAS literature. Newer PGC3-based scores may not have been deposited yet.

**Practical pick: PGS000135**, but expect real-world discrimination closer to AUROC 0.60-0.65 based on the other cohort evaluations.

---

## Major Depressive Disorder

**Recommendation: PGS004885** (Jermy B, 2024) — C-index = 0.62, 801,544 variants, MegaPRS

| Rank | PGS ID | Metric | Value | Variants | Method | Publication | Eval Cohort (N) |
|------|--------|--------|-------|----------|--------|-------------|-----------------|
| 1 | PGS004885 | C-index | 0.620 | 801,544 | megaprs.auto | Jermy B (2024) | GS:SFHS (7,018) |
| 2 | PGS000138 | AUROC | 0.561 | 22,274 | C+T (Ricopili) | Cai N (2020) | TwinGene (36,709) |
| 3 | PGS000139 | AUROC | 0.549 | 21,980 | C+T (Ricopili) | Cai N (2020) | TwinGene (36,709) |
| 4 | PGS000145 | AUROC | 0.525 | 21,510 | C+T (Ricopili) | Cai N (2020) | TwinGene (36,709) |
| 5 | PGS000767 | R² | 0.095 | 14 | GWAS-sig SNPs | Guffanti G (2019) | N=73 |

Depression is the hardest trait here. MDD has a SNP heritability of only ~10%, phenotyping is noisy (self-report vs clinical interview vs ICD codes all capture different things), and environmental factors dominate. An AUROC of 0.56 is barely above chance (0.50).

PGS004885 is the newest and uses MegaPRS (an ensemble method), achieving C-index 0.62 — modest but the best available. The Cai 2020 scores (PGS000138/139/145) use older clumping+thresholding and perform near chance.

PGS000767 reports R² = 0.095 but was evaluated on 73 people — statistically meaningless.

**Practical pick: PGS004885**, but set expectations accordingly. This score is more useful for population-level stratification than individual prediction.

---

## Cognitive Ability

**Recommendation: PGS001232** (Tanigawa Y, 2022) — R² = 0.127, 10,055 variants, snpnet

| Rank | PGS ID | Metric | Value | Variants | Method | Publication | Eval Cohort (N) |
|------|--------|--------|-------|----------|--------|-------------|-----------------|
| 1 | PGS001232 | R² | 0.127 | 10,055 | snpnet | Tanigawa Y (2022) | UKB (10,867) |
| 2 | PGS003724 | R² | 0.121 | 6,683,248 | SBLUP | Hatoum AS (2022) | COL_Twin (916) |

Only 2 scores with evaluation metrics in the catalog (out of 6 total). PGS001232 and PGS003724 perform nearly identically (~R² 0.12-0.13), but PGS001232 achieves this with 10K variants vs 6.7M — dramatically more efficient. The UKB evaluation (fluid intelligence test) is also a larger, more reliable cohort than the twin sample.

Note that "cognitive function" in UKB is measured by a brief fluid intelligence task (13 logic/reasoning questions in 2 minutes). This is a narrow proxy for general cognitive ability. R² of ~0.13 is consistent with the broader literature on IQ/education PGS.

**Practical pick: PGS001232.** Far fewer variants with equivalent performance.

---

## Summary Table

| Trait | Recommended PGS | Metric | Value | Variants | Practical Notes |
|-------|----------------|--------|-------|----------|-----------------|
| Height | PGS001229 (current) | R² | 0.717 | 51,209 | Already in use; PGS003895 is marginal upgrade |
| Heart disease | PGS000013 or PGS001780 | AUROC | 0.92 / 0.91 | 6.6M / 1.1M | PGS001780 if 6.6M variants is too heavy |
| Type 2 diabetes | PGS000014 | C-index | 0.869 | 6,917,436 | Same paper as CAD score (Khera 2018) |
| Alzheimer's | PGS004285 | AUROC | 0.834 | 20 | Mostly APOE; realistic eval unlike top-ranked |
| Schizophrenia | PGS000135 | AUROC | 0.740 | 972,439 | Expect ~0.60-0.65 in practice |
| Depression | PGS004885 | C-index | 0.620 | 801,544 | Low heritability limits prediction |
| Cognitive ability | PGS001232 | R² | 0.127 | 10,055 | Measures fluid intelligence specifically |
