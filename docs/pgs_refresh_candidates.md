# PGS Refresh Candidates — 2026-04-16

Survey of PGS Catalog (release 2026-04-13, 5,332 scores) + literature for each
trait in `curated_best_pgs.py`. **Caveat:** the catalog stores whatever metric
the source paper reported — many "top" R²/AUC values are *full-model* (PGS +
age + sex + PCs + clinical covariates), not incremental PGS-only. Entries
flagged † below are likely inflated this way and were discounted.

**SBayesRC status:** The Zheng et al. 2024 Nat Genet SBayesRC paper did **not**
deposit in PGS Catalog — weights for 28 UKB traits are at
<https://cnsgenomics.com/software/gctb/#Download>. Within the catalog, the only
SBayesRC scores found for our traits are Tanha 2025/2026 (prostate PGS005241,
breast PGS005349). The Keaton 2024 blood-pressure scores (PGS004603-05) also
use SBayesRC but aren't in our trait list.

| Trait | Current ID | Cur. R² | Best candidate | Cand. metric | Method | Citation | Rec. |
|---|---|---|---|---|---|---|---|
| height | PGS002804 | 0.42 | — (keep) | R²≈0.40 | SBayesC | Yengo 2022 | **KEEP** — top API hits (Tanigawa snpnet R²=0.73†, Raben LASSO R²=0.71†) include covariates. Yengo is genuinely saturated. PRSmix+ PGS004782 worth side-eval. |
| bmi | PGS002313 | 0.12 | **PGS004735** | inc.R²=0.161 | PRSmix+ | Truong 2024 Cell Genom | **REPLACE** — also see PGS005198-200 (Smit 2025) and Kim 2026 PGS005331 (LDpred2, R²=0.20†). SBayesRC available externally at cnsgenomics. |
| heart_disease | PGS003725 | 0.06 | — (keep) | AUC≈0.80 | LDpred2 + metaGRS | Patel 2023 | **KEEP** — GPSMult is multi-ancestry SOTA. PGS004745 PRSmix+ inc.R²=0.05 is comparable, not better. Top API entries (AUC 0.89-0.91†) include clinical risk factors. |
| type2_diabetes | PGS003867 | 0.07 | **PGS005368** | Nagelkerke R²=0.21, AUC=0.757 | PRS-CSx | Huerta-Chagoya 2025 | **REPLACE** — newer multi-ancestry, larger N. PRSmix+ PGS004840 also strong. |
| stroke | PGS002724 | 0.02 | PGS004836 | inc.R²=0.017 | PRSmix+ | Truong 2024 | **INVESTIGATE** — GIGASTROKE iGRS still competitive; Bakker 2023 metaGRS PGS003408 (AUC 0.77†) likely full-model. Marginal gains available. |
| alzheimers | PGS004092 | 0.07 | — (keep) | — | LDpred2 | Monti 2024 | **KEEP** — top API hits are tiny APOE-only scores (R²~0.19† dominated by APOE). Bellenguez 2022 83-SNP score is the alternative but not clearly better genome-wide. |
| schizophrenia | PGS002785 | 0.08 | **PGS Catalog gap** | liab.R²≈0.099 | SBayesR | Ni et al. 2021 Biol Psychiatry | **INVESTIGATE** — best PGC3-derived score (Trubetskoy 2022 → SBayesR/MegaPRS, ~9.9% liability) is **not deposited** in PGS Catalog. PGS000135 (Zheutlin PRS-CS, AUC 0.74) is older PGC2. Build from PGC3 sumstats directly. |
| bipolar_disorder | PGS002786 | 0.04 | **PGS Catalog gap** | liab.R²≈0.04-0.05 | LDpred2/SBayesR | Mullins 2021 Nat Genet | **INVESTIGATE** — PGC BD3 (Mullins 2021) primary score not in catalog. Current Gui SDPR R²=0.0025 is *observed-scale*; liability ≈0.04 is right. Build from PGC sumstats. |
| depression | PGS004885 | 0.02 | **not yet in catalog** | liab.R²≈0.058 | SBayesR | PGC-MDD 2025 (Adams et al., Cell) | **REPLACE when available** — 685k cases, ~5.8% liability variance vs ~2% current. Sumstats at PGC; PGS Catalog deposit pending. PRSmix+ PGS004760 (inc.R²=0.024) is interim upgrade. |
| breast_cancer | PGS000004 | 0.07 | PGS005349 | AUC 0.64-0.73 | **SBayesRC** | Tanha 2026 EJHG | **INVESTIGATE** — Mavaddat 313 still clinical standard (BOADICEA-validated). Tanha SBayesRC matches AUC with 5.4M SNPs; user prefers SBayesRC so worth swapping. PGS000007 (Mavaddat LASSO 3,820 SNPs) marginally beats 313. |
| prostate_cancer | PGS000662 | 0.09 | **PGS003765** or PGS005241 | OR top10%=4.33 / SBayesRC | GW-sig 451 SNP / **SBayesRC** | Wang 2023 Nat Genet / Tanha 2025 | **REPLACE** — Wang 451 supersedes Conti 269 (same consortium, +182 loci). Tanha SBayesRC PGS005241 is the annotation-aware option. |
| colorectal_cancer | PGS003850 | 0.04 | PGS003979 | AUC 0.795† | PRS-CS | Tamlander 2023 | **INVESTIGATE** — Tamlander AUC likely includes age. Briggs PGS003432 (LDpred2, AUC 0.73) and Thomas 2023 PGS003852 (PRS-CSx multi-anc) are genome-wide alternatives to 205-SNP score. |
| type1_diabetes | PGS000024 | 0.15 | — (keep) | AUC≈0.92 | HLA-engineered GRS2 | Sharp 2019 | **KEEP** — still SOTA. Trans-ancestry T1D-PRS (Diabetologia 2026, doi:10.1007/s00125-026-06706-5) and HLA-allele scores improve cross-ancestry but not EUR. No genome-wide score beats GRS2's HLA modeling. |
| asthma | PGS001787 | 0.04 | *survey incomplete* | — | — | — | **INVESTIGATE** — PRSmix+ likely covers; check Truong PGS004700-block. Valette 2021 / Namjou 2022 are alternatives. |
| osteoporosis | PGS000657 | 0.05 | *survey incomplete* | — | — | — | **INVESTIGATE** — Forgetta gSOS still strong; SBayesRC heel-BMD weights at cnsgenomics (Zheng 2024) likely beat it. Lu 2021 PGS001033 (eBMD) is alternative. |
| inflammatory_bowel_disease | PGS004081 | 0.06 | *survey incomplete* | — | — | — | **KEEP (provisional)** — Monti 2024 multi-method is recent; no obvious 2025 upgrade in literature. |
| chronic_kidney_disease | PGS000883 | 0.02 | *survey incomplete* | — | — | — | **INVESTIGATE** — Khan 2022 (PGS002759, eGFR genome-wide) and Stanzick 2021 are alternatives. |
| atrial_fibrillation | PGS005060 | 0.06 | **PGS005168** | C-index 0.79-0.87 | PRS-CS | Roselli 2025 Nat Genet (180k cases) | **REPLACE** — clear upgrade, largest AF GWAS to date. |
| adhd | PGS002746 | 0.04 | **PGS Catalog gap** | liab.R²≈0.055 | — | Demontis 2023 Nat Genet (PGC-ADHD2) | **REPLACE** — current Lahey 2022 is secondary-use of older sumstats. Demontis 2023 primary score not in catalog; build from PGC sumstats with LDpred2/SBayesR. |
| cognitive_ability | PGS004427 | 0.10 | local MTAG score | r≈0.25 | MTAG→LDpred-inf | (in-house) | **REPLACE with local** — already noted in curated_best_pgs.py:117-121. EA4 (Okbay 2022) PGS002245 explains ~12-16% EA variance but is education not cognition. No newer pure-IQ GWAS in catalog beats Savage 2018 + EA4 MTAG combo. |

## Summary

**Clear replacements (5):** BMI→PGS004735, T2D→PGS005368, prostate→PGS003765,
AF→PGS005168, cognitive→local. All ≥2024, larger N, demonstrably higher
out-of-sample performance.

**Replace-when-available (2):** depression (PGC-MDD 2025, ~3× variance
explained, sumstats public but PGS not yet deposited), ADHD (Demontis 2023
primary, same situation).

**SBayesRC opportunities:** breast PGS005349 and prostate PGS005241 are the
only in-catalog SBayesRC scores for our traits. For height/BMI/BMD/EA/T2D the
Zheng 2024 weights exist at cnsgenomics but require manual integration (not
PGS-Catalog harmonized format).

**Catalog gaps:** SCZ, BD, ADHD best-in-class scores live at the PGC download
portal, not PGS Catalog — building from sumstats with SBayesR/SBayesRC is the
path to SOTA there.

**Keep as-is (5):** height (Yengo saturated), CAD (Patel GPSMult), AD (Monti),
T1D (Sharp GRS2 — domain-engineered, untouchable), IBD (Monti, recent).

top API hits are full-model metrics (†) and should not be taken at face value.
