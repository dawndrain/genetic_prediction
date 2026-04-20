I had Claude write this essay for me based on a long conversation where I asked many questions about genetics.
(In general I really like this workflow! Ask a hundred questions about whatever I'm interested in, and then have Claude write a doc summarizing everything.)


# Polygenic Prediction for Embryo Selection: A Technical Guide

## 1. Background: Why Genetics Is Surprisingly Linear and Heritable

### 1.1 The Genome at a Glance

The human genome consists of about 3 billion base pairs of DNA, packaged into 23 pairs of chromosomes. Each person carries two copies of the genome — one inherited from each parent. At most positions, all humans share the same nucleotide. But at roughly 10 million positions, there is common variation: some people have an A, others a G. These variable sites are called single-nucleotide polymorphisms (SNPs). An individual's genotype at a SNP is coded as 0, 1, or 2 — the count of the "alternative" allele across their two chromosome copies.

Modern genotyping arrays (like those used by 23andMe) measure ~600,000–700,000 selected SNPs. The remaining common variants can be imputed with high accuracy ($r^2 > 0.95$ for common variants) by matching an individual's observed genotypes to haplotype reference panels using hidden Markov models, yielding ~7 million total common variants.

### 1.2 Heritability: Higher Than Most People Expect

Twin studies compare identical twins (who share 100% of their DNA) with fraternal twins (who share ~50%). For many traits, the numbers are striking:

| Trait | Twin heritability |
|-------|------------------|
| Height | ~0.80 |
| BMI | ~0.70 |
| Cognitive ability (adults) | ~0.50–0.80 |
| Schizophrenia | ~0.80 |
| Major depression | ~0.35–0.40 |
| Type 2 diabetes | ~0.50–0.65 |
| Coronary artery disease | ~0.40–0.60 |

Most people substantially underestimate these numbers. The common intuition — that intelligence, personality, and health are primarily shaped by parenting, schooling, and environment — is not supported by the data. Environment matters, but genetic variation explains a large (often dominant) share of individual differences within a population.

An important caveat: heritability measures how much of the *variation* between individuals is attributable to genetic differences, not how much of the *trait* is "determined" by genes. In a hypothetical world where everyone ate the same diet, the heritability of BMI would approach 1.0 — not because diet doesn't matter, but because there would be no environmental variation left.

### 1.3 Linkage Disequilibrium: Why Nearby SNPs Are Correlated

When DNA is passed from parent to child, chromosomes undergo recombination: during meiosis, paired chromosomes exchange segments at 1–3 random crossover points per chromosome. Over many generations, this shuffling breaks up the association between distant variants. But nearby variants (within a few hundred kilobases) have rarely been separated by recombination in the history of a population, so they tend to be inherited together. This non-random association between alleles at nearby loci is called linkage disequilibrium (LD).

LD is quantified by the correlation $r$ between genotypes at two SNPs across a population. Nearby SNPs can have $|r| > 0.9$; SNPs >1 Mb apart typically have $|r| \approx 0$. LD is why a genotyping array measuring 600K SNPs effectively "tags" information about millions of variants — and why analyses must carefully account for correlations between SNPs.

### 1.4 Additivity: The Surprise That Makes Polygenic Prediction Work

Perhaps the most important empirical finding in quantitative genetics is that, for most complex traits, genetic effects are overwhelmingly additive. Each allele contributes a roughly constant increment to the trait, regardless of what alleles are present at other loci or even at the same locus. Dominance (interaction between alleles at the same locus) and epistasis (interaction between alleles at different loci) exist, but they account for a small fraction of genetic variance for most traits.

This is surprising. One might expect that a biological system as complex as human development would be full of nonlinear gene-gene interactions. And at the molecular level, there are abundant interactions — gene regulation is highly nonlinear. But at the level of trait variation in a population, these nonlinearities mostly average out. The theoretical reasons for this are debated, but the empirical fact is well-established and is the foundation for polygenic scores: if effects are additive, a simple weighted sum $\text{PGS} = \sum_j w_j g_j$ over genotypes $g_j$ is close to the best possible predictor.

## 2. From Genome-Wide Association Studies to Polygenic Scores

### 2.1 GWAS: One SNP at a Time

A genome-wide association study (GWAS) tests each SNP independently for association with a trait. For SNP $j$, the marginal effect estimate is:

$$\hat{\beta}_j = (x_j^\top x_j)^{-1} x_j^\top y$$

where $x_j$ is the genotype vector and $y$ is the phenotype vector across $N$ individuals. This is just a univariate regression — conceptually, asking "do people with more copies of this allele tend to have higher or lower values of the trait?"

The problem is that SNPs are correlated via LD. If the true model is $y = X\beta + \varepsilon$ with joint effects $\beta$, then the marginal estimate for SNP $j$ picks up contributions from every correlated SNP:

$$\hat{\beta}_j \approx \sum_k R_{jk} \beta_k$$

In matrix form: $\hat{\beta} \approx R\beta$, where $R$ is the LD correlation matrix. The marginal estimate for a non-causal SNP that happens to be near a causal one will be nonzero — the signal is "smeared out" by LD. Using marginal betas directly as PGS weights would double-count correlated signal.

Modern GWAS are massive. The largest meta-analyses combine data from UK Biobank (~500K), FinnGen (~500K), 23andMe (millions of customers), the Million Veteran Program (~636K), and other cohorts, reaching effective sample sizes of 1–3 million for some traits. These meta-analyses produce summary statistics (marginal betas, standard errors, p-values) that are publicly shared — enabling PGS construction without access to individual-level data.

### 2.2 The Direct Approach: Penalized Regression on Individual-Level Data

If you have access to the full genotype and phenotype data for all individuals (as within a single biobank), the most natural approach is to fit a joint regression directly:

$$\min_\beta \|y - X\beta\|^2 + \lambda \|\beta\|_1$$

This is the Lasso, which simultaneously selects variables and estimates their joint effects, automatically handling LD through the design matrix $X$. The penalty $\lambda$ controls sparsity — larger values yield sparser models with fewer nonzero coefficients. Cross-validation selects the optimal $\lambda$.

The BASIL/snpnet algorithm implements this efficiently for biobank-scale data (~millions of variants, hundreds of thousands of individuals). It achieves competitive prediction accuracy: e.g., $R^2 \approx 0.18$ for height in UK Biobank.

The advantage of direct penalized regression is conceptual clarity: you're solving the actual prediction problem in one step, with LD handled implicitly through the joint model. The disadvantage is that it requires individual-level data, which limits you to a single biobank's sample size. You cannot easily combine signal from UKB + FinnGen + 23andMe without having all the individual data in one place.

### 2.3 The Summary Statistics Approach: Bayesian LD Deconvolution

The dominant paradigm for PGS construction uses only GWAS summary statistics ($\hat{\beta}$) and an LD reference matrix ($R$), without requiring individual-level data. The core problem: given $\hat{\beta} \approx R\beta$ and some prior $p(\beta)$, estimate the posterior $p(\beta | \hat{\beta}, R)$.

This is valuable because summary statistics from massive multi-cohort meta-analyses (N > 1M) are freely available, whereas individual-level data from those same meta-analyses is not.

#### Clumping + Thresholding (C+T): The Baseline

The simplest approach. Select the most significant SNP in each LD block, discard everything correlated with it (clumping), and keep only SNPs with p-value below some threshold. PGS weights are the raw marginal betas $\hat{\beta}_j$ for the retained SNPs. Because correlated SNPs have been removed, double-counting is approximately handled.

C+T is easy to implement but wasteful — it discards information from sub-threshold variants and handles LD crudely. It typically explains 50–70% as much variance as Bayesian methods.

#### LDpred

Uses a spike-and-slab prior: each SNP is causal with probability $p$, and if causal, $\beta_j \sim N(0, h^2_{SNP}/(Mp))$ where $M$ is the number of SNPs. Gibbs sampling cycles through SNPs, updating each conditional on the current estimates of all others and the LD structure. The special case $p = 1$ (LDpred-inf) is equivalent to ridge regression on the summary statistics.

#### PRS-CS

Uses a continuous shrinkage prior (global-local framework related to the horseshoe). Each SNP's effect has a local variance parameter, so effects are continuously shrunk toward zero rather than discretely set to zero. PRS-CS-auto learns the global shrinkage from the data.

#### SBayesR

Uses a mixture-of-normals prior: $\beta_j \sim \sum_{k=1}^{K} \pi_k N(0, \sigma^2_k)$ with components for zero, small, medium, and large effects. A single set of mixing proportions applies to all SNPs. Operates on summary statistics with a sparse block-diagonal approximation to $R$.

#### Performance Comparison

All Bayesian methods substantially outperform C+T. Among Bayesian methods, differences are modest — typically within a few percent of each other in prediction $R^2$. The ranking varies by trait. For height, the approximate hierarchy is:

| Method | Approximate $R^2$ | Relative to C+T |
|--------|-------------------|-----------------|
| C+T | 0.18 | 1.0x |
| LDpred2 | 0.23 | 1.28x |
| PRS-CS | 0.24 | 1.33x |
| SBayesR | 0.24 | 1.33x |
| SBayesRC (with annotations) | 0.27 | 1.50x |

The big jump is from C+T to any Bayesian method. Among Bayesian methods, the improvement from adding functional annotations (SBayesRC) is the largest remaining gain.

### 2.4 SBayesRC: Incorporating Functional Annotations

SBayesRC extends SBayesR by making the mixture proportions SNP-specific via a 96-dimensional annotation vector $\mathbf{a}_j$:

$$\pi_{jk} \propto \exp(\mathbf{a}_j^\top \gamma_k)$$

This is a softmax regression from annotations to mixture component membership. The 96 annotations (from the Baseline-LD model) include:

- **Coding annotations:** Is the SNP in a protein-coding exon? Nonsynonymous? Synonymous?
- **Conservation:** Is this position conserved across species? (GERP scores, PhastCons elements)
- **Regulatory/chromatin:** Does this overlap histone marks (H3K4me1, H3K4me3, H3K27ac) indicating enhancers, promoters, or active chromatin? (from ENCODE/Roadmap Epigenomics)
- **Open chromatin:** DNase I hypersensitive sites where DNA is accessible to transcription factors
- **Transcription factor binding sites:** Known TFBS from ChIP-seq experiments
- **LD/MAF features:** Level of LD, minor allele frequency, and their interactions

The $\gamma_k$ parameters are estimated jointly with SNP effects via MCMC. A nonsynonymous variant in a conserved coding region gets a higher prior probability of being in the "large effect" component than a random intergenic variant.

SBayesRC improves prediction $R^2$ by ~14% within European ancestry and up to ~34% for cross-ancestry prediction. It only requires summary statistics and an LD reference. The code is open source (GitHub: zhilizheng/SBayesRC).

### 2.5 Cross-Ancestry Methods

Polygenic scores trained on European-ancestry GWAS transfer poorly to other populations — prediction accuracy drops by 40–80% in African-ancestry populations, less so in East Asian and South Asian populations. This is because LD patterns differ across populations: a tagging variant that is in high LD with a causal variant in Europeans may not be in LD with it in Africans. Functional annotations help (since causal variants themselves are shared), which is why SBayesRC shows the largest cross-ancestry gains.

Dedicated cross-ancestry methods like PRS-CSx extend PRS-CS to jointly model summary statistics from multiple ancestry-specific GWAS, learning shared and population-specific effects. PolyPred-S combines predictions from functionally-informed models across populations. These approaches narrow the gap but do not close it — the fundamental problem is insufficient non-European GWAS sample sizes.

## 3. Scaling of Prediction Accuracy and Missing Heritability

### 3.1 The Missing Heritability Problem

Twin studies estimate that cognitive ability has heritability ~0.50–0.80, yet the best current PGS explains ~0.16 of the variance (Herasight's CogPGT) or ~0.07 from published academic scores. The gap between twin heritability and PGS $R^2$ is the "missing heritability."

Several factors contribute:

- **Finite GWAS sample size:** The most important factor. Many causal variants with small effects haven't reached statistical significance yet.
- **Imperfect tagging:** Common genotyping arrays and imputation don't capture all causal variation, especially rare variants (MAF < 1%) and structural variants.
- **Non-additive effects:** Dominance and epistasis contribute to twin heritability but not to additive PGS (though this appears to be a small fraction).
- **Gene-environment correlation:** Twins share environments correlated with their genotypes, inflating twin-based heritability relative to the portion capturable by common SNPs.

### 3.2 How PGS Accuracy Scales with Sample Size

PGS $R^2$ increases with GWAS training sample size $N$ as:

$$R^2 \approx h^2_{SNP} \cdot \frac{N}{N + M_{eff}/h^2_{SNP}}$$

where $h^2_{SNP}$ is the SNP heritability (the fraction of variance captured by all common SNPs) and $M_{eff} \approx 60{,}000$ is the effective number of independent SNP blocks.

This formula reveals why some traits are harder to predict:

- **Highly heritable traits** ($h^2_{SNP}$ large): The half-saturation sample size $N_{1/2} = M_{eff}/h^2_{SNP}$ is smaller, so you reach good prediction with fewer samples. Height ($h^2_{SNP} \approx 0.5$) needs $N_{1/2} \approx 120K$ — achievable with current data.

- **Less heritable / highly polygenic traits** ($h^2_{SNP}$ small or $M_{eff}$ effectively large): The half-saturation point is much larger. Depression ($h^2_{SNP} \approx 0.05$) needs $N_{1/2} \approx 1.2M$.

The growth in GWAS sample sizes over time tells the story:

| Year | Largest cognitive GWAS $N$ | Best PGS $R^2$ for cognition |
|------|---------------------------|------------------------------|
| 2014 | ~18K | ~0.01 |
| 2017 | ~80K | ~0.03 |
| 2018 | ~270K | ~0.05 |
| 2022 | ~3M (educational attainment) | ~0.10–0.12 |
| 2025 | UKB with phenotype engineering | ~0.16 (Herasight CogPGT) |

The trajectory continues upward as sample sizes grow and methods improve, but the rate of gain is decelerating as we move up the saturation curve.

### 3.3 Why Some Traits Have Tiny $R^2$

Traits like chronic pain or insomnia have PGS $R^2 \approx 0.01$. This reflects a combination of factors:

- **Low SNP heritability:** Environmental influences dominate.
- **Phenotype measurement noise:** Self-reported "do you have chronic pain?" is far noisier than a stadiometer reading of height. Measurement error attenuates both heritability estimates and GWAS effect sizes.
- **Phenotypic heterogeneity:** "Chronic pain" encompasses back pain, fibromyalgia, neuropathy, etc., each with partially distinct genetic architectures. Lumping them dilutes the signal.
- **Extreme polygenicity:** Effects spread across so many variants that current samples can't detect most of them.

## 4. Training Data: The Major Biobanks

### 4.1 UK Biobank (UKB)

~500K participants (mostly middle-aged, mostly European). The central resource for PGS development. Rich phenotyping including cognitive tests, biomarkers, imaging, and linked health records. Access requires a health-related research proposal and modest fees (~£500/3yr). Data accessed via a secure cloud platform (UKB-RAP); individual-level data cannot be downloaded. Available to academic, commercial, charity, and government researchers worldwide — no academic affiliation required.

UKB is the only mega-biobank with actual cognitive performance testing (8 tests ranging from fluid intelligence to trail making — see Section 5.2 for details).

### 4.2 FinnGen

~500K Finnish individuals. Phenotypes from national health registries (hospital records, prescriptions): 2,200+ disease endpoints but no cognitive testing. GWAS summary statistics are freely downloadable. Individual-level data requires collaboration with Finnish investigators.

### 4.3 Million Veteran Program (MVP)

~636K US veterans. Highly diverse (29% non-European ancestry). Phenotypes from electronic health records. GWAS summary statistics available through dbGaP. Limited cognitive data: self-report MOS Cognitive Functioning scale only (not a performance test).

### 4.4 All of Us (NIH)

~1M+ participants, specifically recruited for diversity (>50% non-European). Tiered data access: registered tier (demographics, surveys) and controlled tier (genomics, EHR). Access requires registration and data use agreement. Growing resource particularly important for non-European PGS development.

### 4.5 ABCD Study

~12K US children with extensive cognitive testing, brain imaging, and genotyping. Access via NBDC Data Hub with a Data Use Certification; requires affiliation with an NIH-recognized institution. Used by Herasight as an independent validation cohort for CogPGT.

### 4.6 Summary Statistics vs. Individual-Level Data

For constructing polygenic scores, GWAS summary statistics (freely available from FinnGen, MVP, GWAS Catalog, SSGAC, PGC, etc.) are typically sufficient — this is the input to SBayesRC, LDpred2, PRS-CS. Individual-level data (requiring data access agreements) is needed for: phenotype engineering (combining noisy measures into latent factors), within-family validation, and direct penalized regression approaches.

## 5. Cognitive Phenotyping and Herasight's CogPGT

### 5.1 The Challenge

UK Biobank's cognitive assessment was brief (~5 minutes at baseline), administered on touchscreens without supervision, and expanded over time so that not all participants completed all tests. The 13-item fluid intelligence test, taken in 2 minutes as part of a multi-hour intake process, has a Cronbach's alpha of only 0.62 — a noisy snapshot of true ability.

### 5.2 UKB Cognitive Tests

| Test | Domain | N (approx) | g-loading | Test-retest ICC |
|------|--------|-----------|-----------|-----------------|
| Fluid intelligence | Reasoning | 170K | High | 0.65 |
| Reaction time | Processing speed | 470K | Low | 0.57 |
| Pairs matching | Visual memory | 470K | Low | 0.17 |
| Symbol digit substitution | Processing speed/WM | 100K | High | — |
| Trail making A/B | Executive function | 100K | Moderate | — |
| Matrix pattern completion | Nonverbal reasoning | 100K | High | — |
| Tower rearranging | Planning | 100K | Moderate | — |
| Numeric memory | Working memory | 50K | Moderate | — |
| Prospective memory | Prospective memory | 470K | — | — |

Factor analysis of these tests reveals that the first principal component (the $g$ factor) explains ~29% of variance across all tests. Symbol digit substitution has the highest loading on the main factor; reaction time and pairs matching have very high uniqueness (>0.90), meaning they are poor measures of general ability.

### 5.3 Constructing Latent $g$

With complete data, one would simply extract the first principal component. With pervasive missingness (not all participants took all tests), the standard approach is a linear factor model:

$$x_i = \lambda_i \cdot g + \varepsilon_i$$

Loadings $\lambda_i$ are estimated from individuals with complete or near-complete data. For an individual who took subset $S$ of the tests, the posterior mean of $g$ is:

$$E[g | x_S] = \left(\lambda_S^\top \Sigma_{\varepsilon,S}^{-1} \lambda_S + 1\right)^{-1} \lambda_S^\top \Sigma_{\varepsilon,S}^{-1} x_S$$

Individuals with fewer tests get wider posterior variance but point estimates on the same scale.

Running a GWAS on the imputed $g$ rather than on any single noisy test yields larger effect sizes and better downstream PGS, because measurement noise attenuates observed associations.

### 5.4 Herasight's Phenotype Engineering

Herasight reports achieving $R^2 \approx 0.164$ for their CogPGT score (population level), compared to ~0.076 from a published academic $g$-factor PGS. They attribute the gap to "extensive phenotype engineering" and "deep learning-based imputation." Possible approaches beyond standard factor analysis include nonlinear latent variable models (variational autoencoders), joint models integrating cognitive scores with correlated phenotypes like educational attainment, or multi-trait GWAS methods (MTAG, Genomic SEM) that boost effective sample size by borrowing signal from correlated traits.

They explicitly distinguish their approach from using educational attainment as a proxy for cognition, noting that EA GWAS is heavily confounded by socioeconomic factors (within-family attenuation for EA is ~50–60%, much larger than for direct cognitive measures).

## 6. Embryo Genome Reconstruction

### 6.1 Existing IVF Genetic Testing

Preimplantation genetic testing is already common in IVF:

- **PGT-A (Aneuploidy):** Screens for chromosomal abnormalities (trisomy, monosomy). Used in >50% of US IVF cycles. Based on ultra-low-pass (~0.005x) whole-genome sequencing of trophectoderm biopsy cells. The data required for PGT-A is the starting point for ImputePGTA.
- **PGT-M (Monogenic):** Tests for specific single-gene conditions (cystic fibrosis, sickle cell, BRCA). Requires custom probe design per family. The genetic analysis is essentially deterministic — if both parents are carriers, each embryo has a 25% chance of being affected.
- **PGT-P (Polygenic):** Polygenic scoring of embryos. The newest and most controversial form. This is what Herasight and competitors offer.

The key insight of ImputePGTA is that PGT-A data, already routinely generated, contains enough information to reconstruct embryo genomes for polygenic scoring — no additional biopsy or sequencing is needed.

### 6.2 The Biological Setup

Each embryo inherits one haplotype from each parent per chromosome. The father has haplotypes $h_{d1}$ and $h_{d2}$; during meiosis, crossover between them produces a recombinant haplotype (typically 1–3 crossovers per chromosome). The embryo gets this recombinant as its paternal copy. Independently, the mother's $h_{m1}$ and $h_{m2}$ recombine to produce the maternal copy.

Critically, crossover occurs *within* each parent's pair of chromosomes, not between paternal and maternal chromosomes. The embryo's genome is a mosaic of exactly four parental haplotypes, with crossover breakpoints as the only unknowns.

### 6.3 The Trophectoderm Biopsy

The biopsy is performed at the blastocyst stage (day 5–6), when the embryo has ~100–200 cells differentiated into inner cell mass (ICM, which becomes the baby) and trophectoderm (TE, which becomes the placenta). The biopsy removes 5–10 cells from the TE only, leaving the ICM untouched. This is considered safe because the TE can tolerate the loss and the future child is unaffected.

Earlier approaches (day 3) removed 1–2 cells from an 8-cell embryo — removing up to 25% of an undifferentiated embryo where all cells are totipotent (any cell could contribute to any part of the baby). Day 5 TE biopsy is now standard due to better safety profile.

A subtlety: since the biopsy is from TE, not ICM, there is a small chance of mosaicism — the TE and ICM may have different genetic content, leading to false positives or negatives.

### 6.4 ImputePGTA: The Algorithm

ImputePGTA models the embryo genome as a hidden Markov model:

- **Hidden states:** Which pair of parental haplotypes (one from $\{h_{d1}, h_{d2}\}$, one from $\{h_{m1}, h_{m2}\}$) is present at each genomic position.
- **Transitions:** Crossover events (rare; probability derived from genetic maps).
- **Observations:** Sparse ULP sequencing reads (~0.005x coverage), each mapped to a known genomic position and showing one allele. At heterozygous parental sites, a single read can distinguish which haplotype is present.

Parental haplotypes are known from high-coverage whole-genome sequencing (~30x, ~$200–400 per parent). The HMM's output is a posterior distribution over embryo genotypes at all positions, with uncertainty propagated through to PGS posteriors.

**ImputePGTA V2** jointly models all embryos in a cycle. Since siblings share the same four parental haplotypes, information from one embryo helps correct phasing errors in others. Even aneuploid embryos contribute information on their euploid chromosomes.

The reported accuracy is 0.96 dosage correlation for the two clinically relevant ULP embryos tested, though the validation sample size (N=2 for ULP) is very small.

### 6.5 Imputation for Adult Genotype Data

For adult data from SNP arrays (e.g., 23andMe's ~600K SNPs), standard imputation uses a reference panel of phased haplotypes (1000 Genomes, TOPMed with ~100K+ individuals). An HMM matches observed genotypes to reference haplotypes in sliding windows, filling in unobserved positions. Common variants impute at $r^2 > 0.95$. Tools: BEAGLE, IMPUTE5, Minimac4. Free imputation servers: Michigan Imputation Server, TOPMed Imputation Server.

## 7. Within-Family Validation: Why It Matters for Embryo Selection

### 7.1 What Population-Level PGS Captures

A PGS validated at the population level captures a mix of: direct genetic effects (causal), indirect genetic effects / "genetic nurture" (parental genotypes affecting offspring through environment), population stratification (allele frequencies correlated with environmental differences), and assortative mating inflation.

For embryo selection, only direct genetic effects differentiate siblings. Everything else is shared within a family.

### 7.2 The Sibling Test

Given sibling pairs, regress the phenotype on PGS with family fixed effects:

$$y_{ij} = \alpha_i + \gamma \cdot \text{PGS}_{ij} + \varepsilon_{ij}$$

The coefficient $\gamma$ estimates the within-family (direct) effect. For educational attainment, the within-family effect is typically ~50–60% of the population effect — indicating massive confounding. Herasight's CogPGT reports within-family $r \approx 0.46$ in UKB sibling pairs, with minimal attenuation from the population $r \approx 0.41$.

### 7.3 Effect Shrinkage in Embryo Selection

When selecting among embryos, the genetic variance between siblings is half the population genetic variance. This is because each embryo inherits a random half of each parent's genome; the variance of $\text{PGS}_{embryo}$ around the parental midpoint equals half the additive genetic variance.

Concretely: if PGS $R^2 = 0.16$ at the population level (within-family), the expected variance in PGS among embryos from the same parents is halved, so the effective $R^2$ for distinguishing embryos is lower. The gain from selecting the best of $k$ embryos scales as the expected maximum of $k$ draws from a distribution with this reduced variance.

For non-European parents, the effect is further reduced because PGS accuracy drops with genetic distance from the training population. CogPGT retains ~89% of its effect in Hispanic/Latino Americans, ~88% in South Asian Americans, ~64% in African Americans, and ~59% in East Asian Americans.

### 7.4 Gains from Selection: Number of Embryos Matters

The expected gain from selecting the best of $k$ embryos (in terms of standard deviations of the within-family PGS distribution) scales approximately as the expected maximum of $k$ standard normal draws:

| Embryos ($k$) | Expected max (SD units) | Approx IQ gain (Herasight CogPGT) |
|---------------|------------------------|-------------------------------------|
| 2 | 0.56 | ~3 |
| 3 | 0.85 | ~4.5 |
| 5 | 1.16 | ~6 |
| 10 | 1.54 | ~8.5 |
| 20 | 1.87 | ~10 |

These numbers are rough estimates that assume the PGS perfectly captures direct genetic effects at the within-family level. Actual gains depend on the within-family $R^2$, number of viable euploid embryos (often <5), and whether selection is over a single trait or a weighted combination.

## 8. De Novo Mutations

### 8.1 Why They Matter

Each embryo carries ~50–100 de novo single-nucleotide variants. Most are harmless (landing in noncoding, non-conserved sequence — coding regions are only ~1.5% of the genome). But pathogenic de novo mutations (protein-truncating variants in haploinsufficient genes, gain-of-function missense in critical domains) collectively account for a large fraction of severe developmental disorders, intellectual disability, and some congenital conditions.

### 8.2 Detection Challenges

ImputePGTA fundamentally cannot detect de novo mutations: the embryo genome is modeled as a mosaic of parental haplotypes, so truly new variants are invisible by construction. Detection would require direct embryo sequencing at ~30x+ coverage. However, the trophectoderm biopsy yields only ~5–10 cells, requiring whole-genome amplification (WGA), which introduces artifacts (allele dropout, chimeric reads, amplification errors) producing thousands of false positive "de novo" calls. Estimated additional cost: $2,000–5,000 per embryo. Progress depends primarily on better amplification chemistry (a wet lab problem) rather than computational methods.

## 9. The Embryo Selection Decision

### 9.1 Key Traits

For parents considering polygenic screening (excluding monogenic conditions handled by PGT-M):

1. **Coronary artery disease** — leading cause of death
2. **Type 2 diabetes** — extremely common; strong genetic component
3. **Breast cancer** — common and devastating
4. **Alzheimer's disease** — highly feared; partially heritable
5. **Schizophrenia** — highly heritable; severely disabling
6. **Major depressive disorder** — very common
7. **Type 1 diabetes** — childhood onset; strong HLA component
8. **Cognitive ability** — correlated with health, longevity, and life outcomes
9. **Prostate cancer** — common; PGS performs well
10. **Atrial fibrillation** — common cardiac arrhythmia; stroke risk factor

### 9.2 Positive Pleiotropy and the Dimensionality of Selection

Because diseases share genetic risk factors, PGS values across traits for a given embryo are correlated. This could be quantified by computing the genetic correlation matrix (from LD score regression) across all screened traits and examining its eigenstructure. Preliminary considerations suggest ~3–4 approximately independent genetic axes:

- **Cardiometabolic/cognitive axis:** CAD, T2D, cognitive ability, depression, and longevity are substantially genetically correlated.
- **Autoimmune/HLA axis:** T1D, celiac disease, rheumatoid arthritis cluster around HLA variants with distinct genetic architecture.
- **Cancer axis:** Partially distinct, though some overlap with metabolic traits.
- **Psychiatric axis:** Partially overlaps with cognition; schizophrenia and bipolar disorder are highly genetically correlated with each other.

When a company reports "this embryo reduces risk across 10 diseases," the effective number of independent selection dimensions is likely much smaller. The benefits are real but partially redundant.

### 9.3 QALY-Based Expected Utility

A decision framework for comparing embryos:

$$E[\Delta U] = \sum_t \left[ P_t^{(1)} - P_t^{(2)} \right] \cdot D_t \cdot L_t \cdot V \cdot \delta(a_t)$$

where for trait $t$: $P_t^{(1)}, P_t^{(2)}$ are disease risks for embryos 1 and 2; $D_t$ is the QALY disability weight (from Global Burden of Disease); $L_t$ is expected duration of illness; $V$ is the value per QALY (~$100–150K); and $\delta(a_t)$ is a discount factor for age of onset.

Representative disability weights: T2D ~0.05–0.10 per year; well-managed heart disease ~0.05–0.10; schizophrenia ~0.4–0.6; Alzheimer's dementia ~0.5–0.7; major depressive episodes ~0.3–0.5.

Example: reducing T2D risk from 20% to 10%, with onset at 55 and survival to 80 (25 years × 0.08 disability weight = 2 QALYs lost), saves 0.10 × 2 = 0.2 expected QALYs, worth ~$20–30K at standard valuations.

No public QALY-based embryo selection calculator currently exists, though the components (disability weights, lifetime prevalence, PGS-adjusted risks) are all available.

## 10. Open Source Landscape and Reproducibility

### 10.1 Freely Available Resources

**GWAS summary statistics (marginal betas):**
- GWAS Catalog (ebi.ac.uk/gwas) — the main repository for published GWAS results
- FinnGen — summary statistics for 2,200+ disease endpoints
- MVP — PheWAS summary statistics via dbGaP
- SSGAC — educational attainment and cognitive performance GWAS
- PGC — psychiatric traits

**Pre-computed PGS weights (joint/conditional effects, ready for scoring):**
- PGS Catalog (pgscatalog.org) — thousands of scores across hundreds of traits. Note: contains a mix of weights from different methods (C+T marginal betas after clumping, Bayesian posterior means). All are designed for direct use as $\text{PGS} = \sum w_j g_j$.

**PGS methods (all open source):**
- SBayesRC (GitHub: zhilizheng/SBayesRC) — current state of the art
- LDpred2 (R package bigsnpr)
- PRS-CS (GitHub: getian107/PRScs)
- PRS-CSx (cross-ancestry extension)

**Scoring/pipeline tools:**
- pgsc_calc (PGS Catalog team) — automates variant harmonization and scoring, though its usability has been critiqued
- plink2 --score — fast, flexible PGS computation

**Imputation:**
- Michigan Imputation Server and TOPMed Imputation Server (both free web services)
- BEAGLE, IMPUTE5, Minimac4 (local tools)

**Family-based analysis:**
- SNIPar (GitHub: AlexTISYoung/snipar) — parental genotype imputation, family GWAS, within-family PGS analysis. MIT license. Academic work of Herasight co-founder Alex Young.

**Cognitive GWAS data:**
- Published $g$-factor GWAS from UKB (N ~187K) with summary statistics available
- Individual test GWAS (fluid intelligence, reaction time, etc.) from multiple publications
- Large meta-analyses combining UKB with CHARGE and COGENT consortia (N ~300K)

### 10.2 What Remains Proprietary

- **CogPGT weights:** Herasight's optimized cognitive PGS with phenotype engineering
- **Disease PGS weights:** Their optimized scores integrating multiple GWAS sources
- **ImputePGTA code:** Described in a preprint but not released
- **Phenotype engineering methodology:** Specific deep learning approach for constructing latent $g$
- **Clinical pipeline:** IVF clinic integration, genetic counseling, regulatory compliance

### 10.3 Feasibility of an Open Replication

One could approximate Herasight's approach using public data:

1. Download cognitive GWAS summary statistics ($g$-factor GWAS or multi-trait analysis combining individual test GWAS via MTAG/Genomic SEM)
2. Run SBayesRC with Baseline-LD v2.2 functional annotations
3. Download disease GWAS summary statistics from FinnGen, GWAS Catalog, etc.
4. Run SBayesRC for each disease trait
5. Use standard imputation tools for target genotypes

Expected quality gap: publicly available cognitive PGS explains ~7.6% of variance; Herasight claims ~16.4%. The gap reflects phenotype engineering, larger effective GWAS (through multi-trait methods), and proprietary QC. For disease scores the gap may be smaller.

The value of an open alternative lies less in consumer use (parents spending $30K+ on IVF won't cut corners on scoring) and more in: scientific accountability (enabling independent validation of commercial claims), research enablement, and global access (in countries where IVF costs $3–8K and Herasight's fee would dominate total cost).

## 11. Current Market

### 11.1 Cost Structure

| Component | Cost (US) |
|-----------|----------|
| IVF cycle | $15,000–30,000 |
| PGT-A | $4,000–6,000 |
| Embryo biopsy | $1,000–2,000 |
| Parental WGS (both parents) | $400–800 |
| Herasight polygenic scoring | $25,000–50,000 (sliding scale) |
| Computational cost of scoring | Negligible (<$100) |

IVF costs are 3–10x lower in Czech Republic, India, Thailand, and Mexico, making the polygenic scoring fee a proportionally larger barrier.

### 11.2 Adoption Gap

~500K IVF cycles/year in the US; ~2–3M globally. PGT-P adoption is very low (likely low thousands of total customers across all companies). Barriers: physician unfamiliarity and conservatism, no ACOG guidelines, emotional difficulty of ranking embryos, association with eugenics, regulatory restrictions outside the US.

The expected health-economic value per cycle (rough estimate: $10–20K in discounted future health gains) arguably justifies adding polygenic screening at modest marginal cost. The adoption gap reflects awareness, trust, and institutional inertia rather than unfavorable cost-benefit analysis.

### 11.3 Key Companies

- **Herasight:** Claims best-in-class PGS; emphasis on within-family validation; ImputePGTA enables any-clinic integration; ~$25–50K; customers in 11+ countries; emerged from stealth mid-2025
- **Orchid:** Earlier entrant; dedicated embryo genotyping; reportedly used by high-profile customers
- **Nucleus Genomics:** Backed by Peter Thiel; Robert Plomin as collaborator; criticized for lack of within-family validation and potentially inflated claims; plagiarism controversy
- **LifeView (formerly Genomic Prediction):** Longest-running PGT-P provider; more conservative claims
