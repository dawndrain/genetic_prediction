# Genetic Prediction

Predict phenotypes from consumer genotyping data (23andMe, AncestryDNA, etc.) using polygenic scores from the [PGS Catalog](https://www.pgscatalog.org/), and estimate the expected gains from embryo selection.

## What this does

1. **Score genomes** — takes raw DTC genotype files, optionally imputes missing variants (via Beagle or the Michigan Imputation Server), and computes polygenic scores for traits like height, heart disease, diabetes, schizophrenia, cognitive ability, and more.

2. **Validate predictions** — validates the height PGS (PGS001229) against self-reported height data from OpenSNP. On ~100 individuals with imputed genomes, the PGS achieves r ≈ 0.40 (r² ≈ 0.16) with measured height — consistent with published benchmarks for a genetics-only prediction (no age/sex covariates).

3. **Estimate embryo selection value** — `qaly_calculator.py` simulates selecting the best embryo from a set of siblings across ~20 traits, estimating QALY gains and cost savings. It models genetic correlations between traits (e.g. the psychiatric cluster, the metabolic cluster) and uses the liability threshold model for disease traits.

## Quick start

### Score your own genome

```bash
# 1. Convert DTC file to VCF
python convert_to_vcf.py

# 2. Download PGS scoring file from PGS Catalog
#    (e.g. PGS001229 for height — already included in data/)

# 3a. Impute with Beagle (local, downloads reference panel + genetic maps automatically)
python impute.py beagle --sample YOUR_SAMPLE --parallel 4

# 3b. OR prep files for Michigan Imputation Server (better imputation, free, but requires upload)
python impute.py michigan --sample YOUR_SAMPLE

# 4. Compute PGS
python predict.py --pgs data/PGS001229.txt --sample YOUR_SAMPLE
```

### Find the best PGS for a trait

```bash
# Search PGS Catalog API, rank by R²/AUROC, optionally download top scoring files
python find_best_pgs.py --traits height heart_disease --ancestry EUR --download
```

See `pgs_trait_recommendations.md` for curated recommendations across 7 major traits.

### Embryo selection calculator

```bash
# Simulate selecting best of 5 embryos across all traits
python qaly_calculator.py --embryos 5

# Only specific traits
python qaly_calculator.py --embryos 10 --only heart_disease type2_diabetes schizophrenia

# Score an individual's PGS z-scores
python qaly_calculator.py --scores heart_disease=-0.5 height=1.2 cognitive_ability=0.7

# List all available traits and their parameters
python qaly_calculator.py --list-traits
```

## Validation results

Validated the height PGS (PGS001229, 51K variants, snpnet) against OpenSNP self-reported heights:

| Method | r | r² | n |
|--------|---|-----|---|
| Raw (no imputation) | ~0.33 | ~0.11 | ~400 |
| Imputed (slim reference) | ~0.40 | ~0.16 | ~100 |

The imputed scores are better despite fewer samples — imputation fills in the ~80% of PGS variants missing from consumer arrays. The r² ≈ 0.16 is consistent with what you'd expect for a genetics-only height prediction without covariates (the PGS Catalog reports R² = 0.717 for this score, but that includes age+sex covariates in the snpnet model).

## Imputation

Consumer genotyping arrays capture ~600K-900K SNPs, but many PGS use millions of variants. Imputation infers the missing variants using a reference panel of fully-sequenced genomes.

Two options:

- **Beagle (local)**: `impute.py beagle` downloads the 1000 Genomes Phase 3 reference panel and GRCh37 genetic maps automatically. Use `shrink_reference.py` to slim the reference to just PGS-relevant positions (makes imputation ~50-100x faster with minimal accuracy loss for PGS scoring).
- **Michigan Imputation Server**: `impute.py michigan` prepares bgzipped VCFs for upload. Better imputation quality, free, but requires uploading your data. The server requires ≥5 samples — use `--merge dup` to duplicate a single sample, or `--merge multi` to merge multiple samples.

## Embryo selection: caveats

The QALY calculator is a useful framework but has known sources of optimism:

- **Assumes European ancestry** — PGS effect sizes and prevalence estimates are calibrated to European-ancestry populations. Prediction accuracy drops substantially for other ancestries.
- **Population-level R², not sibling-level** — PGS R² is measured at the population level. Among siblings (who share environment), the *incremental* predictive power of the PGS is lower by roughly 0.8x for traits with significant shared-environment effects (e.g. educational attainment, income). Height is less affected since shared environment explains little of its variance.
- **Traits assumed independent** — the genetic correlation matrix handles PGS-level correlations, but the QALY impacts are additive. In reality, e.g. diabetes and heart disease share downstream health consequences.
- **No time discounting** — a QALY at age 30 is weighted the same as one at age 80.

## Pipeline scripts

| Script | Purpose |
|--------|---------|
| `find_best_pgs.py` | Search PGS Catalog API for best scores per trait |
| `convert_to_vcf.py` | Convert DTC genotype files to per-chromosome VCFs |
| `preremap.py` | Remap non-build-37 genomes to GRCh37 |
| `impute.py` | Local Beagle imputation or Michigan server prep |
| `shrink_reference.py` | Slim reference panel to PGS-relevant positions |
| `predict.py` | Compute PGS from imputed VCFs |
| `predict_raw.py` | Compute PGS directly from raw genotype files (no imputation) |
| `batch_impute_score.py` | End-to-end batch pipeline: convert → impute → score |
| `parse_phenotypes.py` | Parse OpenSNP height/sex data for validation |
| `qaly_calculator.py` | QALY/cost calculator and embryo selection simulator |

## With actual biobank access

The approach here uses precomputed PGS from the PGS Catalog — someone else has already done the hard work of fitting the model. With access to individual-level biobank data (e.g. UK Biobank), you could:

1. Fit your own linear model from GWAS summary statistics + LD matrices, using methods like LDpred, PRS-CS, or SBayesR — this lets you tune the model to your target population
2. Use functional annotations (protein-coding regions, conserved regions) to get better priors on which SNPs are causal and their likely effect sizes
3. Validate on held-out data with known phenotypes, including sibling pairs to measure within-family prediction accuracy
