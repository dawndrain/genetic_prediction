# genepred

Predict phenotypes from consumer genotyping data (23andMe, AncestryDNA,
etc.) using polygenic scores from the
[PGS Catalog](https://www.pgscatalog.org/), and estimate the expected
gains from embryo selection.

> **Bottom line:** with today's published polygenic scores, selecting
> the QALY-best of 5 embryos gives an expected gain of **~0.47 QALY**
> (~$120k personal value at $100k/QALY) over a randomly-chosen sibling
> — roughly a quarter of the additive-heritability ceiling. The
> [predictor-scaling analysis](examples/r2_scaling.py) shows how this
> grows as predictors improve.

## What this does

1. **Score genomes** — takes raw DTC genotype files or VCFs, optionally
   imputes missing variants (locally with Beagle, or via the Michigan
   Imputation Server), and computes polygenic scores for ~20 traits
   with z-scores ancestry-adjusted against 1000 Genomes. The score
   set is curated to be roughly best-in-class per trait: most are
   from the PGS Catalog (Yengo height, Patel GPSMult CAD, etc.), but
   for traits where the best-published predictor isn't deposited
   there — cognition, stroke, BMD, AF, asthma — we build SBayesRC
   weights ourselves from open GWAS summary statistics
   (`reference/build_open_sbayesrc.sh`,
   `reference/build_cognition_v2_sbayesrc.sh`).

2. **Validate predictions** — height, BMI, and a homemade cognition
   predictor all reproduce expected R² against openSNP self-reports;
   the other scores pass a genetic-correlation sanity check. See
   `validation/` and the table below.

3. **Estimate embryo-selection value** — Monte Carlo over ~20 traits
   with their genetic correlation matrix, mapping PGS shifts to QALYs
   and cost via a liability-threshold + cited-utility model. See the
   [predictor-scaling report](https://htmlpreview.github.io/?https://github.com/dawndrain/genetic_prediction/blob/main/docs/r2_scaling_report.html)
   ([source](docs/r2_scaling_report.html)).

4. **End-to-end embryo demo** — phased 1KG parents → simulated meiosis
   → PGT-A biopsy at ~0.05× → HMM haplotype recovery → score → rank.
   Output: [embryo report](https://htmlpreview.github.io/?https://github.com/dawndrain/genetic_prediction/blob/main/docs/embryo_report.html)
   ([source](docs/embryo_report.html)).

## Why not just use X?

| Tool | What it lacks |
|---|---|
| [pgsc_calc](https://github.com/PGScatalog/pgsc_calc) (PGS Catalog) | Doesn't impute; doesn't curate (you bring your own PGS IDs); Nextflow + Docker stack is brittle; ancestry normalization scales by within-pop variance but doesn't adjust the R² used downstream. |
| [Michigan PGS Server](https://imputationserver.sph.umich.edu) | Returns raw score sums only — no z-score, no percentile, no ancestry adjustment. |
| [impute.me](https://impute.me) | Defunct since 2023; predictors were never updated past ~2019. |
| Commercial (Herasight, Genomic Prediction, Orchid) | Closed weights and methods; can't be independently validated or reproduced. |

This is ~2k lines of pure Python with numpy/scipy as the only hard
dependencies for scoring — no PLINK, no Nextflow, no Docker. The
imputation step adds Java (for Beagle) but is optional.

## Install

```bash
# with conda (pulls in plink2, bcftools, htslib, java, 7z):
conda env create -f environment.yml
conda activate genepred

# or pip-only (you'll need bcftools/bgzip/tabix/java on $PATH for
# imputation; scoring works without them):
pip install -e .[io]
```

## Quick start

### Score a genome

```bash
genepred fetch-weights                    # one-time, ~300 MB

# no genome of your own? grab a public-domain 23andMe file (24 MB):
./examples/fetch_example_genome.sh        # → data/example_genome.txt

genepred score data/example_genome.txt           # PGS + risk + QALY report
genepred score data/example_genome.txt --basic   # compact table, one row per PGS
```

`score` auto-detects VCF vs DTC text, projects onto 1KG ancestry PCs
(reports your closest super-population), and emits a z-score and
percentile per trait. The
[partial-overlap correction](docs/pgs_vs_qaly_comparison.md) means
un-imputed DTC files give usable z-scores; imputation improves them.

### Imputation (optional)

Consumer arrays capture ~600K–900K SNPs but most PGS use ~1M. Two
backends — pick **Beagle** unless you have a reason not to:

```bash
# Local — no account, ~10 min on 64 cores against 1KG Phase 3.
# Auto-fetches beagle.jar + GRCh37 genetic maps on first run.
genepred impute beagle my_23andme.txt

# Michigan — better panel (HRC), but you receive a decryption password
# by email, so it's a 3-step process:
genepred impute michigan submit my_23andme.txt --out-dir mich/
# ... wait for the job + email ...
genepred impute michigan status mich/
genepred impute michigan fetch mich/ --password "<from-email>"
```

Set `MICHIGAN_API_TOKEN` (get one at
https://imputationserver.sph.umich.edu → Account → API Token) for the
Michigan path.

### Clinical embryo pipeline demo (parents → biopsy → HMM → rank)

This produces a report like [embryo report](https://htmlpreview.github.io/?https://github.com/dawndrain/genetic_prediction/blob/main/docs/embryo_report.html)

```bash
python examples/embryo_selection_demo.py --pop CEU --n-embryos 5
```

todo: how to get a similar report for real embryos

## Layout

```
genepred/        # the installable package + CLI
  io.py          # DTC/VCF loading, 1KG-conformed VCF emission
  pca.py         # project onto 1KG ancestry PCs, assign super-pop
  scoring.py     # raw PGS, z-score (PC-adjusted or ref-pop), √f overlap correction, QALY-annotated report
  catalog.py     # curated current-best PGS per trait + download/verify
  impute/        # beagle (local default), michigan (submit/status/fetch)
  qaly.py        # disease+continuous trait tables, rg matrix, liability model, MC
  resources/     # SHIPPED: PCA loadings, 1KG score summaries, PC coefs (~4 MB)
reference/       # rebuilds resources/ from scratch — reproducibility, not user path
validation/      # openSNP height check + PGS Catalog Calculator comparison
examples/        # embryo demo, batch scoring, synthetic genome
tools/           # vendored binaries (gitignored; conda env supplies these)
data/            # gitignored; PGS weights + 1KG VCFs land here
```

## Validation results

Three predictors were validated end-to-end against openSNP
self-reported phenotypes (2,779 23andMe genotypes from the 2017
Internet Archive snapshot, scored without imputation):

| Trait | PGS | Observed R² | n | Expected after attenuation |
|---|---|---|---|---|
| Height | PGS002804 (Yengo 2022, 1.1M SNPs) | 0.16 (males) / 0.04 (females) | 317 / 221 | ~0.27 |
| BMI | PGS002313 (Weissbrod 2022) | 0.06 (both sexes) | 83 | ~0.06 |
| Cognition | homemade (Savage×EA4 → MTAG → LDpred-inf) | r ≈ +0.25 vs IQ / edu years / SAT | 24–52 | — |

All land on the expected attenuation curve: full-overlap population R²
× ~0.7 (for ~50% array-vs-PGS SNP overlap, given LD redundancy) × ~0.8
(self-report reliability). The female-height result is anomalously low
— the predictor's distribution and the height phenotypes both look
clean by sex, and BMI works fine in females, so it's likely an
openSNP-specific reporting quirk rather than a pipeline bug; unresolved.

The other 17 disease scores pass a genetic-correlation sanity check
(pairwise PGS correlations across 1KG-EUR match published LDSC rg
signs — the cardiometabolic cluster is internally consistent, height
correlates +0.07 with cognition, etc.) but openSNP doesn't have the
case counts to validate prediction accuracy directly.

**Imputation accuracy** (chr22, 1,348 held-out genotypes × 5 samples):

| Method | Genotype concordance | Dosage r² |
|---|---|---|
| Michigan (HRC + Eagle + Minimac4) | 97.8% | 0.964 |
| Beagle 5 (1KG Phase 3, local) | 97.1% | 0.942 |
| Mean-impute (2·AF) | 63.8% | 0.441 |

The Michigan/Beagle gap is the reference-panel size (HRC's 32k
haplotypes vs 1KG's 5k), not the algorithm. Beagle is the recommended
local default — within 1% of Michigan, no account or email-password
loop, ~3–4 min genome-wide on 64 cores.

## Embryo selection: headline numbers and caveats

`examples/r2_scaling.py` runs the Monte Carlo under four
predictor-strength scenarios. At best-of-5:

| Scenario | ΔQALY [p10–p90] | Personal value (QALY×$100k + $) |
|---|---|---|
| **Current PGS** | 0.56 [0.28–0.88] | ~$143k |
| SNP-h² ceiling (infinite GWAS) | 1.05 | ~$274k |
| Twin h² (additive ceiling) | 1.88 | ~$487k |

So today's predictors capture roughly a quarter of what's
theoretically achievable; the gap is mostly rare variants that
common-SNP GWAS don't see.

**Caveats** — known sources of optimism in the model:

- **European ancestry only.** Every R² is from EUR-trained, EUR-validated
  scores. In African ancestry, current PGS R² is typically 2–5× lower;
  in East/South Asian, 1.5–2× lower. The heritability ceilings transfer
  better but are also EUR-estimated.
- **Within-family attenuation.** Population R² overstates between-sibling
  discrimination because some of the population signal is environmental
  confounding siblings share. The model applies per-trait
  `within_family_ratio` (Howe et al. 2022 where available, assumed
  0.75–0.90 for diseases otherwise).
- **Twin-h² for psychiatric traits is probably inflated** —
  equal-environments violation plus de-novo-variant misattribution
  mean the twin-h² scenario overstates schizophrenia/bipolar/ADHD
  headroom.
- **Residual overcounting.** BMI and cognition/income are explicitly
  residualized against the diseases/traits they cause, but the
  per-disease QALY-loss figures may still embed some comorbidity
  effects. Probably a ~5–10% overstatement across the cardiometabolic
  cluster.
- **No age/sex stratification** — population-average prevalences;
  breast/prostate cancer use sex-averaged figures.
- **Linear continuous-trait model** — `qaly_per_sd` is a constant
  slope, which is wrong in the tails (BMI is U-shaped, height is
  mildly inverse-U). Fine for the ±0.5σ range that selection actually
  operates over; misleading for extreme individuals.

## What individual-level biobank access would add

Everything here runs from public summary statistics, which already
captures most of the gain from large-N — meta-analysed sumstats from
GIANT/PGC/DIAGRAM aggregate millions of samples that no single
biobank matches. What individual-level data *does* let you do that
sumstats can't:

- **Direct within-family validation** — measure R² between actual
  sibling pairs instead of inferring it from population R² ×
  Howe-2022 attenuation ratios. This is the biggest uncertainty in
  the embryo-selection numbers.
- **Ancestry-specific tuning** — fit per-population weights or
  ensemble across PRS-CSx ancestry components, instead of applying
  a flat cross-ancestry attenuation factor.
- **Non-additive models** — sumstats are marginal-additive by
  construction. Dominance, epistasis, and genuine G×E need
  individual genotypes (and the gain is probably small for most
  traits, but unknown for the ones where MZ−2·DZ is large).
