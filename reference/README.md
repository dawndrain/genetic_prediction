Scripts that **rebuild** the precomputed resources shipped in
`genepred/resources/`. Users do not need to run these.

## `onekg/` — 1000 Genomes reference prep

Run from the repo root (~2 h, ~30 GB disk):

```bash
./reference/onekg/download_1kg.sh        # 1KG Phase3 VCFs → data/1kg/
                                          #   then dedup_1kg.py → data/1kg_dedup/
                                          #   (drops multi-allelic dup-position
                                          #   rows; ≈ bcftools norm -d both)
python reference/onekg/compute_1kg_pca.py        # → loadings.tsv, sample_pcs.tsv
genepred fetch-weights                            # PGS Catalog → data/pgs_scoring_files/
                                                   #   (rsID backfill runs automatically)
python reference/onekg/build_1kg_reference.py    # score 1KG → 1kg_pgs_summary.tsv,
                                                   #   pgs_pc_coef.tsv

cp data/1kg_pgs_summary.tsv data/1kg_pca/pgs_pc_coef.tsv genepred/resources/
gzip -kc data/1kg_pca/loadings.tsv > genepred/resources/loadings.tsv.gz
gzip -kc data/1kg_pca/sample_pcs.tsv > genepred/resources/sample_pcs.tsv.gz
```

## `weights/` — homemade PGS weight derivation

For traits where the best-published score isn't in the Catalog. All
need GCTB 2.5+ and the `ukbEUR_HM3` LD eigen reference (see script
headers for download URLs); ~30–50 min per trait on 32 threads.

```bash
# Open-license traits (auto-downloads sumstats; redistributable):
./reference/weights/build_open_sbayesrc.sh all   # bmd, af, stroke, asthma

# Cognition (Savage 2018 × EA4 → MTAG → SBayesRC):
./reference/weights/download_cognition_sumstats.sh
python reference/weights/mtag_lite.py
./reference/weights/build_cognition_v2_sbayesrc.sh

# PGC psychiatric (DUA-restricted; sumstats need manual click-through;
# outputs MUST NOT be used in the embryo path or redistributed):
./reference/weights/build_pgc_sbayesrc.sh <depression|schizophrenia|...>
```

`weights/ldpred_inf.py` is the v1 closed-form alternative to SBayesRC
(faster, no GCTB needed, slightly worse R²). `weights/pseudo_validate.py`
sanity-checks a newly-built weight file by scoring 1KG and comparing
the resulting z-distribution to the existing reference.
