Validation scripts. Results are summarised in the top-level README
and in `docs/PHASING.md`; users don't need to run these.

### Embryo-imputation phasing error (`docs/PHASING.md`)

```bash
python validation/embryo_phasing_bench.py     # simulation sweep:
                                               #   SER × cov × n_embryos × method
python validation/embryo_phasing_validate.py  # real-data SER of 1KG phase via a
                                               #   SHAPEIT5 trio (needs
                                               #   tools/shapeit5/phase_common_static)
```

### PGS accuracy against openSNP self-reported phenotypes

```bash
./download_opensnp_archives.sh     # openSNP IA + Zenodo archives (~25 GB)
python validate_height_archive.py # score every genome in the archive on
                                   # the curated PGS set; compare to
                                   # self-reported height/BMI/IQ/edu/SAT
python impute_opensnp_batch.py    # batch-impute the phenotyped genomes
                                   # (Beagle, one multi-sample VCF/chrom)
                                   # and re-score: height R2 0.34->0.43 (M)
                                   # / 0.21->0.34 (F); needs the 1KG panel
```

Ancillary checks live in `aux/`:

- `aux/compare_imputation.py` — Michigan-HRC vs Beagle-1KG vs
  mean-impute concordance on the chr22 holdout from the Michigan
  submit step.
- `aux/michigan_to_zscore.py` — converts the Michigan PGS Server's
  raw-score output (it doesn't normalise) into z-scores against
  our 1KG reference for side-by-side comparison.

We attempted a comparison against the official PGS Catalog
Calculator (`pgsc_calc`) but it requires an un-sandboxed Docker
daemon with a shared writable filesystem, which our environment
doesn't provide. The Nextflow `-profile conda` route should work
on a normal machine if you want to run it.

**Note on openSNP**: the project shut down in 2025 and deliberately
deleted its dataset, citing re-identification risk. The archives
above predate that; the data was CC0 when captured. Used here for
methods validation only.
