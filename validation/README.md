Validation against openSNP self-reported phenotypes (height, BMI,
education years, IQ, SAT). Results are summarised in the top-level
README; users don't need to run these.

```bash
./download_opensnp_archives.sh     # openSNP IA + Zenodo archives (~25 GB)
python validate_height_archive.py # score every genome in the archive on
                                   # the curated PGS set; compare to
                                   # self-reported height/BMI/IQ/edu/SAT
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
