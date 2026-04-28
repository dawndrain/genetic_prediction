Worked examples.

- `fetch_example_genome.sh` — downloads a real public-domain 23andMe
  raw file (~24 MB, 966k SNPs) to `data/example_genome.txt` for trying
  the score/report/impute commands without your own data.
- `r2_scaling.py` — Monte-Carlo selection-gain analysis under four
  predictor-strength scenarios (current PGS / SNP-h² / twin h² / MZ
  correlation), with personal-vs-societal cost decomposition.
  Output: `data/r2_scaling_report.html`.

The embryo-selection demo now lives in the package: run
`genepred embryo-demo` (or `python -m genepred.embryo_cli`); the
phasing-error benchmark and real-data validation scripts are under
`validation/`. See `docs/PHASING.md`.
