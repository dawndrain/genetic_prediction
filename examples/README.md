Worked examples.

- `fetch_example_genome.sh` — downloads a real public-domain 23andMe
  raw file (~24 MB, 966k SNPs) to `data/example_genome.txt` for trying
  the score/report/impute commands without your own data.
- `embryo_selection_demo.py` — the full Herasight-style pipeline:
  CEU-trio parents from 1KG → simulate 5 embryos via meiosis → simulate
  PGT-A biopsy at 0.05× → recover haplotype inheritance via HMM → score
  → rank by QALY. Library functions are in `genepred.embryo`.
- `r2_scaling.py` — Monte-Carlo selection-gain analysis under four
  predictor-strength scenarios (current PGS / SNP-h² / twin h² / MZ
  correlation), with personal-vs-societal cost decomposition.
  Output: `data/r2_scaling_report.html`.
