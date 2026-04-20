Shipped reference data — small enough to install with the package.

These are *outputs* of `reference/`; users do not need to regenerate
them.

| file                  | from                             | used by         |
| --------------------- | -------------------------------- | --------------- |
| `loadings.tsv.gz`     | `reference/compute_1kg_pca.py`   | `pca.project`   |
| `sample_pcs.tsv.gz`   | `reference/compute_1kg_pca.py`   | `pca.assign_*`  |
| `pgs_pc_coef.tsv`     | `reference/build_1kg_reference.py` (PC regression of 1KG raw scores) | `scoring._normalize` (PC-adjusted) |
| `1kg_pgs_summary.tsv` | `reference/build_1kg_reference.py` (per-pop mean/SD of 1KG raw scores) | `scoring._normalize` (ref-pop)     |

`paths.resource()` resolves a `.gz` sibling transparently, so callers
ask for `loadings.tsv` and get the gzipped file.
