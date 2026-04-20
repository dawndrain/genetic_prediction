"""Project a genome onto the 1KG ancestry principal components.

The shipped resources are:
  - loadings.tsv(.gz): per-SNP standardized loadings for PC1..k
  - sample_pcs.tsv:    PC coordinates of the 2,504 1KG samples (with
                       super_pop labels) — used for population assignment
  - pgs_pc_coef.tsv:   per-PGS regression of raw score on PCs (intercept,
                       beta_PC1..k, residual SD) — consumed by scoring.

These are precomputed by reference/compute_1kg_pca.py.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from genepred.paths import open_maybe_gz, resource


@dataclass
class PcaModel:
    pc_cols: list[str]
    loadings: list[tuple]  # (rsid, chrom, pos, ref, alt, alt_af, [load_PC1..k])
    coef: dict[str, tuple]  # pgs_id -> (intercept, [b_PC1..k], resid_sd)


_CACHE: dict[str, object] = {}


def load_pca() -> PcaModel:
    """Load the shipped 1KG PCA loadings and PGS-on-PC coefficients."""
    if "pca" in _CACHE:
        return _CACHE["pca"]  # type: ignore
    rows = []
    with open_maybe_gz(resource("loadings.tsv")) as f:
        hdr = f.readline().rstrip().split("\t")
        pc_cols = [c for c in hdr if c.startswith("PC")]
        ix = {c: i for i, c in enumerate(hdr)}
        for line in f:
            r = line.rstrip().split("\t")
            rows.append(
                (
                    r[ix["rsid"]],
                    r[ix["chrom"]],
                    int(r[ix["pos"]]),
                    r[ix["ref_allele"]],
                    r[ix["alt_allele"]],
                    float(r[ix["alt_af"]]),
                    [float(r[ix[c]]) for c in pc_cols],
                )
            )
    coef = {}
    with open_maybe_gz(resource("pgs_pc_coef.tsv")) as f:
        hdr = f.readline().rstrip().split("\t")
        ix = {c: i for i, c in enumerate(hdr)}
        for line in f:
            r = line.rstrip().split("\t")
            coef[r[ix["pgs_id"]]] = (
                float(r[ix["intercept"]]),
                [float(r[ix[c]]) for c in pc_cols],
                float(r[ix["resid_sd"]]),
            )
    model = PcaModel(pc_cols=pc_cols, loadings=rows, coef=coef)
    _CACHE["pca"] = model
    return model


def project(by_rs, by_pos, model: PcaModel | None = None) -> tuple[list[float], int]:
    """Project a genome onto the PC axes.

    Missing loading SNPs contribute 0 (the standardized mean), which is
    correct mean-imputation. The partial sum is rescaled by n_total/n_used
    to undo the attenuation from missing SNPs under random missingness.
    """
    model = model or load_pca()
    n_pc = len(model.pc_cols)
    pcs = [0.0] * n_pc
    n_used = 0
    for rsid, chrom, pos, ref, alt, af, loads in model.loadings:
        g = by_rs.get(rsid) or by_pos.get((chrom, pos))
        if g is None:
            continue
        if len(g) == 3:
            gref, galt, ds = g
            if (gref, galt) == (ref, alt):
                d = ds
            elif (gref, galt) == (alt, ref):
                d = 2.0 - ds
            else:
                continue
        else:
            a1, a2 = g
            if {a1, a2} <= {ref, alt}:
                d = (a1 == alt) + (a2 == alt)
            else:
                continue
        denom = (2 * af * (1 - af)) ** 0.5
        z = (d - 2 * af) / denom
        for k in range(n_pc):
            pcs[k] += z * loads[k]
        n_used += 1
    if n_used > 0:
        scale = len(model.loadings) / n_used
        pcs = [p * scale for p in pcs]
    return pcs, n_used


def load_sample_pcs() -> list[tuple[str, str, list[float]]]:
    """Returns [(sample_id, super_pop, [PC1..k]), ...] for the 1KG panel."""
    if "sample_pcs" in _CACHE:
        return _CACHE["sample_pcs"]  # type: ignore
    out = []
    with open_maybe_gz(resource("sample_pcs.tsv")) as f:
        hdr = f.readline().rstrip().split("\t")
        ix = {c: i for i, c in enumerate(hdr)}
        pc_cols = [c for c in hdr if c.startswith("PC")]
        for line in f:
            r = line.rstrip().split("\t")
            out.append(
                (
                    r[ix["sample"]],
                    r[ix["super_pop"]],
                    [float(r[ix[c]]) for c in pc_cols],
                )
            )
    _CACHE["sample_pcs"] = out
    return out


def assign_population(
    pcs: list[float], k_neighbors: int = 25, use_pcs: int = 4
) -> tuple[str, float]:
    """Closest 1KG super-population by k-NN in the first `use_pcs` PCs.
    Returns (super_pop, fraction_of_neighbors_in_that_pop)."""
    samples = load_sample_pcs()
    dists = []
    for sid, sp, spc in samples:
        d = sum((pcs[i] - spc[i]) ** 2 for i in range(min(use_pcs, len(pcs))))
        dists.append((d, sp))
    dists.sort()
    votes = Counter(sp for _, sp in dists[:k_neighbors])
    pop, n = votes.most_common(1)[0]
    return pop, n / k_neighbors
