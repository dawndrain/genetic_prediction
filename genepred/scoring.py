"""Polygenic scoring against PGS Catalog harmonized weight files.

Two normalization paths to a z-score / percentile:

  1. PC-adjusted (preferred): the raw score is residualized against a
     regression of 1KG raw scores on the first k ancestry PCs. This
     removes the smooth ancestry component without binning into
     super-populations.

  2. Reference-population: standardize against the empirical mean/SD of
     a chosen 1KG super-population (EUR, EAS, AFR, AMR, SAS, ALL).

Both apply the partial-overlap correction. If the genome only matches
fraction f of the SNPs the reference matched (e.g. raw 23andMe with no
imputation), the partial sum has expectation ~ f·μ_ref and variance
~ f·σ_ref² (under random missingness, additive contributions), so

    z ≈ (raw - f·μ) / (σ·√f).

This is what makes raw, un-imputed DTC files give usable z-scores.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from math import erf, sqrt
from pathlib import Path

from genepred.catalog import CURATED, list_weight_files
from genepred.io import COMPLEMENT, load_genotypes
from genepred.paths import open_maybe_gz, resource
from genepred.pca import assign_population, load_pca, project


@dataclass
class ScoreResult:
    pgs_id: str
    trait: str
    n_total: int
    n_matched: int
    n_imputed: int
    n_ambiguous: int
    raw: float
    z: float | None
    percentile: float | None
    method: str  # "pc-adjusted", "ref-pop:<pop>", "hwe", or "low-overlap"

    @property
    def overlap(self) -> float:
        return self.n_matched / self.n_total if self.n_total else 0.0


# ------------------------------------------------------- weight-file scoring


def read_pgs_header(f) -> list[str]:
    """Skip '##' metadata, return the column-name row."""
    for line in f:
        if (
            line.startswith("##")
            or line.startswith("#pgs")
            or line.startswith("#format")
        ):
            continue
        if line.startswith("#"):
            line = line[1:]
        cols = line.rstrip("\n").split("\t")
        if "effect_allele" in cols and "effect_weight" in cols:
            return cols
    raise ValueError("no header row found in PGS weight file")


def score_one(by_rs, by_pos, pgs_path, build: str = "GRCh37") -> dict:
    """Score one genome against one PGS weight file.

    Matching tries rsID first, then (chrom,pos) in the requested build.
    Strand flips are resolved by complementing observed alleles when that
    yields a match — except A/T and C/G SNPs, which are inherently
    ambiguous and counted in n_ambiguous instead. Sites missing from the
    genome are mean-imputed at 2·EAF if the weight file carries a
    frequency column.
    """
    n_total = n_matched = n_imputed = n_ambiguous = 0
    raw = exp = var = 0.0
    with open_maybe_gz(pgs_path) as f:
        cols = read_pgs_header(f)
        ix = {c: i for i, c in enumerate(cols)}
        i_rs = ix.get("hm_rsID", ix.get("rsID", ix.get("rsid")))
        i_ea = ix["effect_allele"]
        i_oa = ix.get("other_allele", ix.get("hm_inferOtherAllele"))
        i_w = ix["effect_weight"]
        i_af = ix.get("allelefrequency_effect")
        if build == "GRCh38":
            i_chr, i_pos = ix.get("hm_chr"), ix.get("hm_pos")
        else:
            i_chr, i_pos = ix.get("chr_name"), ix.get("chr_position")
        for line in f:
            if line.startswith("#"):
                continue
            row = line.rstrip("\n").split("\t")
            if len(row) <= i_w:
                continue
            n_total += 1
            try:
                w = float(row[i_w])
            except ValueError:
                continue
            ea = row[i_ea].upper()
            oa = row[i_oa].upper() if i_oa is not None and i_oa < len(row) else ""
            af = None
            if i_af is not None and i_af < len(row):
                try:
                    af = float(row[i_af])
                except (ValueError, IndexError):
                    pass

            g = None
            rsid = row[i_rs] if i_rs is not None and i_rs < len(row) else ""
            if rsid and rsid in by_rs:
                g = by_rs[rsid]
            elif i_chr is not None and i_pos is not None:
                try:
                    key = (row[i_chr].lstrip("chr"), int(row[i_pos]))
                    g = by_pos.get(key)
                except (ValueError, IndexError):
                    pass

            dosage = None
            if g is not None and len(g) == 3:
                # Continuous dosage from imputed VCF: (ref, alt, ds_alt).
                ref, alt, ds = g
                pair = {ref, alt}
                if ea == alt and (not oa or oa == ref):
                    dosage = ds
                    n_matched += 1
                elif ea == ref and (not oa or oa == alt):
                    dosage = 2.0 - ds
                    n_matched += 1
                elif {ea.translate(COMPLEMENT), oa.translate(COMPLEMENT)} == pair:
                    if {ea, oa} == pair:
                        n_ambiguous += 1
                    else:
                        dosage = ds if ea.translate(COMPLEMENT) == alt else 2.0 - ds
                        n_matched += 1
            elif g is not None:
                a1, a2 = g
                alleles = {ea, oa} - {""}
                obs = {a1, a2}
                if obs <= alleles:
                    dosage = (a1 == ea) + (a2 == ea)
                    n_matched += 1
                elif {a.translate(COMPLEMENT) for a in obs} <= alleles:
                    if {ea, oa} == {ea.translate(COMPLEMENT), oa.translate(COMPLEMENT)}:
                        n_ambiguous += 1
                    else:
                        eac = ea.translate(COMPLEMENT)
                        dosage = (a1 == eac) + (a2 == eac)
                        n_matched += 1

            if dosage is None and af is not None:
                dosage = 2 * af
                n_imputed += 1
            if dosage is None:
                continue

            raw += w * dosage
            if af is not None:
                exp += w * 2 * af
                var += w * w * 2 * af * (1 - af)

    z = (raw - exp) / sqrt(var) if var > 0 else float("nan")
    return dict(
        n_total=n_total,
        n_matched=n_matched,
        n_imputed=n_imputed,
        n_ambiguous=n_ambiguous,
        raw=raw,
        expected=exp,
        var=var,
        z=z,
    )


# ------------------------------------------------------------- normalization


def load_reference(super_pop: str, path: Path | None = None) -> dict:
    """pgs_id -> (mean, sd, n_snps) for the given 1KG super-population."""
    path = path or resource("1kg_pgs_summary.tsv")
    ref = {}
    with open_maybe_gz(path) as f:
        header = f.readline().rstrip().split("\t")
        ix = {c: i for i, c in enumerate(header)}
        for line in f:
            r = line.rstrip().split("\t")
            if r[ix["super_pop"]] != super_pop:
                continue
            sd = float(r[ix["sd"]])
            if sd <= 0:
                continue
            ref[r[ix["pgs_id"]]] = (float(r[ix["mean"]]), sd, int(r[ix["n_snps"]]))
    return ref


def _normalize(pgs_id, r, *, pcs, pca_model, ref_pop_stats, trait=None) -> tuple:
    """→ (z, percentile, method). Returns (None, None, "low-overlap") if
    overlap fraction f is outside [0.1, 1.2]."""
    keys = (pgs_id, pgs_id.split("_")[0], trait)
    n_ref = (
        next(
            (ref_pop_stats[k] for k in keys if k and k in ref_pop_stats),
            (0, 0, r["n_total"]),
        )[2]
        if ref_pop_stats
        else r["n_total"]
    )
    f = r["n_matched"] / max(n_ref, 1)
    if not (0.1 <= f <= 1.2):
        return None, None, "low-overlap"
    # PC-coefficient table is keyed by whatever build_1kg_reference.py used
    # for the score's filename stem at build time. Try the full pgs_id, then
    # the first token (handles "COGNITION_mtag_sbayesrc" → "COGNITION"), then
    # the trait name as a forward-compat key.
    coef = None
    if pcs is not None and pca_model is not None:
        coef = next(
            (pca_model.coef[k] for k in keys if k and k in pca_model.coef), None
        )
    if coef is not None:
        intercept, b, resid_sd = coef
        if resid_sd > 0:
            pred = intercept + sum(bk * pk for bk, pk in zip(b, pcs))
            z = (r["raw"] - f * pred) / (resid_sd * sqrt(f))
            return z, _z_to_pct(z), "pc-adjusted"
    ref = next((ref_pop_stats[k] for k in keys if k and k in ref_pop_stats), None)
    if ref is not None:
        mean, sd, _ = ref
        z = (r["raw"] - f * mean) / (sd * sqrt(f))
        return z, _z_to_pct(z), "ref-pop"
    if r["var"] > 0:
        return r["z"], _z_to_pct(r["z"]), "hwe"
    return None, None, "no-ref"


def _z_to_pct(z: float) -> float:
    return 50.0 * (1.0 + erf(z / sqrt(2.0)))


# -------------------------------------------------------- top-level pipeline


def score_genome(
    genome,
    *,
    build: str = "GRCh37",
    pc_adjust: bool = True,
    ref_pop: str | None = None,
    weights_dir: Path | None = None,
    verbose: bool = False,
) -> tuple[list[ScoreResult], dict]:
    """Score a genome against every available PGS weight file.

    `genome` may be a path or an already-loaded (by_rs, by_pos) tuple.
    Returns (results, meta) where meta has 'pcs', 'super_pop', 'n_snps'.
    If `ref_pop` is None it is inferred from the projected PCs.
    """
    if isinstance(genome, (str, Path)):
        if verbose:
            print(f"loading genotypes from {genome} ({build}) ...", file=sys.stderr)
        by_rs, by_pos = load_genotypes(genome)
    else:
        by_rs, by_pos = genome
    if verbose:
        print(
            f"  {len(by_pos):,} called SNPs ({len(by_rs):,} with rsID)", file=sys.stderr
        )

    pcs = None
    pca_model = None
    if pc_adjust:
        pca_model = load_pca()
        pcs, n_pc_snps = project(by_rs, by_pos, pca_model)
        if verbose:
            print(
                f"  ancestry PCs ({n_pc_snps:,}/{len(pca_model.loadings):,} "
                f"loading SNPs): "
                + " ".join(
                    f"{c}={v:+.1f}" for c, v in zip(pca_model.pc_cols[:4], pcs[:4])
                ),
                file=sys.stderr,
            )

    if ref_pop is None and pcs is not None:
        ref_pop, _ = assign_population(pcs)
    ref_pop = ref_pop or "EUR"
    ref_stats = load_reference(ref_pop)
    if verbose:
        print(f"  reference: {len(ref_stats)} scores ({ref_pop})", file=sys.stderr)

    pgs_to_trait = {s.pgs_id: trait for trait, s in CURATED.items()}
    results = []
    for pgs_id, wf in list_weight_files(weights_dir):
        trait = pgs_to_trait.get(pgs_id, pgs_id)
        r = score_one(by_rs, by_pos, wf, build)
        z, pct, method = _normalize(
            pgs_id, r, pcs=pcs, pca_model=pca_model, ref_pop_stats=ref_stats,
            trait=trait,
        )
        if method == "ref-pop":
            method = f"ref-pop:{ref_pop}"
        results.append(
            ScoreResult(
                pgs_id=pgs_id,
                trait=trait,
                n_total=r["n_total"],
                n_matched=r["n_matched"],
                n_imputed=r["n_imputed"],
                n_ambiguous=r["n_ambiguous"],
                raw=r["raw"],
                z=z,
                percentile=pct,
                method=method,
            )
        )

    meta = {"pcs": pcs, "super_pop": ref_pop, "n_snps": len(by_pos)}
    return results, meta


def format_results(results: list[ScoreResult], meta: dict) -> str:
    pop = meta["super_pop"]
    out = [
        f"{'trait':<26} {'pgs_id':<11} {'snps':>9} {'matched':>8} "
        f"{'overlap':>8} {'z (' + pop + ')':>9} {'pctile':>7}",
        "-" * 90,
    ]
    for r in results:
        zstr = f"{r.z:+8.2f}" if r.z is not None else "    n/a"
        pstr = f"{r.percentile:6.1f}%" if r.percentile is not None else "   n/a"
        out.append(
            f"{r.trait:<26} {r.pgs_id:<11} {r.n_total:>9,} {r.n_matched:>8,} "
            f"{r.overlap:>7.1%} {zstr:>9} {pstr:>7}"
        )
    return "\n".join(out)
