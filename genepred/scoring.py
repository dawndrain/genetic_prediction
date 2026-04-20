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

import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from math import erf, sqrt
from pathlib import Path

from tqdm import tqdm

from genepred.catalog import CURATED, list_weight_files
from genepred.io import COMPLEMENT, load_genotypes
from genepred.paths import open_maybe_gz, resource
from genepred.pca import assign_population, load_pca, project
from genepred.qaly import ANCESTRY_R2_RATIO


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


_GENOME: tuple | None = None  # (by_rs, by_pos, build) — fork-shared


def _score_worker(item):
    pgs_id, wf = item
    by_rs, by_pos, build = _GENOME  # type: ignore[misc]
    return score_one(by_rs, by_pos, wf, build)


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
    n_jobs: int | None = None,
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

    if len(by_pos) < 10_000:
        raise ValueError(
            f"Only {len(by_pos):,} SNPs loaded — expected ≥100k for a "
            f"DTC array file. Either the file isn't a 23andMe/Ancestry/"
            f"VCF genotype export, or the format wasn't recognized "
            f"(check delimiter, header, encoding)."
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
    weight_files = list(list_weight_files(weights_dir))

    # Parallelize over weight files. Fork-based so the genome dicts
    # (~1M keys each) are inherited copy-on-write rather than pickled.
    n_workers = min(
        len(weight_files),
        max(1, n_jobs if n_jobs is not None else (os.cpu_count() or 4)),
    )
    global _GENOME
    _GENOME = (by_rs, by_pos, build)
    if n_workers > 1 and sys.platform != "win32":
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            it = pool.imap(_score_worker, weight_files)
            if verbose:
                it = tqdm(
                    it, total=len(weight_files), desc="  scoring", unit="pgs",
                    file=sys.stderr,
                    bar_format="{desc}: {n_fmt}/{total_fmt} {bar} {elapsed}",
                )
            scored = list(it)
    else:
        it = weight_files
        if verbose:
            it = tqdm(
                it, desc="  scoring", unit="pgs", file=sys.stderr,
                bar_format="{desc}: {n_fmt}/{total_fmt} {bar} {elapsed}",
            )
        scored = [_score_worker(w) for w in it]
    _GENOME = None

    for (pgs_id, wf), r in zip(weight_files, scored):
        trait = pgs_to_trait.get(pgs_id, pgs_id)
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


def _trait_pct(z: float, r2: float) -> tuple[float, float, float]:
    """(point, lo95, hi95) trait-percentile given PGS z and predictor R²."""
    shift = z * sqrt(max(0.0, r2))
    sd = sqrt(max(0.0, 1 - r2))
    cdf = lambda x: 0.5 * (1 + erf(x / sqrt(2)))  # noqa: E731
    return cdf(shift), cdf(shift - 1.96 * sd), cdf(shift + 1.96 * sd)


def format_results(results: list[ScoreResult], meta: dict) -> str:
    pop = meta["super_pop"]
    ratio = ANCESTRY_R2_RATIO.get(pop, 1.0)
    r2_by_id = {s.pgs_id: s.r2_eur_pop * ratio for s in CURATED.values()}
    tw = max(26, max((len(r.trait) for r in results), default=0))
    pw = max(11, max((len(r.pgs_id) for r in results), default=0))
    out = [
        f"{'trait':<{tw}} {'pgs_id':<{pw}} {'snps':>9} {'matched':>8} "
        f"{'overlap':>8} {'z (' + pop + ')':>9} {'PGS pct':>8} "
        f"{'R²':>6} {'trait pct [95% CI]':>20}",
        "-" * (tw + pw + 80),
    ]
    for r in results:
        zstr = f"{r.z:+8.2f}" if r.z is not None else "    n/a"
        pstr = f"{r.percentile:6.1f}%" if r.percentile is not None else "   n/a"
        r2 = r2_by_id.get(r.pgs_id)
        if r.z is not None and r2 is not None:
            tp, lo, hi = _trait_pct(r.z, r2)
            tpstr = f"{r2:6.3f} {tp*100:5.0f}% [{lo*100:3.0f}–{hi*100:3.0f}]"
        else:
            tpstr = f"{'—':>6} {'—':>20}"
        out.append(
            f"{r.trait:<{tw}} {r.pgs_id:<{pw}} {r.n_total:>9,} {r.n_matched:>8,} "
            f"{r.overlap:>7.1%} {zstr:>9} {pstr:>8} {tpstr}"
        )
    out += [
        "",
        f"PGS pct  = where the score falls in the 1KG-{pop} score "
        f"distribution.",
        "trait pct = predicted trait percentile = Φ(z·√R²); brackets "
        "are the 95% credible interval given the score's R².",
        "         A wide interval means the predictor is weak — the "
        "score barely constrains the trait.",
    ]
    return "\n".join(out)
