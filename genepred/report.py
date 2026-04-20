"""Bridge `scoring.score_genome()` output → QALY-annotated report.

Population-level interpretation only (within-family attenuation is
applied for embryo selection but not for an individual's report).
"""

from __future__ import annotations

from math import erf, sqrt

from genepred.qaly import (
    ANCESTRY_R2_RATIO,
    CONTINUOUS_TRAITS,
    DISEASE_TRAITS,
    liability_threshold_risk,
)
from genepred.scoring import ScoreResult


def _phi(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def annotate(results: list[ScoreResult], ancestry_ratio: float = 1.0) -> dict:
    """Attach risk/shift/QALY/cost to each trait the QALY model knows
    about. Returns {'disease': [...], 'continuous': [...], 'unscored': [...],
    'total_qaly': float, 'total_cost': float}.

    `ancestry_ratio` scales every R² (see qaly.ANCESTRY_R2_RATIO)."""
    diseases, continuous, unscored = [], [], []
    for r in results:
        if r.z is None:
            unscored.append(r)
            continue
        # Homemade-score trait keys carry a method suffix
        # ("osteoporosis_sbrc", "cognitive_ability_mtag"); the qaly
        # model is keyed by base trait. Strip known suffixes.
        tkey = r.trait
        for suf in ("_sbrc", "_mtag"):
            if tkey.endswith(suf) and tkey[: -len(suf)] in (
                DISEASE_TRAITS | CONTINUOUS_TRAITS
            ):
                tkey = tkey[: -len(suf)]
                break
        if tkey in DISEASE_TRAITS:
            dt = DISEASE_TRAITS[tkey]
            r2 = dt.pgs_r2_population * ancestry_ratio
            risk = liability_threshold_risk(r.z, dt.prevalence, r2)
            d_qaly = -(risk - dt.prevalence) * dt.qaly_loss_if_affected
            d_cost = -(risk - dt.prevalence) * dt.lifetime_cost_if_affected
            diseases.append(
                dict(
                    result=r,
                    display=dt.display_name,
                    r2=r2,
                    base=dt.prevalence,
                    risk=risk,
                    rr=risk / dt.prevalence,
                    d_qaly=d_qaly,
                    d_cost=d_cost,
                )
            )
        elif tkey in CONTINUOUS_TRAITS:
            ct = CONTINUOUS_TRAITS[tkey]
            r2 = ct.pgs_r2_population * ancestry_ratio
            shift = r.z * sqrt(r2)
            d_qaly = shift * ct.qaly_per_sd
            d_cost = shift * ct.savings_per_sd
            continuous.append(
                dict(
                    result=r,
                    display=ct.display_name,
                    r2=r2,
                    shift_sd=shift,
                    d_qaly=d_qaly,
                    d_cost=d_cost,
                )
            )
        else:
            unscored.append(r)
    # When both a Catalog and a homemade score map to the same qaly
    # trait, keep only the one with the most matched SNPs so the TOTAL
    # doesn't double-count.
    def _dedup(rows):
        best: dict[str, dict] = {}
        for d in rows:
            k = d["display"]
            if k not in best or d["result"].n_matched > best[k]["result"].n_matched:
                best[k] = d
        return list(best.values())

    diseases = _dedup(diseases)
    continuous = _dedup(continuous)
    diseases.sort(key=lambda d: -abs(d["d_qaly"]))
    continuous.sort(key=lambda d: -abs(d["d_qaly"]))
    return {
        "disease": diseases,
        "continuous": continuous,
        "unscored": unscored,
        "total_qaly": sum(d["d_qaly"] for d in diseases + continuous),
        "total_cost": sum(d["d_cost"] for d in diseases + continuous),
    }


def format_report(results: list[ScoreResult], meta: dict, source: str = "") -> str:
    """Full text report for one genome."""
    pop = meta.get("super_pop", "EUR")
    ratio = ANCESTRY_R2_RATIO.get(pop, 1.0)
    a = annotate(results, ancestry_ratio=ratio)
    L = []
    L += [
        f"PGS report{f' for {source}' if source else ''}",
        f"reference: 1KG {pop}  |  {meta.get('n_snps', 0):,} genotyped SNPs",
        "Population-level interpretation; ΔQALY/Δ$ vs population mean.",
        "",
    ]
    if pop != "EUR":
        L += [
            "! Inferred ancestry: %s. Polygenic scores were trained on" % pop,
            "  European cohorts; predictive R² in %s averages ~%.0f%% of EUR"
            % (pop, ratio * 100),
            "  (range varies widely by trait). Risk and ΔQALY below have been",
            "  scaled by that factor and should be treated as rough estimates.",
            "",
        ]
    L += [
        "'PGS pct' = where the *score* falls in the 1KG reference. "
        "'trait pct' / 'your risk' = where the *trait* is predicted to",
        "fall, accounting for R². For continuous traits the bracket is "
        "the 95% credible interval on the trait given the score —",
        "narrow if R² is high, nearly [0–100] if R² is low. A 7th-pct "
        "PGS at R²=0.05 means trait ≈ 37% [1–94]: the score barely",
        "constrains the trait.",
        "",
        "=" * 116,
        "DISEASE TRAITS  (liability-threshold: z·√R² shifts liability; "
        "risk = P(L > threshold | z))",
        "=" * 116,
        f"{'trait':<26} {'PGS z':>7} {'PGS pct':>8} {'R²':>6} "
        f"{'baseline':>9} {'your risk':>10} {'RR':>5} {'ΔQALY':>8} "
        f"{'Δ$':>10}  overlap",
        "-" * 116,
    ]
    for d in a["disease"]:
        r: ScoreResult = d["result"]
        L.append(
            f"{d['display']:<26} {r.z:+7.2f} {r.percentile:7.1f}% "
            f"{d['r2']:6.3f} {d['base']:8.1%} {d['risk']:9.1%} "
            f"{d['rr']:5.2f} {d['d_qaly']:+8.3f} {d['d_cost']:+10,.0f}  "
            f"{r.n_matched:,}/{r.n_total:,}"
        )
    L += [
        "",
        "=" * 116,
        "CONTINUOUS TRAITS  (predicted shift = z·√R² in trait SD; "
        "trait pct = Φ(shift), with ±√(1-R²) posterior SD)",
        "=" * 116,
        f"{'trait':<26} {'PGS z':>7} {'PGS pct':>8} {'R²':>6} "
        f"{'shift':>8} {'trait pct':>14} {'ΔQALY':>8} {'Δ$':>10}  overlap",
        "-" * 116,
    ]
    for c in a["continuous"]:
        r = c["result"]
        shift = c["shift_sd"]
        post_sd = sqrt(max(0.0, 1 - c["r2"]))
        tp = _phi(shift)
        lo, hi = _phi(shift - 1.96 * post_sd), _phi(shift + 1.96 * post_sd)
        L.append(
            f"{c['display']:<26} {r.z:+7.2f} {r.percentile:7.1f}% "
            f"{c['r2']:6.3f} {shift:+7.2f}σ "
            f"{tp*100:4.0f}% [{lo*100:2.0f}–{hi*100:3.0f}] "
            f"{c['d_qaly']:+8.3f} {c['d_cost']:+10,.0f}  "
            f"{r.n_matched:,}/{r.n_total:,}"
        )
    L += [
        "",
        f"{'TOTAL (vs population mean)':<58} "
        f"{a['total_qaly']:+8.3f} {a['total_cost']:+10,.0f}",
    ]
    n_scored = len(a["disease"]) + len(a["continuous"])
    n_unscored = len(a["unscored"])
    if n_unscored > n_scored:
        L += [
            "",
            f"! WARNING: {n_unscored} of {n_scored + n_unscored} scores "
            f"could not be interpreted. Common causes:",
            "  - shipped resources are stale relative to the curated set "
            "(test_resources_cover_curated would fail; rerun "
            "reference/onekg/build_1kg_reference.py)",
            "  - very low SNP overlap → wrong --build for a "
            "position-only VCF, or weight files not rsID-annotated "
            "(re-run `genepred fetch-weights`)",
        ]
    if a["unscored"]:
        L += ["", "Not interpreted (no 1KG-reference entry, no QALY "
              "mapping, or low SNP overlap):"]
        for r in a["unscored"]:
            L.append(
                f"  {r.trait:<26} {r.pgs_id}  matched {r.n_matched:,}/{r.n_total:,}"
            )
    return "\n".join(L)
