"""Bridge `scoring.score_genome()` output → QALY-annotated report.

Population-level interpretation only (within-family attenuation is
applied for embryo selection but not for an individual's report).
"""

from __future__ import annotations

from math import sqrt

from genepred.qaly import (
    ANCESTRY_R2_RATIO,
    CONTINUOUS_TRAITS,
    DISEASE_TRAITS,
    liability_threshold_risk,
)
from genepred.scoring import ScoreResult


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
        if r.trait in DISEASE_TRAITS:
            dt = DISEASE_TRAITS[r.trait]
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
        elif r.trait in CONTINUOUS_TRAITS:
            ct = CONTINUOUS_TRAITS[r.trait]
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
        "=" * 110,
        "DISEASE TRAITS  (liability-threshold: z·√R² shifts liability; "
        "risk = P(L > threshold | z))",
        "=" * 110,
        f"{'trait':<26} {'PGS z':>7} {'%ile':>6} {'pop R²':>7} "
        f"{'baseline':>9} {'your risk':>10} {'RR':>5} {'ΔQALY':>8} "
        f"{'Δ$':>10}  overlap",
        "-" * 110,
    ]
    for d in a["disease"]:
        r: ScoreResult = d["result"]
        L.append(
            f"{d['display']:<26} {r.z:+7.2f} {r.percentile:5.1f}% "
            f"{d['r2']:7.3f} {d['base']:8.1%} {d['risk']:9.1%} "
            f"{d['rr']:5.2f} {d['d_qaly']:+8.3f} {d['d_cost']:+10,.0f}  "
            f"{r.n_matched:,}/{r.n_total:,}"
        )
    L += [
        "",
        "=" * 110,
        "CONTINUOUS TRAITS  (predicted shift = z·√R² in trait SD)",
        "=" * 110,
        f"{'trait':<26} {'PGS z':>7} {'%ile':>6} {'pop R²':>7} "
        f"{'shift':>9} {'ΔQALY':>8} {'Δ$':>10}  overlap",
        "-" * 110,
    ]
    for c in a["continuous"]:
        r = c["result"]
        L.append(
            f"{c['display']:<26} {r.z:+7.2f} {r.percentile:5.1f}% "
            f"{c['r2']:7.3f} {c['shift_sd']:+8.2f}σ {c['d_qaly']:+8.3f} "
            f"{c['d_cost']:+10,.0f}  {r.n_matched:,}/{r.n_total:,}"
        )
    L += [
        "",
        f"{'TOTAL (vs population mean)':<58} "
        f"{a['total_qaly']:+8.3f} {a['total_cost']:+10,.0f}",
    ]
    if a["unscored"]:
        L += ["", "Not interpreted (no QALY entry, or low SNP overlap):"]
        for r in a["unscored"]:
            L.append(
                f"  {r.trait:<26} {r.pgs_id}  matched {r.n_matched:,}/{r.n_total:,}"
            )
    return "\n".join(L)
