"""How does the expected gain from embryo selection scale with predictor
strength?

Runs the Monte-Carlo sibship simulation under three R² scenarios:
  - "current"  — the PGS R² we actually have today
  - "snp_h2"   — SNP-heritability (LDSC) — the ceiling for common-variant
                 additive prediction; what an infinitely-large GWAS on the
                 same SNP set would reach
  - "twin_h2"  — narrow-sense h² from twin/family studies — the ceiling
                 for *any* additive predictor, including rare variants

(MZ-twin correlation ≈ broad-sense H² is also shown for reference, but
selection on an additive score can't capture dominance/epistasis, so
narrow-sense h² is the relevant bound.)

Also dumps the current per-trait QALY-loss / cost / discounting
parameters so they can be reviewed.

Output: text summary + docs/r2_scaling_report.html
"""

import colorsys
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parents[1]))
from genepred import qaly as q

# Per-trait heritability estimates. SNP-h² mostly from the discovery
# GWAS's own LDSC estimate; twin h² from Polderman et al. 2015 (MaTCH)
# or trait-specific twin registries. Twin-h² values here are
# *deliberately discounted ~10–15%* from headline figures to account
# for equal-environments-assumption inflation (MZ twins share more
# environment than DZ; see the psychiatric caveat in the report).
# Updated 2026-04-16 after independent parameter audit — bumped
# several conservative values toward published medians.
H2 = {
    # trait_key:           (snp_h2, twin_h2_narrow, mz_corr_H2)
    "height": (0.48, 0.80, 0.90),
    "bmi": (0.25, 0.65, 0.78),
    "cognitive_ability": (0.20, 0.65, 0.78),
    "income": (0.05, 0.25, 0.40),
    "subjective_wellbeing": (0.05, 0.32, 0.38),
    # No validated longevity PGS exists in the pipeline, and most
    # lifespan variance is mediated by the diseases already listed.
    "longevity_residual": (0.00, 0.00, 0.00),
    "heart_disease": (0.10, 0.45, 0.55),
    "type2_diabetes": (0.18, 0.50, 0.72),
    # AD SNP-h² is highly sensitive to APOE inclusion and case
    # ascertainment; 0.20 is mid-range with APOE.
    "alzheimers": (0.20, 0.65, 0.78),
    "schizophrenia": (0.24, 0.75, 0.82),
    "depression": (0.08, 0.37, 0.42),
    "atrial_fibrillation": (0.12, 0.45, 0.62),
    # Breast/prostate prevalences in genepred.qaly are sex-averaged
    # (~half the per-sex lifetime risk); h² here is per-sex.
    "breast_cancer": (0.13, 0.30, 0.30),
    "prostate_cancer": (0.20, 0.50, 0.58),
    "stroke": (0.12, 0.35, 0.42),
    "colorectal_cancer": (0.10, 0.30, 0.35),
    "bipolar_disorder": (0.20, 0.70, 0.78),
    "chronic_kidney_disease": (0.07, 0.35, 0.45),
    "asthma": (0.12, 0.60, 0.72),
    "inflammatory_bowel_disease": (0.27, 0.60, 0.68),
    "adhd": (0.22, 0.74, 0.80),
    # T1D is one of the most heritable common diseases (HLA-driven).
    "type1_diabetes": (0.50, 0.85, 0.90),
    "osteoporosis": (0.20, 0.55, 0.70),
    "anxiety_disorders": (0.05, 0.32, 0.38),
}

SCENARIOS = {
    "current PGS": None,  # use traits' own pgs_r2_population
    "SNP-h² ceiling": 0,
    "twin h² (additive ceiling)": 1,
    "MZ correlation (H², for ref)": 2,
}

# Standard willingness-to-pay thresholds for converting QALY → $.
# US literature: $50k (historical), $100k (common), $150k (ICER upper).
# UK NICE: £20–30k ≈ $30–40k.
QALY_TO_USD = 100_000

# We don't have a validated longevity PGS, and it partly double-counts
# disease-mediated mortality. Toggle to exclude from totals.
EXCLUDE_TRAITS: set[str] = set()  # e.g. {"longevity_residual"}


@contextmanager
def _suppress_longevity():
    t = q.CONTINUOUS_TRAITS.get("longevity_residual")
    if t is None:
        yield
        return
    saved = t.pgs_r2_population
    t.pgs_r2_population = 0.0
    try:
        yield
    finally:
        t.pgs_r2_population = saved


@contextmanager
def override_r2(scenario_idx: int | None):
    """Temporarily replace each trait's pgs_r2_population with the
    scenario value. within_family_ratio is left at 1.0 for the
    heritability scenarios, since h² is by definition the direct
    within-family effect (no stratification/nurture inflation)."""
    if scenario_idx is None:
        with _suppress_longevity():
            yield
        return
    saved = {}
    for d in (q.DISEASE_TRAITS, q.CONTINUOUS_TRAITS):
        for k, t in d.items():
            saved[k] = (t.pgs_r2_population, t.within_family_ratio)
            t.pgs_r2_population = H2.get(k, (t.pgs_r2_population,) * 3)[scenario_idx]
            t.within_family_ratio = 1.0
    try:
        yield
    finally:
        for d in (q.DISEASE_TRAITS, q.CONTINUOUS_TRAITS):
            for k, t in d.items():
                t.pgs_r2_population, t.within_family_ratio = saved[k]


PERSONAL_MEDICAL_SHARE = 0.12  # US out-of-pocket fraction (CMS NHE 2023)

# Which traits' within-family ratios come from Howe et al. 2022
# (Nat Genet sibling-GWAS) vs being assumed. Diseases mostly weren't
# in Howe — the 0.75–0.90 defaults assume disease liability is
# mostly direct genetic effect (little nurture/stratification).
WF_RATIO_SOURCED = {
    "height",
    "bmi",
    "cognitive_ability",
    "income",
    "subjective_wellbeing",
    "depression",
}


def _split_savings(result, personal_share=PERSONAL_MEDICAL_SHARE):
    """Decompose the simulation's savings_gain into societal / personal /
    medical / productivity, using each trait's costs() split (60/40
    fallback). Continuous-trait savings_per_sd is treated as societal
    (mostly own earnings → counts as both productivity and personal)."""
    soc = med = prod = pers = 0.0
    for tk, solo in result["per_trait_solo"].items():
        sg = solo["savings_gain"]
        if tk in q.DISEASE_TRAITS:
            t = q.DISEASE_TRAITS[tk]
            m, p = t.costs()
            tot = m + p if (m + p) > 0 else t.lifetime_cost_if_affected or 1
            fm, fp = m / tot, p / tot
            med += sg * fm
            prod += sg * fp
            soc += sg
            pers += sg * (personal_share * fm + fp)
        else:
            soc += sg
            prod += sg
            pers += sg
    return {"societal": soc, "personal": pers, "medical": med, "productivity": prod}


def run_scenarios(
    n_embryos_list=(2, 3, 5, 10), n_sim=50_000, rate=0.0, use_survival=True
):
    out = {}
    for label, idx in SCENARIOS.items():
        out[label] = {}
        with override_r2(idx):
            for n in n_embryos_list:
                r = q.simulate_selection(
                    n_embryos=n,
                    n_simulations=n_sim,
                    use_correlations=True,
                    rate=rate,
                    use_survival=use_survival,
                )
                r["cost_split"] = _split_savings(r)
                out[label][n] = r
        print(f"  {label}: done", file=sys.stderr)
    return out


def _aggregate_value_table(results, scenarios):
    rows = []
    for s in scenarios:
        r5 = results[s][5]
        q_gain = r5["qaly_gain_mean"]
        p10, p90 = r5["qaly_gain_p10"], r5["qaly_gain_p90"]
        pers = r5["cost_split"]["personal"]
        soc = r5["cost_split"]["societal"]
        rows.append(
            f"<tr><td style='text-align:left'>{s}</td>"
            f"<td>{q_gain:.2f} "
            f"<small style='color:#888'>[{p10:.2f}–{p90:.2f}]</small></td>"
            f"<td>${q_gain * QALY_TO_USD:,.0f}</td>"
            f"<td>${pers:,.0f}</td>"
            f"<td><b>${q_gain * QALY_TO_USD + pers:,.0f}</b></td>"
            f"<td style='color:#888'>${q_gain * QALY_TO_USD + soc:,.0f}</td>"
            "</tr>"
        )
    return (
        "<h2>Aggregate value to the individual (best-of-5)</h2>"
        "<table><tr><th>Scenario</th>"
        "<th>ΔQALY <small>[p10–p90]</small></th>"
        f"<th>QALY × ${QALY_TO_USD // 1000}k</th><th>Personal $</th>"
        "<th><b>Total (personal)</b></th>"
        "<th style='color:#888'>Total (societal)</th></tr>" + "".join(rows) + "</table>"
    )


def write_html(results, n_embryos_list, out_path="docs/r2_scaling_report.html"):
    scenarios = list(results)
    colors = ["#888", "#4a90d9", "#2ca02c", "#d62728"]

    # main chart: QALY gain vs N for each scenario
    max_q = max(
        results[s][n]["qaly_gain_mean"] for s in scenarios for n in n_embryos_list
    )
    W, H_, ML, MB = 720, 360, 60, 40
    pw, ph = W - ML - 20, H_ - MB - 20
    xs = {
        n: ML + i * pw / (len(n_embryos_list) - 1) for i, n in enumerate(n_embryos_list)
    }

    def y(v):
        return 20 + ph * (1 - v / max_q)

    lines = []
    for si, s in enumerate(scenarios):
        pts = " ".join(
            f"{xs[n]:.0f},{y(results[s][n]['qaly_gain_mean']):.0f}"
            for n in n_embryos_list
        )
        lines.append(
            f'<polyline points="{pts}" fill="none" '
            f'stroke="{colors[si]}" stroke-width="2.5"/>'
        )
        for n in n_embryos_list:
            v = results[s][n]["qaly_gain_mean"]
            lines.append(
                f'<circle cx="{xs[n]:.0f}" cy="{y(v):.0f}" r="3" fill="{colors[si]}"/>'
            )
        lines.append(
            f'<text x="{W - 15}" y="{y(results[s][n_embryos_list[-1]]["qaly_gain_mean"]):.0f}" '
            f'fill="{colors[si]}" font-size="11" text-anchor="end" '
            f'dy="4">{s}</text>'
        )
    axes = (
        f'<line x1="{ML}" y1="{20 + ph}" x2="{ML + pw}" y2="{20 + ph}" stroke="#333"/>'
        f'<line x1="{ML}" y1="20" x2="{ML}" y2="{20 + ph}" stroke="#333"/>'
    )
    for n in n_embryos_list:
        axes += f'<text x="{xs[n]:.0f}" y="{H_ - 15}" text-anchor="middle" font-size="12">{n}</text>'
    for v in np.linspace(0, max_q, 5):
        axes += (
            f'<text x="{ML - 8}" y="{y(v):.0f}" text-anchor="end" '
            f'font-size="11" dy="4">{v:.1f}</text>'
            f'<line x1="{ML}" y1="{y(v):.0f}" x2="{ML + pw}" y2="{y(v):.0f}" '
            f'stroke="#eee"/>'
        )
    chart1 = (
        f'<svg width="{W}" height="{H_}">{axes}{"".join(lines)}'
        f'<text x="{ML + pw / 2}" y="{H_ - 2}" text-anchor="middle" '
        f'font-size="12">Number of embryos</text>'
        f'<text x="15" y="{20 + ph / 2}" text-anchor="middle" '
        f'font-size="12" transform="rotate(-90 15 {20 + ph / 2})">'
        f"Expected ΔQALY (best-of-N vs sib mean)</text></svg>"
    )

    # stacked bars at N=5: per-trait solo QALY gain under each scenario.
    # (This shows the gain from selecting on each trait *alone* — an
    # upper bound on its contribution to the joint index. The bars sum
    # to more than the joint gain because traits compete for the same
    # selection differential.)
    n5 = 5
    trait_keys = list(results[scenarios[0]][n5]["per_trait_solo"])
    LEGEND_W = 220
    BW, BH = 760 + LEGEND_W, 420
    plot_w = BW - LEGEND_W - 100
    bar_w = plot_w / len(scenarios) * 0.55
    solo_sums = {
        s: sum(
            v["qaly_gain"]
            for v in results[s][n5]["per_trait_solo"].values()
            if v["qaly_gain"] > 0
        )
        for s in scenarios
    }
    bar_max = max(solo_sums.values())
    # 24-color palette assigned by descending total contribution so the
    # biggest segments get the most distinct hues; colors are spaced in
    # HSL with alternating lightness so neighbors don't blur together.
    ranked = sorted(
        trait_keys,
        key=lambda t: -sum(
            results[s][n5]["per_trait_solo"].get(t, {"qaly_gain": 0})["qaly_gain"]
            for s in scenarios
        ),
    )
    palette = {}
    K = len(ranked)
    for i, t in enumerate(ranked):
        h = (i * 0.61803398875) % 1.0  # golden-ratio hue spacing
        light = 0.45 if i % 2 == 0 else 0.62
        r, g, b = colorsys.hls_to_rgb(h, light, 0.65)
        palette[t] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    bars = []
    for si, s in enumerate(scenarios):
        x = 80 + si * plot_w / len(scenarios)
        contribs = sorted(
            ((t, v["qaly_gain"]) for t, v in results[s][n5]["per_trait_solo"].items()),
            key=lambda kv: -kv[1],
        )
        cum = 0.0
        for t, v in contribs:
            if v <= 0:
                continue
            h = v / bar_max * (BH - 80)
            y0 = BH - 50 - cum / bar_max * (BH - 80) - h
            disp = q.DISEASE_TRAITS.get(t) or q.CONTINUOUS_TRAITS.get(t)
            label = getattr(disp, "display_name", t) if disp else t
            bars.append(
                f'<rect x="{x:.0f}" y="{y0:.0f}" width="{bar_w:.0f}" '
                f'height="{h:.0f}" fill="{palette[t]}" '
                f'stroke="#fff" stroke-width="0.5">'
                f"<title>{label}: {v:.3f} QALY</title></rect>"
            )
            if h >= 14:
                bars.append(
                    f'<text x="{x + bar_w / 2:.0f}" y="{y0 + h / 2 + 3:.0f}" '
                    f'text-anchor="middle" font-size="9" fill="#fff" '
                    f'style="pointer-events:none">'
                    f"{label[:14]} {v:.2f}</text>"
                )
            cum += v
        joint = results[s][n5]["qaly_gain_mean"]
        jy = BH - 50 - joint / bar_max * (BH - 80)
        bars.append(
            f'<line x1="{x - 8:.0f}" y1="{jy:.0f}" x2="{x + bar_w + 8:.0f}" '
            f'y2="{jy:.0f}" stroke="#000" stroke-width="2" '
            f'stroke-dasharray="3,2"/>'
            f'<text x="{x + bar_w + 10:.0f}" y="{jy:.0f}" font-size="10" '
            f'dy="3">joint {joint:.2f}</text>'
        )
        bars.append(
            f'<text x="{x + bar_w / 2:.0f}" y="{BH - 50 - cum / bar_max * (BH - 80) - 6:.0f}" '
            f'text-anchor="middle" font-size="10">'
            f"Σsolo {cum:.2f}</text>"
        )
        bars.append(
            f'<text x="{x + bar_w / 2:.0f}" y="{BH - 30}" text-anchor="middle" '
            f'font-size="10">{s.split("(")[0].strip()}</text>'
        )
    legend = []
    lx = 80 + plot_w + 20
    legend.append(
        f'<rect x="{lx - 10}" y="10" width="{LEGEND_W}" '
        f'height="{min(K, 24) * 16 + 18}" fill="#fafafa" '
        f'stroke="#ddd"/>'
    )
    legend.append(
        f'<text x="{lx}" y="24" font-size="11" '
        f'font-weight="600">Trait (by total contribution)</text>'
    )
    for li, t in enumerate(ranked[:24]):
        ly = 32 + li * 16
        disp = q.DISEASE_TRAITS.get(t) or q.CONTINUOUS_TRAITS.get(t)
        label = getattr(disp, "display_name", t) if disp else t
        legend.append(
            f'<rect x="{lx}" y="{ly}" width="12" height="12" '
            f'fill="{palette[t]}"/>'
            f'<text x="{lx + 18}" y="{ly + 10}" font-size="11">'
            f"{label}</text>"
        )
    chart2 = (
        f'<svg width="{BW}" height="{BH}">'
        f'<line x1="70" y1="{BH - 50}" x2="{80 + plot_w}" y2="{BH - 50}" '
        f'stroke="#333"/>'
        f"{''.join(bars)}{''.join(legend)}"
        f'<text x="{80 + plot_w / 2:.0f}" y="{BH - 5}" text-anchor="middle" '
        f'font-size="12">Best-of-{n5} ΔQALY by trait '
        f"(per-trait solo gain)</text></svg>"
    )

    # parameter review table — diseases sorted by current-PGS solo
    # contribution (most impactful first), continuous traits at the end.
    cur5 = results["current PGS"][5]["per_trait_solo"]
    disease_order = sorted(
        q.DISEASE_TRAITS, key=lambda k: -cur5.get(k, {"qaly_gain": 0.0})["qaly_gain"]
    )
    cont_order = sorted(
        q.CONTINUOUS_TRAITS, key=lambda k: -cur5.get(k, {"qaly_gain": 0.0})["qaly_gain"]
    )
    ordered = [(k, q.DISEASE_TRAITS[k]) for k in disease_order] + [
        (k, q.CONTINUOUS_TRAITS[k]) for k in cont_order
    ]
    param_rows = []
    for k, t in ordered:
        solo = cur5.get(k, {"qaly_gain": 0.0, "savings_gain": 0.0})
        gain_cells = (
            f"<td><b>{solo['qaly_gain']:.3f}</b></td>"
            f"<td>${solo['savings_gain']:,.0f}</td>"
        )
        if isinstance(t, q.DiseaseTrait):
            med, prod = t.costs()
            # marginal QALY loss per +1 SD of liability at z=0,
            # for comparison to ContinuousTrait.qaly_per_sd.
            phi_thr = norm.pdf(norm.ppf(1 - t.prevalence))
            qaly_per_sd_lia = -t.qaly_loss_if_affected * phi_thr
            param_rows.append(
                f"<tr><td>{t.display_name}</td><td>disease</td>"
                f"<td>{t.prevalence:.1%}</td>"
                f"<td>{t.qaly_loss_if_affected:.1f} "
                f"<small style='color:#888'>"
                f"({qaly_per_sd_lia:+.2f}/SD)</small></td>"
                f"<td>${med:,.0f} / ${prod:,.0f}</td>"
                f"<td>{t.typical_onset_age:.0f}</td>"
                f"<td>{t.pgs_r2_population:.3f}</td>"
                f"<td>{t.within_family_ratio:.2f}"
                f"{'' if k in WF_RATIO_SOURCED else '*'}</td>"
                f"<td>{H2.get(k, (0, 0, 0))[0]:.2f}</td>"
                f"<td>{H2.get(k, (0, 0, 0))[1]:.2f}</td>"
                f"{gain_cells}</tr>"
            )
        elif isinstance(t, q.ContinuousTrait):
            param_rows.append(
                f"<tr><td>{t.display_name}</td><td>continuous</td>"
                f"<td>—</td>"
                f"<td>{t.qaly_per_sd:+.2f}/SD</td>"
                f"<td>${t.savings_per_sd:+,.0f}/SD</td>"
                f"<td>{t.typical_effect_age:.0f}</td>"
                f"<td>{t.pgs_r2_population:.3f}</td>"
                f"<td>{t.within_family_ratio:.2f}"
                f"{'' if k in WF_RATIO_SOURCED else '*'}</td>"
                f"<td>{H2.get(k, (0, 0, 0))[0]:.2f}</td>"
                f"<td>{H2.get(k, (0, 0, 0))[1]:.2f}</td>"
                f"{gain_cells}</tr>"
            )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Selection gain vs predictor strength</title>
<style>
  body {{ font: 15px/1.5 system-ui, sans-serif; max-width: 1000px;
          margin: 2em auto; color: #222; }}
  h1 {{ font-size: 1.5em; }} h2 {{ font-size: 1.2em; margin-top: 2.2em; }}
  table {{ border-collapse: collapse; font-size: 12px; }}
  th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: right; }}
  th {{ background: #f4f4f4; }} td:first-child {{ text-align: left; }}
  .note {{ color: #555; font-size: 0.93em; max-width: 780px; }}
  .note b {{ color: #222; }}
</style></head><body>
<h1>How much does embryo selection gain as polygenic predictors improve?</h1>

<p class="note" style="background:#fff7e6;border-left:3px solid #e6a800;
padding:0.6em 1em">
<b>European-ancestry only.</b> Every R² figure here comes from
predictors trained on and validated in European-ancestry samples.
In other ancestries, current polygenic scores typically explain
40–80% less variance (because the SNPs that tag causal variants in
Europeans don't tag them as well where LD patterns differ). The
"current PGS" line should be read as roughly 2–5× lower for African
ancestry and 1.5–2× lower for East/South Asian ancestry. The
heritability ceilings (SNP-h², twin h²) are more transferable since
they reflect biology rather than predictor quality, but the
estimates themselves are also mostly from European cohorts.
</p>

<p class="note">
This report asks: if you have N embryos and pick the one with the
best polygenic score profile, how much healthier (in expected
quality-adjusted life-years) is that embryo than a randomly-chosen
sibling? And how does that answer change as the predictors get
better?
</p>
<p class="note">
<b>Method.</b> For each scenario we simulate 50,000 random families.
In each family we draw N sibling embryos whose polygenic scores are
correlated the way real traits are genetically correlated (so an
embryo with a high heart-disease score also tends to have a high
type-2-diabetes score, etc.). We compute each embryo's expected QALY
impact across all traits — for diseases, by converting the score to
an absolute lifetime risk and multiplying by the QALY cost of getting
that disease; for continuous traits like height, by a linear
QALY-per-standard-deviation. We pick the embryo with the highest
total, and record how much better it is than the average sibling.
Future costs are weighted by the probability of surviving to the age
they occur; no additional time-discounting is applied.
</p>

<h2>Expected QALY gain vs number of embryos</h2>
{chart1}
<p class="note">
The four lines are different assumptions about how good the
predictors are:
</p>
<ul class="note">
<li><b>Current PGS</b> — what today's published polygenic scores
actually achieve, after the within-family discount (scores predict
less well between siblings than between strangers, because some of
their population-level signal is environmental confounding that
siblings share).</li>
<li><b>SNP-h² ceiling</b> — what you'd get from an infinitely large
GWAS using today's common-variant SNP arrays. This is the limit of
"more data, same technology."</li>
<li><b>Twin h²</b> — the additive heritability from twin studies.
This is the ceiling for <i>any</i> additive predictor, including ones
that capture rare variants via whole-genome sequencing. It's the
realistic long-run limit. <i>Caveat for psychiatric traits:</i>
twin-study h² for schizophrenia, bipolar, ADHD, and autism is
probably inflated — partly because identical twins share more
environment than the model assumes, and partly because rare de novo
mutations (which contribute a lot to these conditions) are fully
shared by identical twins but not at all by fraternal twins, which
the standard twin model misreads as extra heritability. The
psychiatric segments under "twin h²" should be read as optimistic.</li>
<li><b>MZ correlation</b> — how similar identical twins are. This
includes non-additive genetic effects (dominance, gene-gene
interactions) that a polygenic score can't exploit, so it overstates
what selection could ever do. Shown only as an upper reference.</li>
</ul>
<p class="note">
Reading the chart: today's predictors (grey line) deliver roughly a
quarter of what's theoretically achievable (green line). The gap is
mostly rare genetic variants that current GWAS don't capture.
</p>

<h2>Which traits drive the gain? (best-of-5)</h2>
{chart2}
<p class="note">
Each colored segment shows how much QALY you'd gain if you selected
on that single trait alone. The stacked total is higher than the
dashed "joint" line because in practice traits compete: the embryo
that's best on heart disease usually isn't also best on cognition, so
selecting on a combined index gets you less than the sum of
selecting on each separately.
</p>
<p class="note">
Compare the bars across scenarios to see which traits benefit most
from better predictors. Psychiatric conditions and Alzheimer's grow
disproportionately because their current scores capture only a
small slice of their heritability — there's a lot of headroom.
Height barely grows because today's score is already close to its
ceiling.
</p>

{_aggregate_value_table(results, scenarios)}
<p class="note">
This converts the QALY gain into dollars at <b>${QALY_TO_USD:,} per
QALY</b> (the mid-range US figure for what a year of healthy life is
"worth" in cost-effectiveness analysis; the UK uses about $35k) and
adds the personal cost savings — the medical bills you avoid paying
out-of-pocket plus the income you don't lose to illness. The greyed
societal column is the same calculation using full economic cost
(including what insurers and taxpayers cover).
</p>
<p class="note">
<b>What was removed:</b> earlier versions included a
"longevity_residual" trait that ended up being the single largest
QALY contributor. It's been zeroed out here because there's no
validated longevity polygenic score in the pipeline, and most of
what makes people live longer is already counted via the specific
diseases below — including it separately was double-counting.
</p>
<p class="note">
<b>On overcounting more generally:</b> some traits in this model
cause others — high BMI raises type-2 diabetes and heart disease
risk; cognitive ability raises income. The simulation handles part
of this automatically: it draws each embryo's scores with the
correct genetic correlations, so an embryo with a low BMI score
tends to also have low T2D and CAD scores, and the joint-selection
gain (dashed line in the bar chart) is correctly less than the sum
of per-trait gains. What it does <i>not</i> fully handle is when a
trait's QALY-loss figure already bakes in its downstream effects —
e.g., if "QALY lost to high BMI" includes the diabetes it causes,
and we also separately count diabetes, that's a residual
overcount. Two trait pairs are explicitly residualized to avoid
this: <b>cognition/income</b> (income's $/SD represents only the
earnings effect not mediated by cognition, which is why cognition
is $200k/SD and income is much smaller) and <b>BMI vs
cardiometabolic disease</b> (BMI's QALY/SD is set to the ~45% of
its health burden that is <i>not</i> via T2D/CVD/CKD/cancer —
joint pain, sleep apnea, mobility, psychosocial). The remaining
concern is whether each disease's own QALY-loss figure embeds
comorbidity effects (e.g., does the T2D figure include the CAD it
causes?). Published QALY-loss estimates usually don't fully
disentangle this, so there's likely a residual ~5–10% overcount
across the cardiometabolic cluster.
</p>

<h2>Dollar gain: societal vs personal (best-of-5)</h2>
<table>
<tr><th>Scenario</th><th>Societal $</th><th>Personal $</th>
<th>of which medical</th><th>of which productivity</th></tr>
{
        "".join(
            f"<tr><td style='text-align:left'>{s}</td>"
            f"<td>${results[s][5]['cost_split']['societal']:,.0f}</td>"
            f"<td>${results[s][5]['cost_split']['personal']:,.0f}</td>"
            f"<td>${results[s][5]['cost_split']['medical']:,.0f}</td>"
            f"<td>${results[s][5]['cost_split']['productivity']:,.0f}</td></tr>"
            for s in scenarios
        )
    }
</table>
<p class="note">
The same total cost saving, broken down two ways. <b>Medical vs
productivity</b>: medical is healthcare spending avoided
(hospitalizations, drugs, long-term care); productivity is earnings
not lost to illness or early death. <b>Societal vs personal</b>:
societal is the full amount regardless of who pays; personal is what
the individual themselves avoids paying — their out-of-pocket share
of medical costs (assumed {PERSONAL_MEDICAL_SHARE:.0%}, the US
average) plus their own lost income. Each disease's
medical/productivity ratio comes from published cost-of-illness
studies (cited inline in the parameter table below).
</p>
<p class="note">
One thing this table doesn't yet account for: healthcare costs have
been rising about 2% per year faster than general inflation. The
medical column is in today's dollars, so for a condition that
strikes 40 years from now the real medical cost will be roughly
double what's shown here.
</p>

<h2>Per-trait parameters</h2>
<p class="note">
These are the inputs the model uses for each trait. The QALY and cost
figures come from the disease-burden and cost-of-illness literature;
the R² and heritability columns are what drives the scenario
comparison above. If any number looks wrong to you, it probably
deserves a second look — several of these are rough.
</p>
<table>
<tr><th>Trait</th><th>Type</th><th>Prevalence</th>
<th>QALY loss if affected<br>
<small style='font-weight:normal'>(grey: marginal /SD-liability,<br>
comparable to continuous)</small></th>
<th>Societal cost: med / prod<br><small>(continuous: $/SD)</small></th>
<th>Onset/effect age</th><th>Current R²</th><th>WF ratio</th>
<th>SNP-h²</th><th>Twin h²</th>
<th>ΔQALY (solo, N=5)</th><th>Δ$ (solo, N=5)</th></tr>
{"".join(param_rows)}
</table>
<p class="note">The two right-hand columns show the gain from
selecting on this trait alone, with current PGS, best-of-5.
Diseases are sorted by ΔQALY descending; continuous traits follow.
<b>WF ratio</b> is the within-family attenuation
(within-sibship R² ÷ population R²); values marked * are assumed
defaults (0.75–0.90 for diseases, on the logic that disease
liability has little parental-nurture inflation), unmarked values
are from Howe et al. 2022's sibling-GWAS. <b>Sex-specific cancers</b>
(breast, prostate) use sex-averaged prevalences (~half the per-sex
lifetime risk), since the model doesn't condition on embryo sex.
<b>Twin h²</b> values are deliberately discounted ~10–15% from
headline meta-analytic figures to account for equal-environments
inflation; this is conservative.</p>
</body></html>"""
    Path(out_path).write_text(html)
    print(f"\nHTML → {out_path}", file=sys.stderr)


def main():
    print("Running scenarios (4 × 4 N-values × 50k sims)...", file=sys.stderr)
    results = run_scenarios()

    print(
        f"\n{'scenario':<32} "
        + " ".join(f"N={n:>2}" for n in (2, 3, 5, 10))
        + "    [p10–p90 at N=5]"
    )
    print("-" * 88)
    for s, by_n in results.items():
        vals = " ".join(f"{by_n[n]['qaly_gain_mean']:>5.2f}" for n in (2, 3, 5, 10))
        r5 = by_n[5]
        print(
            f"{s:<32} {vals}  QALY   "
            f"[{r5['qaly_gain_p10']:.2f}–{r5['qaly_gain_p90']:.2f}]"
        )

    # SCZ sensitivity: how much does the headline depend on the 15-QALY
    # schizophrenia assumption?
    scz = q.DISEASE_TRAITS["schizophrenia"]
    saved = scz.qaly_loss_if_affected
    print(
        "\nSensitivity (current PGS, N=5): SCZ qaly_loss 15→10 changes ΔQALY by ",
        end="",
    )
    scz.qaly_loss_if_affected = 10.0
    with _suppress_longevity():
        r_alt = q.simulate_selection(
            n_embryos=5, n_simulations=20_000, use_correlations=True
        )
    scz.qaly_loss_if_affected = saved
    print(
        f"{r_alt['qaly_gain_mean'] - results['current PGS'][5]['qaly_gain_mean']:+.3f}"
    )
    print()
    print(
        f"{'scenario':<32} {'societal $':>12} {'personal $':>12} "
        f"{'medical':>10} {'productivity':>13}  (best-of-5)"
    )
    print("-" * 95)
    for s, by_n in results.items():
        cs = by_n[5]["cost_split"]
        print(
            f"{s:<32} {cs['societal']:>12,.0f} {cs['personal']:>12,.0f} "
            f"{cs['medical']:>10,.0f} {cs['productivity']:>13,.0f}"
        )
    print()
    print(
        "Ratio to current at N=5: "
        + ", ".join(
            f"{s.split()[0]}={results[s][5]['qaly_gain_mean'] / results['current PGS'][5]['qaly_gain_mean']:.1f}×"
            for s in results
        )
    )

    write_html(results, (2, 3, 5, 10))


if __name__ == "__main__":
    main()
