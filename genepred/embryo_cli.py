"""End-to-end simulation of the PGT-based embryo polygenic-score workflow.

  1. Load phased parental genomes from 1KG (CEU trio parents NA12891 +
     NA12892 by default — NA12878 is their real daughter, useful as
     ground truth).
  2. Simulate N children via meiosis: per parent, sample crossover
     points (Poisson on the genetic length), build a gamete as a mosaic
     of the two parental haplotypes; child = paternal⊕maternal gamete.
  3. Simulate a PGT-A trophectoderm biopsy: ~0.05× coverage after WGA,
     so most SNPs see 0 reads; each read samples one allele from the
     diploid genotype with sequencing error ε.
  4. Recover the child's haplotype inheritance with a 4-state HMM
     (which paternal × which maternal haplotype), Viterbi over all
     informative SNPs. Read off the full child genotype from the
     inferred path + parental haplotypes.
  5. Score true and recovered child genomes on all curated PGS; report
     genotype concordance, per-PGS recovery error, and the
     between-embryo score spread (the thing selection acts on).

This is the technical core of what commercial embryo-PGS providers do:
parents are deeply genotyped, the embryo is barely sequenced, and the
HMM bridges the gap by exploiting that recombination is rare.

Library functions live in genepred.embryo; this script is the
multi-chromosome / multi-embryo orchestration + reporting around them.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from genepred import qaly as q
from genepred.catalog import CURATED
from genepred.embryo import (
    apply_switch_errors,
    build_hmm_context,
    hmm_recover,
    hmm_recover_switch_aware,
    joint_recover,
    load_embryo_reads,
    load_parents_cached,
    load_parents_vcf,
    grid_effect_af,
    load_pgs_for_chrom,
    pgs_subset_moments,
    pick_parents,
    score_chrom,
    simulate_biopsy,
    simulate_child,
)
from genepred.io import parse_chroms as _parse_chroms
from genepred.paths import pgs_weights_dir, resource
from genepred.qaly import liability_threshold_risk

_ARGS: argparse.Namespace  # set in main() before Pool fork; inherited by workers
_AF_BY_POS: dict | None = None  # EUR AF sidecar by_pos; set before fork


def _load_af_by_pos():
    from genepred.scoring import load_af_sidecar

    lk = load_af_sidecar("EUR")
    if lk is None:
        print(
            "  (no AF sidecar — run `genepred fetch-weights`; z-scores fall "
            "back to full-genome normalization and inflate at partial overlap)",
            file=sys.stderr,
        )
        return None
    return lk[1]


def _do_chrom(chrom):
    a = _ARGS
    par_true = load_parents_cached(chrom, a.father, a.mother)
    M = len(par_true.pos)
    pgs = load_pgs_for_chrom(chrom, par_true)
    mom = (
        pgs_subset_moments(pgs, grid_effect_af(par_true, _AF_BY_POS))
        if _AF_BY_POS is not None
        else {}
    )
    pmid = score_chrom((par_true.pat.sum(0) + par_true.mat.sum(0)) / 2, pgs)

    rng_sw = np.random.default_rng((a.seed, int(chrom), 999))
    par_obs, _, _ = apply_switch_errors(par_true, a.switch_error_rate, rng_sw)
    ctx = build_hmm_context(par_obs)

    truths, biops = [], []
    for e in range(a.n_embryos):
        rng = np.random.default_rng((a.seed, e, int(chrom)))
        true_geno, _, _ = simulate_child(par_true, rng)
        truths.append(true_geno)
        biops.append(
            simulate_biopsy(
                true_geno, a.coverage, a.seq_err, rng,
                ado=a.ado, cov_dispersion=a.cov_dispersion,
            )
        )

    if a.method == "naive":
        recs = [
            hmm_recover(par_obs, ctx, *biops[e], a.seq_err)[0]
            for e in range(a.n_embryos)
        ]
    elif a.method == "switch_aware":
        recs = [
            hmm_recover_switch_aware(
                par_obs, *biops[e], a.seq_err,
                switch_rate=max(a.switch_error_rate, 1e-4),
            )[0]
            for e in range(a.n_embryos)
        ]
    elif a.method == "joint":
        _, dose, vdose, _, _ = joint_recover(
            par_obs, biops, a.seq_err,
            switch_rate=max(a.switch_error_rate, 1e-4), n_iter=2,
        )
        recs = list(dose)
        vdoses = list(vdose)
    else:
        raise ValueError(f"unknown --method {a.method}")

    emb = []
    for e in range(a.n_embryos):
        rec_geno = recs[e]
        # Per-PGS variance contribution from this chromosome
        # (Σ wᵢ²·Var[dᵢ]); zero for hard-call methods.
        var_chrom = (
            {pid: float((w**2 * vdoses[e][idx]).sum()) for pid, (idx, _, w) in pgs.items()}
            if a.method == "joint"
            else {pid: 0.0 for pid in pgs}
        )
        emb.append(
            (
                int((truths[e] == np.round(rec_geno)).sum()),
                M,
                score_chrom(truths[e], pgs),
                score_chrom(rec_geno, pgs),
                var_chrom,
            )
        )
    return chrom, M, len(ctx.inf_idx), set(pgs), pmid, emb, mom


def _accumulate_moments(mom_total: dict, mom: dict) -> None:
    for pid, vals in mom.items():
        t = mom_total.setdefault(pid, [0.0, 0.0, 0, 0])
        for k, v in enumerate(vals):
            t[k] += v


def _full_file_moments(pids):
    """pid -> (exp, var, n_af) over the ENTIRE weight file, frequency-
    implied via the AF sidecar (score_one on an empty genome)."""
    from genepred.scoring import load_af_sidecar, score_one

    from genepred.catalog import read_header

    lk = load_af_sidecar("EUR")
    if lk is None:
        return {}
    out = {}
    for wf in sorted(pgs_weights_dir().glob("*_hmPOS_GRCh38.txt.gz")):
        pid = wf.name.split("_")[0]
        if pid not in pids:
            continue
        # Mirror load_pgs_for_chrom's build filter so the full-file moments
        # describe the same file the raw sums were scored from.
        if read_header(wf).get("genome_build", "GRCh37") not in (
            "GRCh37", "hg19", "NCBI37",
        ):
            continue
        r = score_one({}, {}, wf, af_lookup=lk)
        out[pid] = (r["expected"], r["var_all"], r["n_af"])
    return out


def _build_normalizers(pids, moments) -> dict[str, tuple[float, float]]:
    """pid -> (center, scale) for converting partial-grid raw sums to z vs
    1KG-EUR. With AF-sidecar moments for the scored subset, centers on the
    subset's frequency-implied mean (plus the matched share of the
    full-score residual) and scales by the matched share of the
    frequency-implied variance — the same partial-overlap correction
    genepred.scoring applies to arrays. Falls back to the raw full-genome
    (mean, sd), which inflates |z| by multiple sigma at partial overlap."""
    ref = pd.read_csv(resource("1kg_pgs_summary.tsv"), sep="\t")
    eur = ref[ref.super_pop == "EUR"].set_index("pgs_id")
    full = _full_file_moments({p for p in pids if p in (moments or {})})
    out: dict[str, tuple[float, float]] = {}
    fallback = []
    for pid in pids:
        if pid not in eur.index:
            continue
        mu = float(eur.at[pid, "mean"])
        sd = float(eur.at[pid, "sd"])
        if sd <= 0:
            continue
        m = (moments or {}).get(pid)
        fm = full.get(pid)
        # Require the sidecar to cover (nearly) every SNP in the raw sum —
        # uncovered scored SNPs contribute to raw but not to the center,
        # which would re-create the constant-offset artifact.
        if (
            m
            and fm
            and m[2] > 0
            and m[1] > 0
            and fm[1] > 0
            and m[2] >= 0.95 * m[3]
        ):
            exp_m, var_m, n_m = m[0], m[1], m[2]
            exp_all, var_all, n_all = fm
            n_ref = (
                float(eur.at[pid, "n_snps"])
                if "n_snps" in eur.columns and float(eur.at[pid, "n_snps"]) > 0
                else float(n_all or n_m)
            )
            f = min(n_m / n_ref, 1.0)
            # The residual term charges the gap between the reference's
            # empirical mean and the frequency-implied mean to the subset.
            # That is only valid when the frequency sum spans (almost) the
            # same SNPs the reference scored; otherwise the gap is mostly
            # the missing SNPs' mean contribution and double-charges the
            # subset (prostate: 394 of 444 -> a constant -3 sigma). Same
            # 95% gate genepred.scoring uses.
            gap = (mu - exp_all) if n_all >= 0.95 * n_ref else 0.0
            out[pid] = (exp_m + f * gap, sd * (var_m / var_all) ** 0.5)
        else:
            out[pid] = (mu, sd)
            fallback.append(pid)
    if fallback:
        print(
            "  (full-genome normalization for "
            + ", ".join(sorted(fallback))
            + " — no AF moments; |z| inflates at partial overlap)",
            file=sys.stderr,
        )
    return out


def _ranking_confidence(
    rec_total, var_total, pids, args, norms=None, n_draws: int = 500
):
    """Sample each embryo's raw PGS from N(point, √var), convert to
    z vs 1KG-EUR, run through the QALY model, and report how often
    each embryo comes out best across draws."""
    if norms is None:
        norms = _build_normalizers(pids, None)
    id2t = {s.pgs_id: t for t, s in CURATED.items()}
    id2t["COGNITION"] = "cognitive_ability"  # keep parity with _qaly_report

    usable = [p for p in pids if p in norms and id2t.get(p) is not None]
    if not usable:
        return
    n = args.n_embryos
    rng = np.random.default_rng(args.seed)
    qaly_draws = np.empty((n_draws, n))
    for d in range(n_draws):
        for e in range(n):
            zsc = {}
            for pid in usable:
                raw = rng.normal(
                    rec_total[e].get(pid, 0.0), np.sqrt(var_total[e].get(pid, 0.0))
                )
                center, scale = norms[pid]
                zsc[id2t[pid]] = (raw - center) / scale
            qaly_draws[d, e] = q.compute_all(zsc)["total_qaly_delta"]
    top = np.bincount(qaly_draws.argmax(1), minlength=n)
    mu, sd = qaly_draws.mean(0), qaly_draws.std(0)
    print(
        f"\nRanking confidence — {n_draws} posterior draws of imputed "
        f"dosage → QALY ranking:"
    )
    print("  embryo | P(best) | ΔQALY vs sib-mean (mean ± SD across draws)")
    for e in np.argsort(-top):
        print(
            f"   e{e + 1:<4} | {top[e] / n_draws:>6.1%} | "
            f"{mu[e] - mu.mean():+7.3f} ± {sd[e]:.3f}"
        )
    print(
        "  (Imputation SE assumes per-site independence — a lower bound; "
        "see genepred.embryo.score_with_uncertainty.)"
    )


def _qaly_report(embryo_scores, pgs_ids, norms=None, html_out=None):
    """Convert per-embryo genome-wide raw PGS to z (vs 1KG-EUR, with the
    partial-overlap correction when AF moments for the scored subset are
    available), then to absolute risk (diseases) or trait shift
    (continuous), then to QALY, and rank embryos."""

    if norms is None:
        norms = _build_normalizers(pgs_ids, None)
    id2t = {s.pgs_id: t for t, s in CURATED.items()}
    id2t["COGNITION"] = "cognitive_ability"

    n_emb = len(embryo_scores)
    print(f"\n{'=' * 120}")
    print(
        "GENOME-WIDE EMBRYO REPORT — z vs 1KG-EUR, "
        "liability-threshold risk, ΔQALY vs sibling mean"
    )
    print(f"{'=' * 120}")

    qaly_per_emb = np.zeros(n_emb)
    detail = []
    for pid in pgs_ids:
        if pid not in norms:
            continue
        center, scale = norms[pid]
        raw = np.array([s.get(pid, np.nan) for s in embryo_scores], dtype=float)
        z = (raw - center) / scale
        tk = id2t.get(pid)
        if tk in q.DISEASE_TRAITS:
            dt = q.DISEASE_TRAITS[tk]
            risk = np.array(
                [
                    liability_threshold_risk(zi, dt.prevalence, dt.pgs_r2_population)
                    for zi in z
                ]
            )
            dq = -(risk - risk.mean()) * dt.qaly_loss_if_affected
            detail.append((dt.display_name, pid, z, risk, dq, "disease"))
        elif tk in q.CONTINUOUS_TRAITS:
            ct = q.CONTINUOUS_TRAITS[tk]
            shift = z * np.sqrt(ct.pgs_r2_population)
            dq = (shift - shift.mean()) * ct.qaly_per_sd
            detail.append((ct.display_name, pid, z, shift, dq, "cont"))
        else:
            continue
        qaly_per_emb += detail[-1][4]

    hdr = f"{'trait':<24} {'pgs':<11} " + "".join(
        f"  e{i + 1}:z   risk/σ  ΔQALY " for i in range(n_emb)
    )
    print(hdr)
    print("-" * len(hdr))
    for name, pid, z, val, dq, _ in sorted(detail, key=lambda d: -np.abs(d[4]).max()):
        cells = "".join(
            f" {z[i]:+5.2f} {val[i]:7.3f} {dq[i]:+6.3f}" for i in range(n_emb)
        )
        print(f"{name:<24} {pid:<11}{cells}")

    print("-" * len(hdr))
    print(
        f"{'TOTAL ΔQALY vs sib-mean':<36}"
        + "".join(f"{'':14}{qaly_per_emb[i]:+6.3f}" for i in range(n_emb))
    )
    best = int(np.argmax(qaly_per_emb))
    print(
        f"\n→ Selected embryo: e{best + 1} "
        f"(+{qaly_per_emb[best] - qaly_per_emb.mean():.3f} QALY "
        f"vs sibling mean, +{qaly_per_emb[best] - qaly_per_emb.min():.3f} "
        f"vs worst)"
    )

    _write_html(
        detail, qaly_per_emb, best, n_emb, q,
        out=html_out or "docs/embryo_report.html",
    )


def _write_html(detail, qaly_per_emb, best, n_emb, q, out="docs/embryo_report.html"):
    # Sort by expected selection impact: SD of ΔQALY across embryos.
    # A trait with high QALY-spread is one where embryo choice matters most.
    detail = sorted(detail, key=lambda d: -float(np.std(d[4])))
    max_dq = max((float(np.abs(d[4]).max()) for d in detail), default=1.0)

    def cell_color(dq):
        a = min(abs(dq) / max(max_dq, 1e-6), 1.0) ** 0.7
        return (
            f"background:rgba(40,160,80,{a:.2f})"
            if dq > 0
            else f"background:rgba(210,70,60,{a:.2f})"
        )

    cont_dir = {
        t: ("↑" if ct.qaly_per_sd > 0 else "↓") for t, ct in q.CONTINUOUS_TRAITS.items()
    }

    rows = []
    for name, pid, z, val, dq, kind in detail:
        unreliable = bool(np.any(np.abs(z) > 5))
        impact = float(np.std(dq))
        dir_tag = ""
        if kind == "cont":
            tk = next(
                (t for t, ct in q.CONTINUOUS_TRAITS.items() if ct.display_name == name),
                None,
            )
            arrow = cont_dir.get(tk, "")
            dir_tag = f' <span title="higher is better">{arrow}</span>' if arrow else ""
        warn = (
            ' <span class="warn" title="z outside ±5 — reference '
            "distribution likely incompatible (different SNP set or "
            'build); ΔQALY shown but treat as unreliable">⚠</span>'
            if unreliable
            else ""
        )
        cls = ' class="unrel"' if unreliable else ""
        cells = "".join(
            f'<td style="{cell_color(dq[i])}"{cls}>'
            f"<b>{'risk ' + format(val[i], '.1%') if kind == 'disease' else format(val[i], '+.2f') + 'σ for trait'}</b><br>"
            f"<small>{z[i]:+.2f}σ raw</small><br>"
            f'<small class="dq">{dq[i]:+.3f} QALY</small></td>'
            for i in range(n_emb)
        )
        rows.append(
            f'<tr><td class="trait">{name}{dir_tag}{warn}<br>'
            f"<small>{pid} · impact {impact:.3f}</small></td>{cells}</tr>"
        )

    qrow = "".join(
        f'<td class="{"best" if i == best else ""}"><b>{qaly_per_emb[i]:+.3f}</b></td>'
        for i in range(n_emb)
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Embryo PGS report</title>
<style>
  body {{ font: 14px -apple-system, system-ui, sans-serif; max-width: 1200px;
          margin: 2em auto; color: #222; }}
  h1 {{ font-size: 1.4em; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 1em; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: center; }}
  th {{ background: #f4f4f4; }}
  td.trait {{ text-align: left; font-weight: 600; background: #fafafa; }}
  td.trait small {{ font-weight: normal; color: #888; }}
  td.unrel {{ opacity: 0.45; }}
  .warn {{ color: #c70; cursor: help; }}
  small.dq {{ color: #555; }}
  tr.total td {{ background: #fffbe6; font-size: 1.1em; }}
  td.best {{ outline: 3px solid #2a2; outline-offset: -3px; }}
  .legend {{ font-size: 0.9em; color: #555; }}
  .legend span {{ display: inline-block; width: 1em; height: 1em;
                  vertical-align: middle; margin: 0 0.3em; }}
</style></head><body>
<h1>Simulated embryo PGS report</h1>
<p class="legend">
{n_emb} embryos from 1KG CEU parents, 0.05× biopsy coverage, HMM-recovered
genotypes scored on {len(detail)} traits. Rows are sorted by
<b>expected selection impact</b> (SD of ΔQALY across embryos — traits where
your choice matters most are at the top). Each cell: implied risk (diseases)
or trait shift in SD (continuous; ↑ marks traits where higher is better),
PGS z-score vs 1KG-EUR, and ΔQALY relative to the sibling mean. Cell shade
is by <b>ΔQALY</b>:
<span style="background:rgba(40,160,80,0.6)"></span> better than sib-mean,
<span style="background:rgba(210,70,60,0.6)"></span> worse — so green is
always good regardless of trait direction. ⚠ marks rows where the reference
distribution is incompatible (z outside ±5); those cells are dimmed.
The ΔQALY total ranks embryos; the selected one is outlined.
<br><b>Caveats:</b> z-scores are population-relative (1KG-EUR), not
PC-adjusted; ΔQALY uses population R² (within-family is lower); this is a
methods demo on simulated data, not clinical guidance.
</p>
<table>
<tr><th>Trait</th>{"".join(f"<th>Embryo {i + 1}</th>" for i in range(n_emb))}</tr>
{"".join(rows)}
<tr class="total"><td class="trait">TOTAL ΔQALY vs sib mean</td>{qrow}</tr>
</table>
</body></html>"""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(html)
    print(f"\nHTML report → {out}", file=sys.stderr)


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroms", default="22", help="e.g. '22' or '1-22' or '1,5,22'")
    ap.add_argument("--father", default=None)
    ap.add_argument("--mother", default=None)
    ap.add_argument(
        "--pop", default="CEU", help="if father/mother unset, pick from this 1KG pop"
    )
    ap.add_argument("--n-embryos", type=int, default=5)
    ap.add_argument(
        "--coverage",
        type=float,
        default=0.05,
        help="mean sequencing depth of the biopsy",
    )
    ap.add_argument("--seq-err", type=float, default=0.01)
    ap.add_argument(
        "--ado",
        type=float,
        default=0.0,
        help="WGA allelic-dropout rate per het site (MDA ≈ 0.1–0.25)",
    )
    ap.add_argument(
        "--cov-dispersion",
        type=float,
        default=0.0,
        help="WGA coverage coefficient-of-variation (MDA ≈ 0.5–1)",
    )
    ap.add_argument(
        "--switch-error-rate",
        type=float,
        default=0.0,
        help="parental phasing switch-error rate per consecutive het "
        "(0 = perfect phase; 0.005–0.02 is realistic for statistical phasing)",
    )
    ap.add_argument(
        "--method",
        choices=["naive", "switch_aware", "joint"],
        default="naive",
        help="naive = original 4-state HMM; switch_aware = same HMM with "
        "inflated transition prior; joint = 2^E-state HMM pooling all embryos",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if not args.father or not args.mother:
        args.father, args.mother = pick_parents(args.pop)
    chroms = _parse_chroms(args.chroms)
    print(
        f"parents {args.father} × {args.mother} ({args.pop}), "
        f"{args.n_embryos} embryos, {args.coverage}× biopsy, "
        f"SER={args.switch_error_rate}, method={args.method}, "
        f"chroms {chroms[0]}..{chroms[-1]}",
        file=sys.stderr,
    )

    # Per-embryo accumulators across chromosomes
    true_total = [dict() for _ in range(args.n_embryos)]
    rec_total = [dict() for _ in range(args.n_embryos)]
    var_total = [dict() for _ in range(args.n_embryos)]
    parent_mid_total: dict[str, float] = {}
    conc_sum = np.zeros(args.n_embryos)
    conc_n = np.zeros(args.n_embryos)
    all_pids: set[str] = set()

    global _ARGS, _AF_BY_POS
    _ARGS = args
    _AF_BY_POS = _load_af_by_pos()

    n_proc = min(len(chroms), int(os.environ.get("COO_CPUS") or os.cpu_count() or 8))
    print(f"  {n_proc} chromosome workers", file=sys.stderr)
    t0 = time.time()
    mom_total: dict[str, list[float]] = {}
    with mp.get_context("fork").Pool(n_proc) as pool:
        for chrom, M, n_inf, pids, pmid, emb, mom in pool.imap_unordered(_do_chrom, chroms):
            print(
                f"  [chr{chrom}] {M:,} sites ({n_inf:,} informative, "
                f"{n_inf / M:.1%}), {len(pids)} scores",
                file=sys.stderr,
            )
            all_pids |= pids
            for pid, v in pmid.items():
                parent_mid_total[pid] = parent_mid_total.get(pid, 0.0) + v
            _accumulate_moments(mom_total, mom)
            for e in range(args.n_embryos):
                cs, cn, st, sr, sv = emb[e]
                conc_sum[e] += cs
                conc_n[e] += cn
                for pid, v in st.items():
                    true_total[e][pid] = true_total[e].get(pid, 0.0) + v
                for pid, v in sr.items():
                    rec_total[e][pid] = rec_total[e].get(pid, 0.0) + v
                for pid, v in sv.items():
                    var_total[e][pid] = var_total[e].get(pid, 0.0) + v
    print(f"  all chroms done in {time.time() - t0:.1f}s", file=sys.stderr)

    rows = [
        (e + 1, 0, conc_sum[e] / conc_n[e], true_total[e], rec_total[e])
        for e in range(args.n_embryos)
    ]
    parent_mid = parent_mid_total
    pgs = {
        pid: (np.array([0]), np.array([1]), np.array([0.0])) for pid in all_pids
    }  # placeholder for the per-chrom report block

    print(
        "\ngenome-wide genotype concordance per embryo: "
        + " ".join(
            f"e{e + 1}={conc_sum[e] / conc_n[e]:.4%}" for e in range(args.n_embryos)
        )
    )

    if len(chroms) > 1:
        moments = {p: tuple(v) for p, v in mom_total.items()}
        norms = _build_normalizers(sorted(all_pids), moments)
        _qaly_report(true_total, sorted(all_pids), norms)
        if args.method == "joint":
            _ranking_confidence(
                rec_total, var_total, sorted(all_pids), args, norms
            )
        print("\n--- recovery fidelity (true vs HMM-recovered, genome-wide) ---")
        for pid in sorted(all_pids):
            t = np.array([true_total[e].get(pid, 0) for e in range(args.n_embryos)])
            r = np.array([rec_total[e].get(pid, 0) for e in range(args.n_embryos)])
            if t.std() > 0:
                rc = float(np.corrcoef(t, r)[0, 1])
                print(
                    f"  {pid:<11} rank-cor={rc:+.3f} "
                    f"rmse/sd={np.sqrt(((t - r) ** 2).mean()) / t.std():.3f}"
                )
        return

    id2t = {s.pgs_id: t for t, s in CURATED.items()}

    pids = sorted(pgs, key=lambda p: -len(pgs[p][0]))
    print(f"\n{'=' * 100}")
    print(
        f"PGS spread across {args.n_embryos} embryos "
        f"(chr{chroms[0]} contribution only)"
    )
    print(f"{'=' * 100}")
    print(
        f"{'trait':<24} {'pgs_id':<11} {'snps':>6} "
        f"{'parent_mid':>11} | "
        + " ".join(f"e{i + 1:>5}" for i in range(args.n_embryos))
        + " | rec_err(rmse)"
    )
    print("-" * 100)
    for pid in pids[:14]:
        n = len(pgs[pid][0])
        true = np.array([r[3][pid] for r in rows])
        rec = np.array([r[4][pid] for r in rows])
        rmse = float(np.sqrt(((true - rec) ** 2).mean()))
        sd = float(true.std(ddof=1)) if len(true) > 1 else 0
        print(
            f"{id2t.get(pid, pid):<24} {pid:<11} {n:>6} "
            f"{parent_mid[pid]:>11.4f} | "
            + " ".join(f"{v:>+6.3f}" for v in (true - parent_mid[pid]))
            + f" | {rmse:.4f} (sd={sd:.4f})"
        )

    print(
        "\nSelection check (would you pick the same embryo from "
        "recovered scores as from true scores?):"
    )
    for pid in pids[:6]:
        true = np.array([r[3][pid] for r in rows])
        rec = np.array([r[4][pid] for r in rows])
        rank_corr = float(np.corrcoef(true, rec)[0, 1]) if len(true) > 1 else 1
        same_best = int(np.argmax(true)) == int(np.argmax(rec))
        print(
            f"  {id2t.get(pid, pid):<24} rank-cor(true,rec)={rank_corr:+.3f}  "
            f"same top embryo: {'yes' if same_best else 'NO'}"
        )


def score_real_embryos(
    parent_vcf: Path,
    father: str,
    mother: str,
    embryo_vcfs: list[Path],
    *,
    chroms: list[str] | None = None,
    embryo_samples: list[str] | None = None,
    switch_rate: float = 0.01,
    seq_err: float = 0.01,
    html_out: str | None = "docs/embryo_report.html",
) -> tuple[list[dict[str, float]], list[dict[str, float]], set[str]]:
    """Run the joint phase-aware HMM on actual parent + embryo data.

    parent_vcf: a single phased VCF/VCF.gz containing both parents
        (e.g. SHAPEIT5/Beagle output, or long-read WhatsHap).
    embryo_vcfs: one low-coverage VCF per embryo with FORMAT/AD
        (or RO/AO) allele-depth fields — typical output of a
        bcftools-mpileup or DeepVariant low-pass run.
    switch_rate: assumed parental phasing SER per consecutive het.
        ~0.001 for long-read/trio-phased parents, ~0.005–0.02 for
        statistical phasing depending on panel and ancestry.

    Returns (rec_total, var_total, pgs_ids) — per-embryo genome-wide
    raw PGS and its imputation variance — and writes the QALY/HTML
    report."""
    chroms = chroms or [str(c) for c in range(1, 23)]
    n_emb = len(embryo_vcfs)
    samples = embryo_samples or [None] * n_emb
    rec_total: list[dict[str, float]] = [dict() for _ in range(n_emb)]
    var_total: list[dict[str, float]] = [dict() for _ in range(n_emb)]
    all_pids: set[str] = set()
    cov_est: list[float] = []
    af_by_pos = _load_af_by_pos()
    mom_total: dict[str, list[float]] = {}

    for chrom in chroms:
        t0 = time.time()
        par = load_parents_vcf(parent_vcf, chrom, father, mother)
        het = (par.pat[0] != par.pat[1]) | (par.mat[0] != par.mat[1])
        pgs = load_pgs_for_chrom(chrom, par)
        all_pids |= set(pgs)
        if af_by_pos is not None:
            _accumulate_moments(
                mom_total, pgs_subset_moments(pgs, grid_effect_af(par, af_by_pos))
            )
        biops = [
            load_embryo_reads(ev, par, samples[e])
            for e, ev in enumerate(embryo_vcfs)
        ]
        for nr, na in biops:
            cov_est.append(float((nr + na).mean()))
        _, dose, vdose, _, _ = joint_recover(
            par, biops, seq_err, switch_rate=switch_rate, n_iter=2
        )
        for e in range(n_emb):
            for pid, v in score_chrom(dose[e], pgs).items():
                rec_total[e][pid] = rec_total[e].get(pid, 0.0) + v
            for pid, (idx, _, w) in pgs.items():
                var_total[e][pid] = var_total[e].get(pid, 0.0) + float(
                    (w**2 * vdose[e, idx]).sum()
                )
        print(
            f"  [chr{chrom}] {len(par.pos):,} sites ({het.sum():,} parent-het), "
            f"{len(pgs)} PGS, mean embryo depth "
            f"{np.mean([(nr + na).mean() for nr, na in biops]):.3f}× "
            f"({time.time() - t0:.1f}s)",
            file=sys.stderr,
        )

    print(
        f"\n{n_emb} embryos, mean coverage {np.mean(cov_est):.3f}×, "
        f"assumed parental SER {switch_rate}",
        file=sys.stderr,
    )
    moments = {p: tuple(v) for p, v in mom_total.items()}
    norms = _build_normalizers(sorted(all_pids), moments)
    _qaly_report(rec_total, sorted(all_pids), norms, html_out=html_out)
    args = argparse.Namespace(n_embryos=n_emb, seed=0)
    _ranking_confidence(rec_total, var_total, sorted(all_pids), args, norms)
    return rec_total, var_total, all_pids


def main_real(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(
        description="Score real embryo biopsies against phased parental "
        "haplotypes. See docs/PHASING.md for how to phase the parents."
    )
    ap.add_argument(
        "--parent-vcf",
        required=True,
        help="phased VCF(.gz) containing both parents",
    )
    ap.add_argument("--father", required=True, help="sample ID of the father")
    ap.add_argument("--mother", required=True, help="sample ID of the mother")
    ap.add_argument(
        "--embryos",
        required=True,
        nargs="+",
        help="low-coverage VCF(.gz) per embryo with FORMAT/AD (or RO/AO)",
    )
    ap.add_argument(
        "--embryo-samples",
        nargs="+",
        default=None,
        help="sample IDs within each embryo VCF (default: first sample)",
    )
    ap.add_argument("--chroms", default="1-22")
    ap.add_argument(
        "--switch-rate",
        type=float,
        default=0.01,
        help="assumed parental SER per consecutive het; ~0.001 for "
        "long-read/trio-phased parents, ~0.005–0.02 for statistical phasing",
    )
    ap.add_argument("--seq-err", type=float, default=0.01)
    ap.add_argument("--html-out", default="docs/embryo_report.html")
    a = ap.parse_args(argv)
    score_real_embryos(
        Path(a.parent_vcf),
        a.father,
        a.mother,
        [Path(p) for p in a.embryos],
        chroms=_parse_chroms(a.chroms),
        embryo_samples=a.embryo_samples,
        switch_rate=a.switch_rate,
        seq_err=a.seq_err,
        html_out=a.html_out,
    )


if __name__ == "__main__":
    main()
