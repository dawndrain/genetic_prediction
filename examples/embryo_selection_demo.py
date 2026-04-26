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
from embryo import (
    score_chrom as score,
    build_hmm_context,
    hmm_recover,
    load_parents,
    load_pgs_for_chrom,
    pick_parents,
    simulate_biopsy,
    simulate_child,
)
from genepred.io import parse_chroms as _parse_chroms
from genepred.paths import resource
from genepred.qaly import liability_threshold_risk

_ARGS: argparse.Namespace  # set in main() before Pool fork; inherited by workers


def _do_chrom(chrom):
    a = _ARGS
    par = load_parents(chrom, a.father, a.mother)
    M = len(par.pos)
    pgs = load_pgs_for_chrom(chrom, par)
    ctx = build_hmm_context(par)
    pmid = score((par.pat.sum(0) + par.mat.sum(0)) / 2, pgs)
    emb = []
    for e in range(a.n_embryos):
        rng = np.random.default_rng((a.seed, e, int(chrom)))
        true_geno, _, _ = simulate_child(par, rng)
        n_ref, n_alt = simulate_biopsy(true_geno, a.coverage, a.seq_err, rng)
        rec_geno, _, _ = hmm_recover(par, ctx, n_ref, n_alt, a.seq_err)
        emb.append(
            (
                int((true_geno == rec_geno).sum()),
                M,
                score(true_geno, pgs),
                score(rec_geno, pgs),
            )
        )
    return chrom, M, len(ctx.inf_idx), set(pgs), pmid, emb


def _qaly_report(embryo_scores, pgs_ids):
    """Convert per-embryo genome-wide raw PGS to z (vs 1KG-EUR), then to
    absolute risk (diseases) or trait shift (continuous), then to QALY,
    and rank embryos."""

    ref = pd.read_csv(resource("1kg_pgs_summary.tsv"), sep="\t")
    eur = ref[ref.super_pop == "EUR"].set_index("pgs_id")[["mean", "sd"]].astype(float)
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
        if pid not in eur.index:
            continue
        mu = float(eur.at[pid, "mean"])  # type: ignore[arg-type]
        sd = float(eur.at[pid, "sd"])  # type: ignore[arg-type]
        if sd <= 0:
            continue
        raw = np.array([s.get(pid, np.nan) for s in embryo_scores], dtype=float)
        z = (raw - mu) / sd
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

    _write_html(detail, qaly_per_emb, best, n_emb, q)


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
    Path(out).write_text(html)
    print(f"\nHTML report → {out}", file=sys.stderr)


def main():
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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.father or not args.mother:
        args.father, args.mother = pick_parents(args.pop)
    chroms = _parse_chroms(args.chroms)
    print(
        f"parents {args.father} × {args.mother} ({args.pop}), "
        f"{args.n_embryos} embryos, {args.coverage}× biopsy, "
        f"chroms {chroms[0]}..{chroms[-1]}",
        file=sys.stderr,
    )

    # Per-embryo accumulators across chromosomes
    true_total = [dict() for _ in range(args.n_embryos)]
    rec_total = [dict() for _ in range(args.n_embryos)]
    parent_mid_total: dict[str, float] = {}
    conc_sum = np.zeros(args.n_embryos)
    conc_n = np.zeros(args.n_embryos)
    all_pids: set[str] = set()

    global _ARGS
    _ARGS = args

    n_proc = min(len(chroms), int(os.environ.get("COO_CPUS") or os.cpu_count() or 8))
    print(f"  {n_proc} chromosome workers", file=sys.stderr)
    t0 = time.time()
    with mp.get_context("fork").Pool(n_proc) as pool:
        for chrom, M, n_inf, pids, pmid, emb in pool.imap_unordered(_do_chrom, chroms):
            print(
                f"  [chr{chrom}] {M:,} sites ({n_inf:,} informative, "
                f"{n_inf / M:.1%}), {len(pids)} scores",
                file=sys.stderr,
            )
            all_pids |= pids
            for pid, v in pmid.items():
                parent_mid_total[pid] = parent_mid_total.get(pid, 0.0) + v
            for e in range(args.n_embryos):
                cs, cn, st, sr = emb[e]
                conc_sum[e] += cs
                conc_n[e] += cn
                for pid, v in st.items():
                    true_total[e][pid] = true_total[e].get(pid, 0.0) + v
                for pid, v in sr.items():
                    rec_total[e][pid] = rec_total[e].get(pid, 0.0) + v
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
        _qaly_report(true_total, sorted(all_pids))
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
        f"(chr{args.chrom} contribution only)"
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


if __name__ == "__main__":
    main()
