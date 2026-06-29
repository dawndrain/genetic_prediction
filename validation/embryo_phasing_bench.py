"""Benchmark embryo-genome imputation under realistic parental phasing error.

The embryo demo (`genepred embryo-demo`) originally assumed the parents'
haplotypes are perfectly phased. In practice they are phased
statistically, which introduces *switch errors* — points where the
hap0/hap1 labels flip. At a switch-error rate (SER) of 1 % per
consecutive het pair, a parent has ~10²–10³ switches per chromosome,
versus ~1 meiotic crossover. The 4-state HMM's recombination prior is
far too tight to follow that many flips, so it mis-tracks the
inheritance path and corrupts the imputed embryo genotype.

This script measures that damage and compares three recoveries:

  oracle        4-state HMM with the *true* parental phase
                (unrealistic upper bound).
  naive         4-state HMM with the mis-phased parents and the
                recombination prior — what the original demo does.
  switch-aware  Same per-embryo HMM but with the transition prior
                inflated to recomb + switch. Cheapest fix.
  joint         2^E-state HMM over all embryos that separates shared
                parental switches from embryo-specific recombination,
                with forward-backward posterior decoding and
                coordinate-ascent between the two parents
                (genepred.embryo.joint_recover).

For each (SER, coverage, n_embryos) cell it reports:
  - genotype concordance at informative (parent-het) sites
  - per-site dosage RMSE at informative sites
  - PGS recovery: r² between true and imputed per-embryo scores,
    and Spearman rank correlation (selection fidelity)

Run from the repo root:
    python validation/embryo_phasing_bench.py --chroms 22 --reps 3
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

# Allow running as `python validation/embryo_phasing_bench.py` without an
# installed genepred.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genepred import embryo as E
from genepred.io import parse_chroms


@dataclass
class Result:
    method: str
    ser: float
    cov: float
    n_emb: int
    conc_het: list[float] = field(default_factory=list)
    rmse_het: list[float] = field(default_factory=list)
    pgs_r2: dict[str, list[float]] = field(default_factory=dict)
    pgs_rank: dict[str, list[float]] = field(default_factory=dict)
    wall_s: list[float] = field(default_factory=list)

    def key(self):
        return (self.method, self.ser, self.cov, self.n_emb)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _score_all(genos: np.ndarray, pgs) -> dict[str, np.ndarray]:
    """genos: (E, M) hard or soft dosage. Returns {pid: (E,) raw score}."""
    out: dict[str, np.ndarray] = {}
    for pid, (idx, sgn, w) in pgs.items():
        d = np.where(sgn[None, :] == 1, genos[:, idx], 2 - genos[:, idx])
        out[pid] = (w[None, :] * d).sum(1)
    return out


def run_cell(
    par_true: E.Parents,
    pgs,
    het_mask: np.ndarray,
    ser: float,
    cov: float,
    n_emb: int,
    rep: int,
    seq_err: float,
    methods: list[str],
    n_sibs: int = 0,
    ado: float = 0.0,
    cov_disp: float = 0.0,
    gp_phase: str = "none",
    gp_err: float = 0.005,
):
    rng_sw = np.random.default_rng((101, rep, int(ser * 1e6)))
    par_obs, _, _ = E.apply_switch_errors(par_true, ser, rng_sw)

    # Grandparent anchoring: simulate the chosen side's grandparents from
    # the TRUE haplotypes (genotypes are phase-independent, so simulating
    # from truth is exact), then anchor-correct the switch-errored phase.
    # A corrected parent runs the joint HMM at a near-zero switch rate.
    sw_joint: float | tuple[float, float] = max(ser, 1e-4)
    if gp_phase != "none":
        rng_gp = np.random.default_rng((606, rep))
        rates = {"pat": max(ser, 1e-4), "mat": max(ser, 1e-4)}
        for side in ["pat"] if gp_phase == "father" else ["pat", "mat"]:
            hap_true = getattr(par_true, side)
            gp1, gp2 = E.simulate_grandparents(hap_true, rng_gp, geno_err=gp_err)
            hap_fix, _ = E.anchor_correct_phase(
                getattr(par_obs, side), hap_true.sum(0), gp1, gp2,
                switch_prior=max(ser, 1e-3),
            )
            par_obs = replace(par_obs, **{side: hap_fix})
            # Fixed near-zero rate for the corrected parent. Deriving it
            # from info["residual_anchor_disagreement"] was tried and is
            # WORSE (98.5% vs 99.4% het-conc at 2 embryos): the residual
            # reflects grandparent genotyping noise at point anchors, not
            # remaining long-range switches, so it over-loosens the prior.
            rates[side] = 1e-4
        sw_joint = (rates["pat"], rates["mat"])

    truth = np.empty((n_emb, len(par_true.pos)), dtype=np.int8)
    biop: list[tuple[np.ndarray, np.ndarray]] = []
    for e in range(n_emb):
        g, _, _ = E.simulate_child(par_true, np.random.default_rng((202, rep, e)))
        truth[e] = g
        biop.append(
            E.simulate_biopsy(
                g, cov, seq_err, np.random.default_rng((303, rep, e, int(cov * 1e6))),
                ado=ado, cov_dispersion=cov_disp,
            )
        )

    # Born siblings: extra children of the same couple at 30× — appended
    # to the joint HMM so their reads pin the parental switch track.
    sib_biop: list[tuple[np.ndarray, np.ndarray]] = []
    for s in range(n_sibs):
        sg, _, _ = E.simulate_child(par_true, np.random.default_rng((404, rep, s)))
        sib_biop.append(
            E.simulate_biopsy(sg, 30.0, seq_err, np.random.default_rng((505, rep, s)))
        )

    true_pgs = _score_all(truth.astype(np.float64), pgs)
    ctx_true = E.build_hmm_context(par_true)
    ctx_obs = E.build_hmm_context(par_obs)

    out = []
    for m in methods:
        t0 = time.time()
        if m == "oracle":
            rec = np.stack(
                [E.hmm_recover(par_true, ctx_true, *biop[e], seq_err)[0] for e in range(n_emb)]
            ).astype(np.float64)
        elif m == "naive":
            rec = np.stack(
                [E.hmm_recover(par_obs, ctx_obs, *biop[e], seq_err)[0] for e in range(n_emb)]
            ).astype(np.float64)
        elif m == "switch_aware":
            rec = np.stack(
                [
                    E.hmm_recover_switch_aware(
                        par_obs, *biop[e], seq_err, switch_rate=max(ser, 1e-3)
                    )[0]
                    for e in range(n_emb)
                ]
            ).astype(np.float64)
        elif m == "joint":
            _, rec, _, _, _ = E.joint_recover(
                par_obs, biop + sib_biop, seq_err,
                switch_rate=sw_joint, n_iter=2,
            )
            rec = rec[:n_emb]
        elif m == "joint_full":
            _, rec, _, _, _ = E.joint_recover_full(
                par_obs, biop + sib_biop, seq_err, switch_rate=sw_joint,
            )
            rec = rec[:n_emb]
        elif m == "rephase":
            rec, _, _, _ = E.joint_rephase_recover(
                par_obs, biop, seq_err, switch_rate=sw_joint
            )
            rec = rec.astype(np.float64)
        else:
            raise ValueError(m)
        wall = time.time() - t0

        diff = rec[:, het_mask] - truth[:, het_mask]
        conc = float((np.round(rec[:, het_mask]) == truth[:, het_mask]).mean())
        rmse = float(np.sqrt((diff**2).mean()))

        rec_pgs = _score_all(rec, pgs)
        r2 = {}
        rk = {}
        for pid in pgs:
            t, r = true_pgs[pid], rec_pgs[pid]
            if t.std() > 0 and r.std() > 0:
                r2[pid] = float(np.corrcoef(t, r)[0, 1] ** 2)
            else:
                r2[pid] = float("nan")
            rk[pid] = _spearman(t, r)
        out.append((m, conc, rmse, r2, rk, wall))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chroms", default="22")
    ap.add_argument("--father", default=None)
    ap.add_argument("--mother", default=None)
    ap.add_argument("--pop", default="CEU")
    ap.add_argument(
        "--ser",
        default="0,0.005,0.01,0.02",
        help="comma-separated switch-error rates per consecutive het",
    )
    ap.add_argument("--cov", default="0.005,0.02,0.05,0.1")
    ap.add_argument("--n-embryos", default="3,5,8")
    ap.add_argument(
        "--n-sibs",
        type=int,
        default=0,
        help="number of born siblings (30× WGS) appended to the joint HMM",
    )
    ap.add_argument(
        "--ado",
        type=float,
        default=0.0,
        help="allelic-dropout rate per het site (MDA ≈ 0.1–0.25; PTA ≈ 0.01–0.05)",
    )
    ap.add_argument(
        "--cov-dispersion",
        type=float,
        default=0.0,
        help="coverage CV from WGA amplification bias (MDA ≈ 0.5–1; 0 = Poisson)",
    )
    ap.add_argument(
        "--gp-phase",
        choices=["none", "father", "both"],
        default="none",
        help="anchor-correct the named parent(s)' phase with simulated "
        "grandparent genotypes (trio_phase + anchor_correct_phase) before "
        "recovery; joint methods then use a near-zero switch rate for the "
        "corrected parent",
    )
    ap.add_argument(
        "--gp-err",
        type=float,
        default=0.005,
        help="grandparent genotyping-error rate (exercises the "
        "Mendelian-conflict filter)",
    )
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seq-err", type=float, default=0.01)
    ap.add_argument(
        "--methods",
        default="oracle,naive,switch_aware,joint,rephase",
    )
    ap.add_argument(
        "--max-pgs",
        type=int,
        default=6,
        help="cap PGS files loaded (largest by SNP count) for speed",
    )
    args = ap.parse_args()

    sers = [float(x) for x in args.ser.split(",")]
    covs = [float(x) for x in args.cov.split(",")]
    nembs = [int(x) for x in args.n_embryos.split(",")]
    methods = args.methods.split(",")

    if not args.father or not args.mother:
        args.father, args.mother = E.pick_parents(args.pop)

    chroms = parse_chroms(args.chroms)
    print(
        f"# parents {args.father} × {args.mother}, chroms {chroms}, "
        f"{args.reps} reps, methods={methods}",
        file=sys.stderr,
    )

    # One Result per (method, ser, cov, n_emb), accumulated over chroms+reps.
    results: dict[tuple, Result] = {}

    for chrom in chroms:
        t0 = time.time()
        par = E.load_parents_cached(chrom, args.father, args.mother)
        het_mask = (par.pat[0] != par.pat[1]) | (par.mat[0] != par.mat[1])
        pgs_full = E.load_pgs_for_chrom(chrom, par)
        pgs = dict(
            sorted(pgs_full.items(), key=lambda kv: -len(kv[1][0]))[: args.max_pgs]
        )
        print(
            f"# chr{chrom}: {len(par.pos):,} sites, {het_mask.sum():,} informative, "
            f"{len(pgs)} PGS (load {time.time() - t0:.1f}s)",
            file=sys.stderr,
        )

        for ser in sers:
            for cov in covs:
                for ne in nembs:
                    for rep in range(args.reps):
                        for m, conc, rmse, r2, rk, wall in run_cell(
                            par, pgs, het_mask, ser, cov, ne, rep,
                            args.seq_err, methods, args.n_sibs,
                            args.ado, args.cov_dispersion,
                            args.gp_phase, args.gp_err,
                        ):
                            key = (m, ser, cov, ne)
                            if key not in results:
                                results[key] = Result(m, ser, cov, ne)
                            R = results[key]
                            R.conc_het.append(conc)
                            R.rmse_het.append(rmse)
                            R.wall_s.append(wall)
                            for pid, v in r2.items():
                                R.pgs_r2.setdefault(pid, []).append(v)
                            for pid, v in rk.items():
                                R.pgs_rank.setdefault(pid, []).append(v)
                    print(
                        f"#   ser={ser:<6} cov={cov:<6} E={ne:<2} done",
                        file=sys.stderr,
                    )

    _report(results, sers, covs, nembs, methods)


def _report(results, sers, covs, nembs, methods):
    def cell(R: Result | None, what: str) -> str:
        if R is None:
            return " " * 8
        if what == "conc":
            return f"{np.mean(R.conc_het):7.3%}"
        if what == "rmse":
            return f"{np.mean(R.rmse_het):7.4f}"
        if what == "pgs_r2":
            allv = [v for vs in R.pgs_r2.values() for v in vs if np.isfinite(v)]
            return f"{np.mean(allv):7.4f}" if allv else "   n/a "
        if what == "pgs_rk":
            allv = [v for vs in R.pgs_rank.values() for v in vs if np.isfinite(v)]
            return f"{np.mean(allv):7.4f}" if allv else "   n/a "
        if what == "wall":
            return f"{np.mean(R.wall_s):6.2f}s"
        return "?"

    print("\n" + "=" * 100)
    print("Genotype concordance at informative (parent-het) sites")
    print("=" * 100)
    for ne in nembs:
        print(f"\nn_embryos = {ne}")
        hdr = f"{'method':<14}{'SER':>6} | " + " ".join(f"{c:>9}×" for c in covs)
        print(hdr)
        print("-" * len(hdr))
        for m in methods:
            for ser in sers:
                row = " ".join(
                    cell(results.get((m, ser, c, ne)), "conc") + "  " for c in covs
                )
                print(f"{m:<14}{ser:>6} | {row}")
            if m != methods[-1]:
                print()

    print("\n" + "=" * 100)
    print("Dosage RMSE at informative sites (lower is better; "
          "soft dosage for `joint`)")
    print("=" * 100)
    for ne in nembs:
        print(f"\nn_embryos = {ne}")
        hdr = f"{'method':<14}{'SER':>6} | " + " ".join(f"{c:>9}×" for c in covs)
        print(hdr)
        print("-" * len(hdr))
        for m in methods:
            for ser in sers:
                row = " ".join(
                    cell(results.get((m, ser, c, ne)), "rmse") + "  " for c in covs
                )
                print(f"{m:<14}{ser:>6} | {row}")
            if m != methods[-1]:
                print()

    print("\n" + "=" * 100)
    print("PGS recovery: mean r²(true, imputed) across scores  "
          "(selection-relevant signal)")
    print("=" * 100)
    for ne in nembs:
        print(f"\nn_embryos = {ne}")
        hdr = f"{'method':<14}{'SER':>6} | " + " ".join(f"{c:>9}×" for c in covs)
        print(hdr)
        print("-" * len(hdr))
        for m in methods:
            for ser in sers:
                row = " ".join(
                    cell(results.get((m, ser, c, ne)), "pgs_r2") + "  " for c in covs
                )
                print(f"{m:<14}{ser:>6} | {row}")
            if m != methods[-1]:
                print()

    print("\n" + "=" * 100)
    print("PGS recovery: mean Spearman rank correlation across scores  "
          "(would you pick the same embryo?)")
    print("=" * 100)
    for ne in nembs:
        print(f"\nn_embryos = {ne}")
        hdr = f"{'method':<14}{'SER':>6} | " + " ".join(f"{c:>9}×" for c in covs)
        print(hdr)
        print("-" * len(hdr))
        for m in methods:
            for ser in sers:
                row = " ".join(
                    cell(results.get((m, ser, c, ne)), "pgs_rk") + "  " for c in covs
                )
                print(f"{m:<14}{ser:>6} | {row}")
            if m != methods[-1]:
                print()

    print("\n" + "=" * 100)
    print("Wall time per cell (mean over reps)")
    print("=" * 100)
    ne = nembs[-1]
    print(f"\nn_embryos = {ne}")
    hdr = f"{'method':<14}{'SER':>6} | " + " ".join(f"{c:>9}×" for c in covs)
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        ser = sers[-1]
        row = " ".join(cell(results.get((m, ser, c, ne)), "wall") + "  " for c in covs)
        print(f"{m:<14}{ser:>6} | {row}")


if __name__ == "__main__":
    main()
