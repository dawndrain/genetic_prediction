"""Benchmark embryo-genome imputation under realistic parental phasing error.

The original demo (embryo_selection_demo.py) assumes the parents'
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
                (embryo.joint_recover).

For each (SER, coverage, n_embryos) cell it reports:
  - genotype concordance at informative (parent-het) sites
  - per-site dosage RMSE at informative sites
  - PGS recovery: r² between true and imputed per-embryo scores,
    and Spearman rank correlation (selection fidelity)

Run from the repo root:
    python examples/embryo_phasing_bench.py --chroms 22 --reps 3
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import embryo as E  # noqa: E402,I001
from genepred.io import parse_chroms  # noqa: E402


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
):
    rng_sw = np.random.default_rng((101, rep, int(ser * 1e6)))
    par_obs, _, _ = E.apply_switch_errors(par_true, ser, rng_sw)

    truth = np.empty((n_emb, len(par_true.pos)), dtype=np.int8)
    biop: list[tuple[np.ndarray, np.ndarray]] = []
    for e in range(n_emb):
        g, _, _ = E.simulate_child(par_true, np.random.default_rng((202, rep, e)))
        truth[e] = g
        biop.append(
            E.simulate_biopsy(
                g, cov, seq_err, np.random.default_rng((303, rep, e, int(cov * 1e6)))
            )
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
            _, rec, _, _ = E.joint_recover(
                par_obs, biop, seq_err, switch_rate=max(ser, 1e-4), n_iter=2
            )
        elif m == "rephase":
            rec, _, _, _ = E.joint_rephase_recover(
                par_obs, biop, seq_err, switch_rate=max(ser, 1e-4)
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
                            args.seq_err, methods,
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
