"""Iterated embryo selection under a mis-specified PGS.

What happens if you select the top-PGS embryo each generation, when
the true genotype→phenotype map has structure the PGS can't see?

Two failure modes are modelled:

  1. Saturation. The true phenotype is y = f(g) + ε with f a
     logistic, but the PGS is the linear g (the local tangent at
     the founding mean). The score keeps rising; the trait
     plateaus.

  2. Recessive load. Each founder carries ~k heterozygous
     deleterious recessives drawn from a realistic DFE. Inbreeding
     (small founding population → rising F) exposes them as
     homozygotes; fitness drops. The additive PGS, fit on outbred
     data, is blind to this.

The two interact: with a tiny founding population, inbreeding
depression dominates long before saturation matters. With a large
population (the livestock case), saturation is the eventual limit.

Output: a TSV of per-generation (mean PGS, mean realised trait,
mean fitness, mean F, var(g)) for each scenario, plus a text
summary.

    python examples/iterated_selection.py \\
        --n-founders 2,10,50,500 --n-gens 20 --n-embryos 5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np


@dataclass
class Architecture:
    """Genetic architecture of the simulated trait + load."""

    n_trait_loci: int = 2000
    h2: float = 0.5
    sat_scale: float = 4.0  # logistic half-saturation, in trait-SD units
    n_load_loci: int = 5000
    load_per_founder: float = 80.0  # expected het deleterious carried
    load_dfe_shape: float = 0.2  # Gamma shape for s_rec
    load_dfe_mean: float = 0.03  # mean s_rec among deleterious
    p_lethal: float = 0.02  # fraction of load loci with s_rec ≈ 1


def make_arch(arch: Architecture, rng: np.random.Generator):
    """Trait-locus effect sizes + load-locus s_rec, AF, and founders."""
    beta = rng.normal(0, 1, arch.n_trait_loci)
    beta *= np.sqrt(arch.h2 / (2 * 0.2 * 0.8 * arch.n_trait_loci)) / beta.std()
    af_trait = rng.beta(1.5, 4.5, arch.n_trait_loci).clip(0.05, 0.5)
    s_rec = rng.gamma(
        arch.load_dfe_shape, arch.load_dfe_mean / arch.load_dfe_shape, arch.n_load_loci
    ).clip(0, 1)
    lethal = rng.random(arch.n_load_loci) < arch.p_lethal
    s_rec[lethal] = 1.0
    af_load = arch.load_per_founder / (2 * arch.n_load_loci)
    return beta, af_trait, s_rec, af_load


def make_founders(n: int, arch: Architecture, seed: int):
    rng = np.random.default_rng(seed)
    beta, af_t, s_rec, af_l = make_arch(arch, rng)
    L = arch.n_trait_loci + arch.n_load_loci
    af = np.concatenate([af_t, np.full(arch.n_load_loci, af_l)])
    haps = (rng.random((n, 2, L)) < af).astype(np.int8)
    # Population-reference SD from a large independent draw, so σ is
    # comparable across runs regardless of n_founders.
    ref = (rng.random((2000, 2, arch.n_trait_loci)) < af_t).astype(np.int8).sum(1)
    g_ref = ref @ beta
    return haps, beta, s_rec, af, float(g_ref.mean()), float(g_ref.std())


def optimal_haploid(parent: np.ndarray, beta: np.ndarray, n_t: int) -> np.ndarray:
    """The per-locus-optimal mosaic of this parent's two haplotypes:
    at every het locus, take whichever allele has the higher β·allele.
    This is the ceiling iterated meiosis converges toward — reachable
    only if crossover can decouple every locus (i.e. in the
    independent-assortment limit, not under realistic ~1
    crossover/chromosome)."""
    L = parent.shape[1]
    out = parent[0].copy()
    het = parent[0, :n_t] != parent[1, :n_t]
    better1 = (parent[1, :n_t] - parent[0, :n_t]) * beta > 0
    swap = np.zeros(L, dtype=bool)
    swap[:n_t] = het & better1
    out[swap] = parent[1, swap]
    return out


def gametes(
    parent_haps: np.ndarray,
    n: int,
    rng: np.random.Generator,
    n_chrom: int = 23,
    co_per_chrom: float = 1.5,
):
    """n recombinant gametes from one parent (n, L).

    Realistic mode (default): loci are split evenly across `n_chrom`
    chromosomes; each chromosome independently assorts and gets
    Poisson(`co_per_chrom`) crossovers at uniform positions, so
    nearby loci stay linked. With co_per_chrom=inf you recover the
    old per-locus independent-assortment limit (every locus
    decoupled), which is the ceiling case but not biology."""
    L = parent_haps.shape[1]
    if not np.isfinite(co_per_chrom):
        pick = rng.integers(0, 2, size=(n, L))
    else:
        bnds = np.linspace(0, L, n_chrom + 1, dtype=int)
        pick = np.empty((n, L), dtype=np.int8)
        for c in range(n_chrom):
            lo, hi = bnds[c], bnds[c + 1]
            for j in range(n):
                k = rng.poisson(co_per_chrom)
                cuts = np.sort(rng.integers(lo, hi, size=k)) if k else np.empty(0, int)
                phase = rng.integers(0, 2)
                prev = lo
                for cut in cuts:
                    pick[j, prev:cut] = phase
                    phase ^= 1
                    prev = cut
                pick[j, prev:hi] = phase
    return np.take_along_axis(
        parent_haps[None].repeat(n, 0), pick[:, None, :], 1
    )[:, 0, :]


def iterated_meiosis(
    parent: np.ndarray,
    beta: np.ndarray,
    n_t: int,
    n_rounds: int,
    n_gametes: int,
    rng: np.random.Generator,
    scheme: str = "top2",
    co_per_chrom: float = 1.5,
    pool_size: int = 4,
    sel_intensity: float = 1.5,
) -> tuple[np.ndarray, list[float]]:
    """Within-parent iterated meiosis. Schemes:

      top2     fuse the two best gametes (greedy; loses diversity fast)
      top+rand fuse the best with a uniformly random other
      top+far  fuse the best with the gamete most Hamming-distant
               from it (explicit diversity preservation)
      pool     keep `pool_size` diploids; each round, gametes from
               all of them, fuse random top-2k into k new diploids
      useful   pool, but choose which gamete pairs to fuse by the
               usefulness criterion μ + i·σ — pair mean plus the
               next-round selection differential its heterozygosity
               supports (Schnell 1983)
      ohv      pool, but rank candidate diploids by their *optimal
               haploid value* — the best gamete each could in
               principle produce (Daetwyler 2015). One-step look-ahead.

    Returns the best haploid found and the per-round best-score
    track. No inbreeding penalty: the intermediate diploids are cell
    cultures, not organisms."""
    track: list[float] = []
    beta2 = beta**2

    def pair_useful(g1: np.ndarray, g2: np.ndarray) -> float:
        mu = float((g1[:n_t] + g2[:n_t]) @ beta) / 2
        het = g1[:n_t] != g2[:n_t]
        sigma = float(np.sqrt((beta2[het]).sum() / 4))
        return mu + sel_intensity * sigma

    def pair_ohv(g1: np.ndarray, g2: np.ndarray) -> float:
        dip = np.stack([g1, g2])
        return float(optimal_haploid(dip, beta, n_t)[:n_t] @ beta)

    if scheme in ("pool", "useful", "ohv"):
        pool = np.repeat(parent[None], pool_size, 0)
        best_hap = parent[0]
        best_s = -np.inf
        for _ in range(n_rounds):
            gam = np.concatenate(
                [gametes(d, n_gametes, rng, co_per_chrom=co_per_chrom) for d in pool]
            )
            score = gam[:, :n_t] @ beta
            order = np.argsort(-score)
            track.append(float(score[order[0]]))
            if score[order[0]] > best_s:
                best_s, best_hap = float(score[order[0]]), gam[order[0]].copy()
            if scheme == "pool":
                sel = gam[order[: 2 * pool_size]]
                rng.shuffle(sel)
                pool = sel.reshape(pool_size, 2, -1)
            else:
                # Greedy pairing among the top-K gametes by the chosen
                # criterion: repeatedly pick the unmatched pair with the
                # highest μ+iσ (or OHV), remove both, until pool is full.
                K = min(4 * pool_size, len(gam))
                cand = list(order[:K])
                crit = pair_useful if scheme == "useful" else pair_ohv
                new_pool = []
                while len(new_pool) < pool_size and len(cand) >= 2:
                    best_pair, best_v = None, -np.inf
                    for ii in range(len(cand)):
                        for jj in range(ii + 1, len(cand)):
                            v = crit(gam[cand[ii]], gam[cand[jj]])
                            if v > best_v:
                                best_v, best_pair = v, (ii, jj)
                    ii, jj = best_pair
                    new_pool.append(np.stack([gam[cand[ii]], gam[cand[jj]]]))
                    for k in sorted((ii, jj), reverse=True):
                        cand.pop(k)
                pool = np.stack(new_pool)
        return best_hap, track

    dip = parent.copy()
    best_hap = parent[0]
    best_s = -np.inf
    for _ in range(n_rounds):
        gam = gametes(dip, n_gametes, rng, co_per_chrom=co_per_chrom)
        score = gam[:, :n_t] @ beta
        order = np.argsort(-score)
        track.append(float(score[order[0]]))
        if score[order[0]] > best_s:
            best_s, best_hap = float(score[order[0]]), gam[order[0]].copy()
        if scheme == "top2":
            dip = gam[order[:2]]
        elif scheme == "top+rand":
            other = order[1 + rng.integers(0, max(len(order) - 1, 1))]
            dip = gam[[order[0], other]]
        elif scheme == "top+far":
            dist = (gam != gam[order[0]]).sum(1)
            dip = gam[[order[0], int(np.argmax(dist))]]
        else:
            raise ValueError(scheme)
    return best_hap, track


def run_iterated_meiosis(
    n_couples: int,
    n_rounds: int,
    n_gametes: int,
    arch: Architecture,
    seed: int,
    scheme: str,
    co_per_chrom: float,
):
    """One generation of embryos from per-parent iterated meiosis.

    Returns (conv_track, embryo_PGS, embryo_y, fitness, F,
    ceiling_PGS, frac_of_ceiling)."""
    rng = np.random.default_rng(seed)
    haps, beta, s_rec, _, g_mean0, g_sd0 = make_founders(2 * n_couples, arch, seed)
    n_t = arch.n_trait_loci
    H0 = (haps.sum(1) == 1).mean()

    embryos = np.empty((n_couples, 2, haps.shape[2]), dtype=np.int8)
    ceiling = np.empty((n_couples, 2, haps.shape[2]), dtype=np.int8)
    conv = np.zeros((n_couples, 2, n_rounds))
    for c in range(n_couples):
        for p in (0, 1):
            hap, tr = iterated_meiosis(
                haps[2 * c + p], beta, n_t, n_rounds, n_gametes, rng,
                scheme=scheme, co_per_chrom=co_per_chrom,
            )
            embryos[c, p] = hap
            ceiling[c, p] = optimal_haploid(haps[2 * c + p], beta, n_t)
            conv[c, p] = tr
    g_std, y, w, het = phenotype(embryos, beta, s_rec, arch, rng, g_mean0, g_sd0)
    g_ceil, *_ = phenotype(
        ceiling, beta, s_rec, arch, np.random.default_rng(seed), g_mean0, g_sd0
    )
    F = 1 - het.mean() / max(H0, 1e-9)
    conv_std = (conv.mean(axis=(0, 1)) - g_mean0 / 2) / (g_sd0 / np.sqrt(2))
    frac = float(g_std.mean() / max(g_ceil.mean(), 1e-9))
    return conv_std, g_std.mean(), y.mean(), w.mean(), F, g_ceil.mean(), frac


def phenotype(
    haps: np.ndarray, beta: np.ndarray, s_rec: np.ndarray, arch: Architecture,
    rng: np.random.Generator, g_mean0: float, g_sd0: float,
):
    """Returns (g_std, y, fitness, F) per individual."""
    geno = haps.sum(1)  # (N, L)
    n_t = arch.n_trait_loci
    g = geno[:, :n_t] @ beta
    g_std = (g - g_mean0) / g_sd0
    y_lat = g_std + rng.normal(0, np.sqrt(1 - arch.h2), len(g))
    y = arch.sat_scale * np.tanh(y_lat / arch.sat_scale)
    hom_load = (geno[:, n_t:] == 2).astype(float)
    log_w = -(hom_load * s_rec).sum(1)
    fitness = np.exp(log_w)
    F = (geno == 1).mean(1)  # crude heterozygosity proxy → 1 - H/H0
    return g_std, y, fitness, F


def run(
    n_founders: int,
    n_gens: int,
    n_embryos: int,
    arch: Architecture,
    seed: int,
    retrain: bool,
    fitness_filter: bool,
    co_per_chrom: float = 1.5,
):
    rng = np.random.default_rng(seed)
    haps, beta, s_rec, af, g_mean0, g_sd0 = make_founders(n_founders, arch, seed)
    n_t = arch.n_trait_loci
    geno0 = haps.sum(1)
    H0 = (geno0 == 1).mean()

    rows = []
    for gen in range(n_gens + 1):
        g_std, y, w, het = phenotype(haps, beta, s_rec, arch, rng, g_mean0, g_sd0)
        F = 1 - het.mean() / max(H0, 1e-9)
        rows.append(
            (gen, g_std.mean(), y.mean(), w.mean(), F, g_std.var(), len(haps))
        )
        if gen == n_gens:
            break
        N = len(haps)
        if fitness_filter:
            keep = rng.random(N) < (w / max(w.max(), 1e-9))
            if keep.sum() < 2:
                keep[:] = True
            haps = haps[keep]
            N = len(haps)
        # Random mating with replacement to keep N constant; each
        # mating produces n_embryos and the top-PGS one is kept.
        next_haps = []
        idx_a = rng.integers(0, N, size=N)
        idx_b = rng.integers(0, N, size=N)
        idx_b = np.where(idx_b == idx_a, (idx_b + 1) % max(N, 2), idx_b)
        for a, b in zip(idx_a, idx_b, strict=True):
            ga = gametes(haps[a], n_embryos, rng, co_per_chrom=co_per_chrom)
            gb = gametes(haps[b], n_embryos, rng, co_per_chrom=co_per_chrom)
            emb = np.stack([ga, gb], 1)
            pgs = (emb.sum(1)[:, :n_t] @ beta - g_mean0) / g_sd0
            if retrain:
                pass  # retraining would re-fit beta on current pop; TODO
            next_haps.append(emb[int(np.argmax(pgs))])
        if not next_haps:
            break
        haps = np.stack(next_haps)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-founders", default="2,10,50,500")
    ap.add_argument("--n-gens", type=int, default=20)
    ap.add_argument("--n-embryos", type=int, default=5)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sat-scale", type=float, default=4.0)
    ap.add_argument(
        "--fitness-filter",
        action="store_true",
        help="cull each generation proportional to fitness "
        "(phenotypic selection's lethal-combination filter)",
    )
    ap.add_argument(
        "--no-load",
        action="store_true",
        help="disable recessive load (saturation only)",
    )
    ap.add_argument(
        "--meiosis",
        action="store_true",
        help="iterated-meiosis mode: optimise each parent's haploid "
        "separately for --n-gens rounds, then form one embryo per "
        "couple. No multi-generation mating, so no inbreeding.",
    )
    ap.add_argument(
        "--scheme",
        choices=["top2", "top+rand", "top+far", "pool", "useful", "ohv", "all"],
        default="all",
        help="iterated-meiosis fusion scheme (see iterated_meiosis docstring)",
    )
    ap.add_argument(
        "--co-per-chrom",
        type=float,
        default=1.5,
        help="mean crossovers per chromosome per meiosis; use 'inf' "
        "for per-locus independent assortment (the unrealistic ceiling)",
    )
    args = ap.parse_args()
    args.co_per_chrom = float(args.co_per_chrom)

    arch = Architecture(sat_scale=args.sat_scale)
    if args.no_load:
        arch.n_load_loci = 0

    if args.meiosis:
        schemes = (
            ["top2", "pool", "useful", "ohv"]
            if args.scheme == "all"
            else [args.scheme]
        )
        print(
            f"# iterated MEIOSIS: {args.n_embryos} gametes/round, "
            f"{args.n_gens} rounds, co/chrom={args.co_per_chrom}, "
            f"sat@{arch.sat_scale}σ  (σ = population reference SD)"
        )
        for nc in (int(x) for x in args.n_founders.split(",")):
            for scheme in schemes:
                agg = []
                for r in range(args.reps):
                    agg.append(
                        run_iterated_meiosis(
                            nc, args.n_gens, args.n_embryos, arch,
                            args.seed + r, scheme, args.co_per_chrom,
                        )
                    )
                conv = np.mean([a[0] for a in agg], 0)
                g, y, w, F, ceil, frac = np.mean([a[1:] for a in agg], 0)
                print(
                    f"# n_couples={nc} scheme={scheme:<8}: "
                    f"embryo PGS={g:+.2f}σ realised={y:+.2f}σ "
                    f"fitness={w:.3f} F={F:+.3f}  | "
                    f"ceiling={ceil:+.2f}σ ({frac:.0%} reached)",
                    file=sys.stderr,
                )
                for rd in range(args.n_gens):
                    print(f"{nc}\t{scheme}\t{rd}\t{conv[rd]:+.3f}")
        return

    print(
        f"# iterated selection: {args.n_embryos} embryos/couple, "
        f"{args.n_gens} gens, sat@{arch.sat_scale}σ, "
        f"load={arch.load_per_founder if arch.n_load_loci else 0} LoF/founder "
        f"(mean s_rec={arch.load_dfe_mean}), "
        f"fitness-filter={'on' if args.fitness_filter else 'off'}"
    )
    print(
        "n_founders\tgen\tmean_PGS\tmean_y\tmean_fitness\tF\tvar_g\tN"
    )
    for nf in (int(x) for x in args.n_founders.split(",")):
        agg: dict[int, list] = {}
        for r in range(args.reps):
            for row in run(
                nf, args.n_gens, args.n_embryos, arch, args.seed + r,
                retrain=False, fitness_filter=args.fitness_filter,
            ):
                agg.setdefault(row[0], []).append(row[1:])
        for gen in sorted(agg):
            a = np.array(agg[gen])
            m = a.mean(0)
            print(
                f"{nf}\t{gen}\t{m[0]:+.3f}\t{m[1]:+.3f}\t{m[2]:.3f}\t"
                f"{m[3]:+.3f}\t{m[4]:.3f}\t{m[5]:.0f}"
            )
        # one-line summary
        last = np.array(agg[max(agg)]).mean(0)
        first = np.array(agg[0]).mean(0)
        print(
            f"# n_founders={nf}: PGS Δ={last[0] - first[0]:+.2f}σ, "
            f"realised Δ={last[1] - first[1]:+.2f}σ, "
            f"fitness {first[2]:.2f}→{last[2]:.2f}, F→{last[3]:+.2f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
