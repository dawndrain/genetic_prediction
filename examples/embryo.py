"""Embryo simulation: meiosis, PGT-A biopsy, and HMM haplotype recovery.

This is the technical core of what commercial embryo-PGS providers do:
parents are deeply genotyped, the embryo is barely sequenced, and an
HMM bridges the gap by exploiting that recombination is rare.

  1. Two phased parental genomes from 1KG (CEU trio parents NA12891 +
     NA12892 by default — NA12878 is their real daughter, useful as
     ground truth).
  2. N children via meiosis: per parent, sample crossover points
     (Poisson on the genetic length); each gamete is a mosaic of the
     two parental haplotypes; child = paternal ⊕ maternal gamete.
  3. PGT-A trophectoderm biopsy: ~0.05× coverage after WGA, so most
     SNPs see 0 reads; each read samples one allele from the diploid
     genotype with sequencing error ε.
  4. Haplotype inheritance via 4-state Viterbi (which paternal × which
     maternal haplotype) over informative sites; the child genotype is
     read off the path + parental haplotypes.

Functions here are library-grade; the genome-wide demo orchestration
lives in `genepred.cli:embryo_demo`.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from genepred.paths import data_dir, kg_dir, pgs_weights_dir

# Approximate genetic length per physical Mb (cM/Mb), per autosome.
# chr22 ≈ 70 cM → expected crossovers per meiosis ≈ 0.7.
CM_PER_MB = {
    str(c): v
    for c, v in zip(
        range(1, 23),
        [
            2.0,
            1.9,
            1.9,
            1.8,
            1.8,
            1.8,
            1.9,
            1.7,
            1.8,
            2.0,
            2.0,
            2.1,
            2.0,
            2.1,
            2.4,
            2.4,
            2.5,
            2.6,
            2.8,
            2.9,
            3.4,
            3.6,
        ],
    )
}


@dataclass
class Parents:
    chrom: str
    pos: np.ndarray  # (M,) GRCh37 positions
    ref: np.ndarray  # (M,) '<U1' REF allele
    alt: np.ndarray  # (M,) '<U1' ALT allele
    pat: np.ndarray  # (2, M) int8 — father's two haplotypes (0=REF, 1=ALT)
    mat: np.ndarray  # (2, M) int8 — mother's two haplotypes


def pick_parents(pop: str = "CEU") -> tuple[str, str]:
    """First male/female sample IDs from the given 1KG population."""
    panel = kg_dir() / "integrated_call_samples_v3.20130502.ALL.panel"
    males, females = [], []
    with open(panel) as f:
        f.readline()
        for line in f:
            r = line.rstrip().split("\t")
            if r[1] == pop:
                (males if r[3] == "male" else females).append(r[0])
    return males[0], females[0]


def load_parents(chrom: str, father: str, mother: str) -> Parents:
    """Load phased haplotypes for two samples from the 1KG VCF."""
    vcf = next(kg_dir().glob(f"ALL.chr{chrom}.*.vcf.gz"))
    pos, ref, alt, pat, mat = [], [], [], [], []
    fc = mc = -1
    with gzip.open(vcf, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                hdr = line.rstrip().split("\t")
                fc, mc = hdr.index(father), hdr.index(mother)
                continue
            r = line.rstrip().split("\t")
            if len(r[3]) != 1 or len(r[4]) != 1 or "," in r[4]:
                continue
            fg, mg = r[fc], r[mc]
            if len(fg) < 3 or len(mg) < 3 or fg[1] != "|" or mg[1] != "|":
                continue
            pos.append(int(r[1]))
            ref.append(r[3])
            alt.append(r[4])
            pat.append((int(fg[0]), int(fg[2])))
            mat.append((int(mg[0]), int(mg[2])))
    return Parents(
        chrom=chrom,
        pos=np.array(pos, dtype=np.int64),
        ref=np.array(ref, dtype="<U1"),
        alt=np.array(alt, dtype="<U1"),
        pat=np.array(pat, dtype=np.int8).T,
        mat=np.array(mat, dtype=np.int8).T,
    )


def load_parents_cached(chrom: str, father: str, mother: str) -> Parents:
    """load_parents() with an npz cache — the gzip VCF parse is ~2 min/chr."""
    cache = data_dir() / "embryo_cache"
    cache.mkdir(exist_ok=True)
    f = cache / f"chr{chrom}.{father}.{mother}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=False)
        return Parents(
            chrom, d["pos"], d["ref"], d["alt"], d["pat"], d["mat"]
        )
    par = load_parents(chrom, father, mother)
    np.savez_compressed(
        f, pos=par.pos, ref=par.ref, alt=par.alt, pat=par.pat, mat=par.mat
    )
    return par


# ----------------------------------------------------- phasing-error model
#
# Real parental genomes are phased statistically (SHAPEIT/Eagle/Beagle),
# not perfectly. The dominant artefact is a *switch error*: at some
# heterozygous site the assignment of alleles to hap0/hap1 flips and
# stays flipped until the next switch. Modern phasers on EUR samples
# with a large reference panel achieve a switch-error rate (SER) of
# roughly 0.3–1 % per consecutive het pair; without a panel, 2–5 %.
#
# A father with ~28 k hets on chr22 at SER = 1 % therefore has ~280
# switch points — two orders of magnitude more than the ~1 meiotic
# crossover per chromosome. The 4-state HMM, whose transition prior is
# the crossover rate, will refuse to follow that many flips and will
# mis-track the inheritance path over long stretches.


def _inject_switches(hap2: np.ndarray, ser: float, rng: np.random.Generator):
    """Return (hap2_obs, swap_track) where swap_track[i] ∈ {0,1} says
    whether hap0/hap1 are swapped at site i relative to the truth.
    Switches occur i.i.d. between consecutive heterozygous sites."""
    M = hap2.shape[1]
    het = np.flatnonzero(hap2[0] != hap2[1])
    swap = np.zeros(M, dtype=np.int8)
    if len(het) == 0 or ser <= 0:
        return hap2.copy(), swap
    flips = rng.random(len(het) - 1) < ser
    state = np.concatenate(([0], np.cumsum(flips) & 1)).astype(np.int8)
    # piecewise-constant extension: swap is undefined at hom sites (both
    # haps equal there), so any value is fine; carry the het-site state.
    swap[het[0] :] = np.repeat(
        state, np.diff(np.append(het, M))
    )
    obs = hap2.copy()
    obs[:, swap == 1] = obs[::-1, swap == 1]
    return obs, swap


def apply_switch_errors(
    par: Parents, ser: float, rng: np.random.Generator
) -> tuple[Parents, np.ndarray, np.ndarray]:
    """Corrupt both parents' phase. Returns (par_obs, pat_swap, mat_swap)."""
    pat_obs, pat_swap = _inject_switches(par.pat, ser, rng)
    mat_obs, mat_swap = _inject_switches(par.mat, ser, rng)
    return replace(par, pat=pat_obs, mat=mat_obs), pat_swap, mat_swap


# ------------------------------------------------------------------ meiosis


def simulate_gamete(
    hap2: np.ndarray, pos: np.ndarray, chrom: str, rng: np.random.Generator
):
    """One meiotic product from a (2, M) haplotype pair.
    Returns (gamete (M,) int8, path (M,) int8 of which hap was used)."""
    M = hap2.shape[1]
    span_mb = (pos[-1] - pos[0]) / 1e6
    n_co = rng.poisson(span_mb * CM_PER_MB.get(chrom, 2.0) / 100)
    co_pos = np.sort(rng.integers(pos[0], pos[-1], size=n_co))
    path = np.zeros(M, dtype=np.int8)
    cur = rng.integers(0, 2)
    last = 0
    for cp in co_pos:
        idx = np.searchsorted(pos, cp)
        path[last:idx] = cur
        cur = 1 - cur
        last = idx
    path[last:] = cur
    return hap2[path, np.arange(M)], path


def simulate_child(par: Parents, rng: np.random.Generator):
    """Returns (geno (M,) int8 ALT-dosage 0/1/2, ppath, mpath)."""
    pg, ppath = simulate_gamete(par.pat, par.pos, par.chrom, rng)
    mg, mpath = simulate_gamete(par.mat, par.pos, par.chrom, rng)
    return pg + mg, ppath, mpath


def simulate_biopsy(
    geno: np.ndarray, coverage: float, err: float, rng: np.random.Generator
):
    """Per-SNP (n_ref, n_alt) read counts at the given mean coverage."""
    n_reads = rng.poisson(coverage, size=len(geno))
    p_alt = geno / 2 * (1 - err) + (1 - geno / 2) * err
    n_alt = rng.binomial(n_reads, p_alt)
    return n_reads - n_alt, n_alt


# ----------------------------------------------------------- HMM recovery


_STATES = np.array([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=np.int8)


def _build_log_T(pos: np.ndarray, recomb_per_bp: float):
    """(M-1, 4, 4) log-transition matrix; depends only on positions."""
    dpos = np.diff(pos).astype(np.float64)
    r = np.clip(recomb_per_bp * dpos, 1e-12, 0.49)
    lr, l1r = np.log(r), np.log1p(-r)
    log_T = np.empty((len(dpos), 4, 4))
    for a, (pa, ma) in enumerate(_STATES):
        for b, (pb, mb) in enumerate(_STATES):
            dp = int(pa != pb)
            dm = int(ma != mb)
            log_T[:, a, b] = dp * lr + (1 - dp) * l1r + dm * lr + (1 - dm) * l1r
    return log_T


@dataclass
class HMMContext:
    """Per-chromosome state shared across embryos: which sites are
    informative for haplotype inference, and the transition matrix
    over those sites."""

    inf_idx: np.ndarray
    log_T: np.ndarray
    pat_inf: np.ndarray
    mat_inf: np.ndarray


def build_hmm_context(par: Parents, recomb_per_bp: float = 1e-8) -> HMMContext:
    """Sites where at least one parent is heterozygous are the only ones
    that distinguish between the four inheritance states."""
    informative = (par.pat[0] != par.pat[1]) | (par.mat[0] != par.mat[1])
    idx = np.flatnonzero(informative)
    return HMMContext(
        inf_idx=idx,
        log_T=_build_log_T(par.pos[idx], recomb_per_bp),
        pat_inf=par.pat[:, idx],
        mat_inf=par.mat[:, idx],
    )


def hmm_recover(
    par: Parents, ctx: HMMContext, n_ref: np.ndarray, n_alt: np.ndarray, err: float
):
    """4-state Viterbi over informative sites; expand the path to all
    sites (path is piecewise-constant between informative sites, and
    non-informative sites have a deterministic child genotype anyway).
    Returns (geno (M,), ppath (M,), mpath (M,))."""
    M = len(par.pos)
    K = len(ctx.inf_idx)
    if K == 0:
        return par.pat[0] + par.mat[0], np.zeros(M, np.int8), np.zeros(M, np.int8)

    nr, na = n_ref[ctx.inf_idx], n_alt[ctx.inf_idx]
    n_tot = nr + na
    log_emit = np.zeros((K, 4))
    for s, (pi, mi) in enumerate(_STATES):
        d = ctx.pat_inf[pi] + ctx.mat_inf[mi]
        p = d / 2 * (1 - err) + (1 - d / 2) * err
        with np.errstate(divide="ignore", invalid="ignore"):
            log_emit[:, s] = na * np.log(p) + nr * np.log1p(-p)
    log_emit = np.nan_to_num(log_emit, nan=0.0, neginf=-1e9)
    log_emit[n_tot == 0] = 0.0

    V = np.log(0.25) + log_emit[0]
    bp = np.empty((K, 4), dtype=np.int8)
    for t in range(1, K):
        cand = V[:, None] + ctx.log_T[t - 1]
        bp[t] = np.argmax(cand, axis=0)
        V = cand[bp[t], np.arange(4)] + log_emit[t]

    path_inf = np.empty(K, dtype=np.int8)
    path_inf[-1] = int(np.argmax(V))
    for t in range(K - 2, -1, -1):
        path_inf[t] = bp[t + 1, path_inf[t + 1]]

    full = np.empty(M, dtype=np.int8)
    full[: ctx.inf_idx[0]] = path_inf[0]
    for k in range(K - 1):
        full[ctx.inf_idx[k] : ctx.inf_idx[k + 1]] = path_inf[k]
    full[ctx.inf_idx[-1] :] = path_inf[-1]

    ppath = _STATES[full, 0]
    mpath = _STATES[full, 1]
    geno = par.pat[ppath, np.arange(M)] + par.mat[mpath, np.arange(M)]
    return geno, ppath, mpath


# ------------------------------------------- joint, phase-aware recovery
#
# The per-embryo HMM tracks w_e = "which of the parent's *observed*
# haplotypes embryo e carries". With switch errors in the parental
# phase, w_e flips at every switch *and* every recombination — far
# more transitions than the recombination prior allows. Two fixes:
#
#   switch-aware single: same 4-state HMM, but inflate the transition
#       rate to recomb + switch. Tracks the path but, with the prior
#       loosened ~100×, is much more vulnerable to read noise — bad at
#       very low coverage.
#
#   joint: pool all embryos. A parental switch flips *every* embryo's
#       w simultaneously; a recombination flips only one. A 2^E-state
#       HMM over (w_1,…,w_E) with a transition kernel that mixes
#       "shared switch" (rate p_s) and "independent recomb" (rate p_r)
#       can therefore call switches with E× the read evidence while
#       keeping each embryo's recombination prior tight.
#
# Paternal and maternal inference are decoupled by restricting each to
# sites where the *other* parent is homozygous, so that parent's
# contribution to the embryo dosage is a known constant. (<0.1 % of
# sites have both parents het; those sites' reads are unused for the
# joint path but their genotypes are still reconstructed correctly
# from the piecewise-constant path extension.)


def hmm_recover_switch_aware(
    par: Parents,
    n_ref: np.ndarray,
    n_alt: np.ndarray,
    err: float,
    recomb_per_bp: float = 1e-8,
    switch_rate: float = 0.01,
):
    """Per-embryo 4-state HMM with the transition prior inflated to
    cover phasing switches. switch_rate is per consecutive het of the
    relevant parent, converted to per-bp via mean het spacing."""
    het_pat = (par.pat[0] != par.pat[1]).mean()
    het_mat = (par.mat[0] != par.mat[1]).mean()
    span = par.pos[-1] - par.pos[0]
    M = len(par.pos)
    # switches/bp ≈ ser × hets/bp; take the max over parents so the
    # prior is loose enough for either.
    sw_per_bp = switch_rate * max(het_pat, het_mat) * M / max(span, 1)
    ctx = build_hmm_context(par, recomb_per_bp + sw_per_bp)
    return hmm_recover(par, ctx, n_ref, n_alt, err)


def _extend_path(idx: np.ndarray, vals: np.ndarray, M: int) -> np.ndarray:
    """Piecewise-constant extension of vals (defined at idx) to [0, M)."""
    out = np.empty(M, dtype=vals.dtype)
    if len(idx) == 0:
        out.fill(0)
        return out
    out[: idx[0]] = vals[0]
    out[idx[-1] :] = vals[-1]
    if len(idx) > 1:
        out[idx[0] : idx[-1]] = np.repeat(vals[:-1], np.diff(idx))
    return out


def _per_embryo_emission(
    this_hap: np.ndarray,
    other_hap: np.ndarray,
    idx: np.ndarray,
    biopsies: list[tuple[np.ndarray, np.ndarray]],
    err: float,
    other_w: np.ndarray | None,
):
    """log P(reads_e at idx | w_e = h) for h ∈ {0,1}, shape (E, K, 2).

    At sites where the other parent is heterozygous, the dose depends
    on the other parent's path too. If `other_w` (E, M) is given, use
    it; otherwise marginalize 50/50 over the other parent's hap."""
    E = len(biopsies)
    K = len(idx)
    other_het = other_hap[0, idx] != other_hap[1, idx]
    le = np.zeros((E, K, 2))
    lp = np.empty(3)
    for d in range(3):
        p = d / 2 * (1 - err) + (1 - d / 2) * err
        lp[d] = np.log(p)
    l1p = np.log1p(-np.exp(lp))
    for e, (nr, na) in enumerate(biopsies):
        nr_i = nr[idx].astype(np.float64)
        na_i = na[idx].astype(np.float64)
        if other_w is None:
            oc0 = other_hap[0, idx].astype(np.int64)
            oc1 = other_hap[1, idx].astype(np.int64)
        else:
            oc0 = other_hap[other_w[e, idx], idx].astype(np.int64)
            oc1 = oc0
        for h in (0, 1):
            th = this_hap[h, idx].astype(np.int64)
            d0, d1 = th + oc0, th + oc1
            ll0 = na_i * lp[d0] + nr_i * l1p[d0]
            ll1 = na_i * lp[d1] + nr_i * l1p[d1]
            mix = np.where(
                other_het & (other_w is None),
                np.logaddexp(ll0, ll1) - np.log(2),
                ll0,
            )
            le[e, :, h] = mix
        nz = (nr_i + na_i) == 0
        le[e, nz, :] = 0.0
    return np.nan_to_num(le, nan=0.0, neginf=-1e9)


def _joint_one_parent(
    this_hap: np.ndarray,
    other_hap: np.ndarray,
    pos: np.ndarray,
    biopsies: list[tuple[np.ndarray, np.ndarray]],
    err: float,
    recomb_per_bp: float,
    switch_rate: float,
    other_w: np.ndarray | None = None,
    decode: str = "posterior",
):
    """2^E-state HMM over (w_1,…,w_E) at all of this parent's het sites.

    Transition kernel is a two-component mixture: a shared switch
    (rate `switch_rate` per consecutive het) flips every w_e; an
    independent recombination (rate `recomb_per_bp`·Δpos) flips one.
    Decode is "posterior" (forward-backward, per-site MAP — better for
    genotype accuracy) or "viterbi".

    Returns (w_full (E, M) int8, p_w1 (E, M) float posterior, idx)."""
    M = this_hap.shape[1]
    E = len(biopsies)
    if E > 12:
        raise ValueError(
            f"joint HMM has 2^E = {1 << E} states; E > 12 is impractical. "
            f"Run on subsets and merge re-phased parents, or use "
            f"hmm_recover_switch_aware per embryo."
        )
    idx = np.flatnonzero(this_hap[0] != this_hap[1])
    K = len(idx)
    if K == 0:
        return np.zeros((E, M), np.int8), np.full((E, M), 0.5), idx

    S = 1 << E
    states = np.arange(S, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(E)) & 1).astype(np.int8)  # (S, E)
    popcnt = bits.sum(1).astype(np.int64)
    xor_pop = popcnt[states[:, None] ^ states[None, :]]  # (S, S)

    le = _per_embryo_emission(this_hap, other_hap, idx, biopsies, err, other_w)

    # Collapse het sites where no embryo has a read — emission is
    # flat there, so consecutive such sites contribute only their
    # composed transition, which is again of the switch+recomb form:
    # over k hets, P(net switch) = (1 − (1−2p_s)^k)/2 and per-embryo
    # P(net recomb) = (1 − Π(1−2p_rᵢ))/2.
    has_read = np.zeros(K, dtype=bool)
    for nr, na in biopsies:
        has_read |= (nr[idx] + na[idx]) > 0
    has_read[0] = has_read[-1] = True
    keep = np.flatnonzero(has_read)
    Kc = len(keep)
    seg = np.searchsorted(keep, np.arange(K), side="right") - 1
    seg = np.clip(seg, 0, Kc - 2)
    dpos_full = np.diff(pos[idx]).astype(np.float64)
    p_r_full = np.clip(recomb_per_bp * dpos_full, 1e-12, 0.49)
    log_keep = np.zeros(Kc - 1)
    np.add.at(log_keep, seg[:-1], np.log1p(-2 * p_r_full))
    p_r = np.clip((1 - np.exp(log_keep)) / 2, 1e-12, 0.49)
    n_het_per = np.bincount(seg[:-1], minlength=Kc - 1)
    p_s = np.clip(
        (1 - (1 - 2 * max(switch_rate, 1e-12)) ** n_het_per) / 2, 1e-12, 0.49
    )

    le_c = le[:, keep, :]
    le_joint = np.zeros((Kc, S))
    for e in range(E):
        le_joint += le_c[e][:, bits[:, e]]

    lp_r, l1p_r = np.log(p_r), np.log1p(-p_r)
    lp_s, l1p_s = np.log(p_s), np.log1p(-p_s)
    h = np.arange(E + 1)[None, :]
    a = l1p_s[:, None] + h * lp_r[:, None] + (E - h) * l1p_r[:, None]
    b = lp_s[:, None] + (E - h) * lp_r[:, None] + h * l1p_r[:, None]
    log_T_by_h = np.logaddexp(a, b)  # (Kc-1, E+1)
    K = Kc

    # exp(T) lookup: per gap, only E+1 distinct values (by popcount).
    exp_T_by_h = np.exp(log_T_by_h)  # (K-1, E+1)
    pop_xor = xor_pop  # (S, S) ints in [0, E]

    if decode == "viterbi":
        V = np.full(S, -np.log(S)) + le_joint[0]
        bp = np.empty((K, S), dtype=np.int32)
        for t in range(1, K):
            cand = V[:, None] + log_T_by_h[t - 1, pop_xor]
            bp[t] = np.argmax(cand, axis=0)
            V = cand[bp[t], states] + le_joint[t]
        path = np.empty(K, dtype=np.int64)
        path[-1] = int(np.argmax(V))
        for t in range(K - 2, -1, -1):
            path[t] = bp[t + 1, path[t + 1]]
        w_at_idx = bits[path].T  # (E, K)
        p1_at_idx = w_at_idx.astype(np.float64)
    else:
        # Forward-backward; transition is symmetric in (a,b) so the
        # same matrix serves both directions. Work in linear space
        # with per-step rescaling to avoid the exp() of an S×S matrix.
        log_alpha = np.empty((K, S))
        log_alpha[0] = -np.log(S) + le_joint[0]
        for t in range(1, K):
            m = log_alpha[t - 1].max()
            a_lin = np.exp(log_alpha[t - 1] - m)
            log_alpha[t] = (
                m + np.log(a_lin @ exp_T_by_h[t - 1, pop_xor]) + le_joint[t]
            )
        log_beta = np.zeros((K, S))
        for t in range(K - 2, -1, -1):
            v = log_beta[t + 1] + le_joint[t + 1]
            m = v.max()
            log_beta[t] = m + np.log(exp_T_by_h[t, pop_xor] @ np.exp(v - m))
        log_post = log_alpha + log_beta
        log_post -= log_post.max(1, keepdims=True)
        post = np.exp(log_post)
        post /= post.sum(1, keepdims=True)
        p1_at_idx = (post @ bits.astype(np.float64)).T  # (E, K)
        w_at_idx = (p1_at_idx > 0.5).astype(np.int8)

    # Extend back to all M sites. The HMM ran on collapsed sites
    # idx_c; between them the path may flip at any of the skipped
    # hets. Linear interpolation of the posterior over the original
    # het indices puts the expected flip at the segment midpoint.
    het_rank = np.arange(len(idx))
    rank_c = het_rank[keep]
    w_full = np.empty((E, M), dtype=np.int8)
    p1_full = np.full((E, M), 0.5)
    for e in range(E):
        p1_at_het = np.interp(het_rank, rank_c, p1_at_idx[e])
        p1_full[e] = _extend_path(idx, p1_at_het, M)
        w_full[e] = (p1_full[e] > 0.5).astype(np.int8)
    return w_full, p1_full, idx


def joint_recover(
    par: Parents,
    biopsies: list[tuple[np.ndarray, np.ndarray]],
    err: float,
    recomb_per_bp: float = 1e-8,
    switch_rate: float = 0.01,
    n_iter: int = 2,
    decode: str = "posterior",
):
    """Phase-aware recovery using all embryos jointly.

    Alternates between the paternal and maternal 2^E-state HMMs
    (`n_iter` rounds), each time conditioning on the other parent's
    current path estimate at doubly-heterozygous sites.

    Returns (genos (E, M) int8, dosage (E, M) float, ppaths, mpaths)."""
    M = len(par.pos)
    E = len(biopsies)
    ar = np.arange(M)
    wp = wm = None
    pp1 = pm1 = None
    for _ in range(n_iter):
        wp, pp1, _ = _joint_one_parent(
            par.pat, par.mat, par.pos, biopsies, err,
            recomb_per_bp, switch_rate, other_w=wm, decode=decode,
        )
        wm, pm1, _ = _joint_one_parent(
            par.mat, par.pat, par.pos, biopsies, err,
            recomb_per_bp, switch_rate, other_w=wp, decode=decode,
        )
    genos = np.empty((E, M), dtype=np.int8)
    dosage = np.empty((E, M), dtype=np.float64)
    for e in range(E):
        genos[e] = par.pat[wp[e], ar] + par.mat[wm[e], ar]
        # E[dose] = E[pat allele] + E[mat allele]; at het sites the
        # allele is Bernoulli on the path posterior, at hom sites it's
        # fixed.
        ep = par.pat[0] * (1 - pp1[e]) + par.pat[1] * pp1[e]
        em = par.mat[0] * (1 - pm1[e]) + par.mat[1] * pm1[e]
        dosage[e] = ep + em
    return genos, dosage, wp, wm


def _switch_track_from_paths(w: np.ndarray, idx: np.ndarray, M: int) -> np.ndarray:
    """Given joint-inferred paths w (E, K) at this parent's het sites
    `idx`, estimate the parental switch track ŝ (M,) by majority vote:
    ŝ flips between consecutive hets iff > half the embryos' paths
    flip there. A true recombination in one embryo flips one path; a
    parental switch flips all of them."""
    E, K = w.shape
    if K < 2:
        return np.zeros(M, dtype=np.int8)
    flips = (np.diff(w.astype(np.int64), axis=1) != 0).sum(0) > E // 2
    s_at_idx = np.concatenate(([0], np.cumsum(flips) & 1)).astype(np.int8)
    return _extend_path(idx, s_at_idx, M)


def joint_rephase_recover(
    par: Parents,
    biopsies: list[tuple[np.ndarray, np.ndarray]],
    err: float,
    recomb_per_bp: float = 1e-8,
    switch_rate: float = 0.01,
):
    """Two-stage recovery: (1) joint Viterbi over all embryos to detect
    parental switch errors and re-phase the parents; (2) standard
    per-embryo 4-state HMM on the re-phased parents with the tight
    recombination prior.

    This separates the *shared* problem (where are the parental
    switches?) — which benefits from pooling reads across embryos —
    from the *per-embryo* problem (where did this embryo recombine?),
    which the original HMM already solves well once the parental
    phase is correct. The final pass also uses doubly-het sites at
    full strength.

    Returns (genos (E, M) int8, par_rephased, ŝ_pat, ŝ_mat)."""
    M = len(par.pos)
    E = len(biopsies)

    wp, _, idx_p = _joint_one_parent(
        par.pat, par.mat, par.pos, biopsies, err,
        recomb_per_bp, switch_rate, other_w=None, decode="viterbi",
    )
    wm, _, idx_m = _joint_one_parent(
        par.mat, par.pat, par.pos, biopsies, err,
        recomb_per_bp, switch_rate, other_w=wp, decode="viterbi",
    )
    s_pat = _switch_track_from_paths(
        np.stack([wp[e, idx_p] for e in range(E)]), idx_p, M
    )
    s_mat = _switch_track_from_paths(
        np.stack([wm[e, idx_m] for e in range(E)]), idx_m, M
    )

    pat_re = par.pat.copy()
    pat_re[:, s_pat == 1] = pat_re[::-1, s_pat == 1]
    mat_re = par.mat.copy()
    mat_re[:, s_mat == 1] = mat_re[::-1, s_mat == 1]
    par_re = replace(par, pat=pat_re, mat=mat_re)

    ctx_re = build_hmm_context(par_re, recomb_per_bp)
    genos = np.empty((E, M), dtype=np.int8)
    for e in range(E):
        genos[e], _, _ = hmm_recover(par_re, ctx_re, *biopsies[e], err)
    return genos, par_re, s_pat, s_mat


# ----------------------------------------------- per-chromosome PGS scoring


def load_pgs_for_chrom(chrom: str, par: Parents, weights_dir: Path | None = None):
    """For each weight file present, return
    {pgs_id: (idx_into_par, ea_is_alt(±1), w)} restricted to this chrom."""
    weights_dir = weights_dir or pgs_weights_dir()
    pos_to_idx = {int(p): i for i, p in enumerate(par.pos)}
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for wf in sorted(weights_dir.glob("*_hmPOS_GRCh38.txt.gz")):
        pgs_id = wf.name.split("_")[0]
        idxs, signs, ws = [], [], []
        with gzip.open(wf, "rt") as f:
            cols: dict[str, int] = {}
            for line in f:
                if not line.startswith("#"):
                    cols = {c: i for i, c in enumerate(line.rstrip("\n").split("\t"))}
                    break
            if not cols or "chr_name" not in cols:
                continue
            i_chr = cols["chr_name"]
            i_pos = cols["chr_position"]
            i_ea = cols["effect_allele"]
            i_oa = cols.get("other_allele", cols.get("hm_inferOtherAllele", -1))
            i_w = cols["effect_weight"]
            for line in f:
                if line.startswith("#"):
                    continue
                r = line.rstrip("\n").split("\t")
                if r[i_chr].lstrip("chr") != chrom:
                    continue
                try:
                    p = int(r[i_pos])
                    w = float(r[i_w])
                except (ValueError, IndexError):
                    continue
                j = pos_to_idx.get(p)
                if j is None:
                    continue
                ea = r[i_ea].upper()
                oa = r[i_oa].upper() if 0 <= i_oa < len(r) else ""
                if ea == par.alt[j] and (oa in (par.ref[j], "")):
                    idxs.append(j)
                    signs.append(1)
                    ws.append(w)
                elif ea == par.ref[j] and (oa in (par.alt[j], "")):
                    idxs.append(j)
                    signs.append(-1)
                    ws.append(w)
        if idxs:
            out[pgs_id] = (np.array(idxs), np.array(signs, dtype=np.int8), np.array(ws))
    return out


def score_chrom(geno: np.ndarray, pgs_map) -> dict[str, float]:
    """ALT-dosage → effect-allele dosage = geno where sign=+1, 2−geno
    where sign=−1; raw chromosome score per PGS id."""
    out = {}
    for pid, (idx, sgn, w) in pgs_map.items():
        d = np.where(sgn == 1, geno[idx], 2 - geno[idx])
        out[pid] = float((w * d).sum())
    return out
