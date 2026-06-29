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
lives in `genepred.embryo_cli` (CLI: `genepred embryo-demo`); the
phasing-error benchmark and real-data validation are under
`validation/`. See docs/PHASING.md for the full writeup.
"""

from __future__ import annotations

import gzip
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numba as nb
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


def load_parents_vcf(
    vcf: Path | str, chrom: str, father: str, mother: str
) -> Parents:
    """Load phased haplotypes for two named samples from any VCF.

    Keeps biallelic SNPs on `chrom` where both samples are phased
    (`a|b`). The VCF should already be phased — via SHAPEIT5,
    Beagle, long-read WhatsHap, or trio phasing — since the embryo
    HMM relies on the parental haplotype labels being consistent."""
    pos, ref, alt, pat, mat = [], [], [], [], []
    fc = mc = -1
    want = chrom.lstrip("chr")
    with gzip.open(vcf, "rt") if str(vcf).endswith(".gz") else open(vcf) as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                hdr = line.rstrip().split("\t")
                fc, mc = hdr.index(father), hdr.index(mother)
                continue
            r = line.rstrip().split("\t")
            if r[0].lstrip("chr") != want:
                continue
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
    if not pos:
        raise ValueError(
            f"no phased biallelic SNPs on chr{chrom} for {father}/{mother} in {vcf}"
        )
    return Parents(
        chrom=want,
        pos=np.array(pos, dtype=np.int64),
        ref=np.array(ref, dtype="<U1"),
        alt=np.array(alt, dtype="<U1"),
        pat=np.array(pat, dtype=np.int8).T,
        mat=np.array(mat, dtype=np.int8).T,
    )


def load_embryo_reads(
    vcf: Path | str, par: Parents, sample: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-site (n_ref, n_alt) read counts at the parents' SNP
    positions from a low-coverage embryo VCF.

    Reads the FORMAT/AD field (ref,alt depth) where present;
    falls back to RO/AO. Sites in `par` not present in the embryo
    VCF get (0, 0). REF/ALT are checked against the parents and
    swapped if reversed; mismatches are dropped."""
    M = len(par.pos)
    n_ref = np.zeros(M, dtype=np.int64)
    n_alt = np.zeros(M, dtype=np.int64)
    pos2i = {int(p): i for i, p in enumerate(par.pos)}
    want = par.chrom
    sc = -1
    fmt_ad = fmt_ro = fmt_ao = -1
    with gzip.open(vcf, "rt") if str(vcf).endswith(".gz") else open(vcf) as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                hdr = line.rstrip().split("\t")
                sc = hdr.index(sample) if sample else 9
                continue
            r = line.rstrip().split("\t")
            if r[0].lstrip("chr") != want:
                continue
            i = pos2i.get(int(r[1]))
            if i is None:
                continue
            fmt = r[8].split(":")
            if fmt_ad < 0 and "AD" in fmt:
                fmt_ad = fmt.index("AD")
            if fmt_ro < 0 and "RO" in fmt:
                fmt_ro = fmt.index("RO")
                fmt_ao = fmt.index("AO") if "AO" in fmt else -1
            cell = r[sc].split(":")
            nr = na = 0
            if 0 <= fmt_ad < len(cell):
                ad = cell[fmt_ad].split(",")
                if len(ad) >= 2 and ad[0] != ".":
                    nr, na = int(ad[0]), int(ad[1])
            elif 0 <= fmt_ro < len(cell):
                nr = int(cell[fmt_ro]) if cell[fmt_ro] != "." else 0
                na = (
                    int(cell[fmt_ao].split(",")[0])
                    if 0 <= fmt_ao < len(cell) and cell[fmt_ao] != "."
                    else 0
                )
            if r[3] == par.ref[i] and r[4] == par.alt[i]:
                n_ref[i], n_alt[i] = nr, na
            elif r[3] == par.alt[i] and r[4] == par.ref[i]:
                n_ref[i], n_alt[i] = na, nr
    return n_ref, n_alt


def load_parents(chrom: str, father: str, mother: str) -> Parents:
    """Load phased haplotypes for two samples from the 1KG VCF."""
    vcf = next(kg_dir().glob(f"ALL.chr{chrom}.phase3_*v5b.*.genotypes.vcf.gz"))
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


def align_relative_to_parents(
    by_pos: dict, par: Parents, *, min_match: float = 0.2
) -> tuple[np.ndarray, int]:
    """Map a relative's genotypes (genepred.io.load_genotypes by_pos dict)
    onto the parents' site grid as ALT-dosage 0/1/2 (-1 = missing).

    Handles the boring failure modes so callers don't have to: imputed
    (ref, alt, dosage) tuples are hard-called; strand flips are resolved
    by complement; strand-ambiguous A/T and C/G grid sites are dropped
    entirely — for anchoring, one confidently wrong homozygous call is
    worse than a missing one, and a palindromic SNP's direct-vs-flipped
    report is indistinguishable. Raises if the fraction of grid sites
    with a usable call is below `min_match` — that pattern means a
    wrong genome build (positions don't line up), not a sparse chip.

    Returns (geno (M,) int8, n_matched)."""
    from genepred.io import COMPLEMENT, hard_call

    M = len(par.pos)
    geno = np.full(M, -1, dtype=np.int8)
    n_matched = 0
    # Iterate the (sparser) relative, not the grid: cost scales with the
    # chip, and grid sites the chip lacks stay missing for free.
    pos2i = {int(p): i for i, p in enumerate(par.pos)}
    for (chrom, pos), g in by_pos.items():
        if chrom != par.chrom:
            continue
        i = pos2i.get(pos)
        if i is None:
            continue
        ref, alt = par.ref[i], par.alt[i]
        if ref == alt.translate(COMPLEMENT):
            continue  # palindromic A/T or C/G site: unorientable
        a1, a2 = hard_call(g)
        pair = {ref, alt}
        if {a1, a2} <= pair:
            geno[i] = (a1 == alt) + (a2 == alt)
            n_matched += 1
        elif {a1.translate(COMPLEMENT), a2.translate(COMPLEMENT)} <= pair:
            c1, c2 = a1.translate(COMPLEMENT), a2.translate(COMPLEMENT)
            geno[i] = (c1 == alt) + (c2 == alt)
            n_matched += 1
    if M and n_matched / M < min_match:
        raise ValueError(
            f"relative matches only {n_matched:,}/{M:,} "
            f"({n_matched / M:.0%}) of the parental grid on chr{par.chrom} — "
            f"likely a genome-build mismatch (parents are GRCh37); "
            f"lift over or re-export the relative's data."
        )
    return geno, n_matched


def simulate_grandparents(
    hap2: np.ndarray, rng: np.random.Generator, geno_err: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Unphased genotypes for the two parents of a person with haplotypes
    `hap2` (2, M): grandparent k contributed hap k; their other allele is
    drawn from a per-site frequency proxy. `geno_err` flips single calls
    at that rate (for benching the Mendelian-conflict filter).

    Returns (gp1_geno, gp2_geno), each (M,) int dosage 0/1/2."""
    M = hap2.shape[1]
    af = np.clip(hap2.mean(0), 0.05, 0.95)
    gps = []
    for k in (0, 1):
        g = hap2[k].astype(np.int64) + (rng.random(M) < af).astype(np.int64)
        if geno_err > 0:
            hit = rng.random(M) < geno_err
            shift = rng.integers(1, 3, size=M)  # ±1/±2 mod 3 keeps it in range
            g[hit] = (g[hit] + shift[hit]) % 3
        gps.append(g)
    return gps[0], gps[1]


def _gp_hom_allele(g: np.ndarray | None) -> np.ndarray | None:
    """Per-site homozygous allele of a grandparent genotype (0/1), or -1
    where the grandparent is het or missing (negative dosage = no call)."""
    return None if g is None else np.where((g == 0) | (g == 2), g // 2, -1)


def mendelian_conflicts(
    child_geno: np.ndarray,
    gp1_geno: np.ndarray | None,
    gp2_geno: np.ndarray | None,
) -> np.ndarray:
    """Sites where the child's genotype is Mendelian-impossible given the
    grandparents — almost always a genotyping error in one of the three
    (expect ~0.1–1 % of sites), or a build/sample mix-up when the rate
    is much larger.

    A homozygous grandparent must transmit its allele, so the child must
    carry ≥1 copy of it; if both grandparents are homozygous the child's
    genotype is fully determined. Missing calls (negative) never
    conflict. Returns (M,) bool."""
    a1, a2 = _gp_hom_allele(gp1_geno), _gp_hom_allele(gp2_geno)
    conflict = np.zeros(len(child_geno), dtype=bool)
    for a in (a1, a2):
        if a is None:
            continue
        conflict |= ((a == 1) & (child_geno == 0)) | ((a == 0) & (child_geno == 2))
    if a1 is not None and a2 is not None:
        both = (a1 >= 0) & (a2 >= 0) & (child_geno >= 0)
        conflict |= both & (child_geno != a1 + a2)
    return conflict


def trio_phase(
    child_geno: np.ndarray,
    gp1_geno: np.ndarray | None,
    gp2_geno: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mendelian phasing of `child_geno` (the embryo's parent) from
    one or both of *their* parents' unphased genotypes (dosage 0/1/2;
    negative = missing call).

    At each het site of the child where at least one grandparent is
    homozygous, the parent-of-origin of each allele is determined.
    Resolves ~70–90 % of hets with zero switch errors; the remaining
    sites (both grandparents het) are left as 0|1 — they are
    individually ambiguous but, being scattered, do not introduce
    long-range switches. Mendelian-conflict sites (genotyping errors —
    see mendelian_conflicts) are excluded rather than allowed to force
    a wrong anchor.

    Returns (hap (2, M) int8 ordered [from-gp1, from-gp2], resolved (M,) bool)."""
    hap, resolved, _ = _trio_phase_full(child_geno, gp1_geno, gp2_geno)
    return hap, resolved


def _trio_phase_full(
    child_geno: np.ndarray,
    gp1_geno: np.ndarray | None,
    gp2_geno: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """trio_phase plus the Mendelian-conflict mask it used (computed once,
    so callers reporting conflict counts can't drift from the masking)."""
    M = len(child_geno)
    hap = np.zeros((2, M), dtype=np.int8)
    hom = (child_geno == 2).astype(np.int8)
    hap[0] = hap[1] = hom
    het = child_geno == 1
    resolved = ~het
    bad = mendelian_conflicts(child_geno, gp1_geno, gp2_geno)

    a1, a2 = _gp_hom_allele(gp1_geno), _gp_hom_allele(gp2_geno)
    if a1 is not None:
        m = het & ~bad & (a1 >= 0)
        hap[0, m] = a1[m]
        hap[1, m] = 1 - a1[m]
        resolved |= m
    if a2 is not None:
        m = het & ~bad & ~resolved & (a2 >= 0)
        hap[1, m] = a2[m]
        hap[0, m] = 1 - a2[m]
        resolved |= m
    resolved &= ~(bad & het)
    hap[1, het & ~resolved] = 1  # arbitrary 0|1 at unresolved hets
    return hap, resolved, bad


def anchor_correct_phase(
    hap: np.ndarray,
    child_geno: np.ndarray,
    gp1_geno: np.ndarray | None,
    gp2_geno: np.ndarray | None,
    *,
    anchor_err: float = 0.005,
    switch_prior: float = 0.01,
) -> tuple[np.ndarray, dict]:
    """Fix long-range switch errors in a statistically phased haplotype
    pair using Mendelian anchors from one or two grandparents.

    Grandparent-resolved het sites give the true hap orientation at
    ~70–90 % of hets with zero switch error; the statistical phase
    supplies the within-segment detail the anchors can't see. A 2-state
    Viterbi over the anchor sequence (states: hap rows swapped / not;
    emission error `anchor_err` absorbs genotyping errors that slip
    past the Mendelian-conflict filter; transition probability scales
    `switch_prior` by the het distance between anchors) finds the
    maximum-likelihood piecewise flip track, applied to `hap` with cuts
    at midpoints between disagreeing anchors.

    The statistical phase is kept wherever anchors are silent, so
    isolated point flips survive — those are harmless to the embryo
    HMM; only the long-range switches it cannot follow are removed.

    Returns (hap_corrected (2, M) int8, info dict: n_anchors,
    n_conflicts, n_flip_segments, residual_anchor_disagreement)."""
    M = hap.shape[1]
    anchor_hap, resolved, bad = _trio_phase_full(child_geno, gp1_geno, gp2_geno)
    n_conflicts = int(bad.sum())
    het = hap[0] != hap[1]
    anchors = np.flatnonzero(resolved & het & (child_geno == 1))
    info = {
        "n_anchors": int(len(anchors)),
        "n_conflicts": n_conflicts,
        "n_flip_segments": 0,
        "residual_anchor_disagreement": 0.0,
    }
    if len(anchors) < 2:
        return hap.copy(), info

    agree = hap[0, anchors] == anchor_hap[0, anchors]
    het_idx = np.flatnonzero(het)
    gaps = np.diff(np.searchsorted(het_idx, anchors))
    p_sw = np.clip(switch_prior * gaps, 1e-9, 0.5)
    le_bad, le_ok = np.log(anchor_err), np.log1p(-anchor_err)
    em = np.empty((len(anchors), 2))
    em[:, 0] = np.where(agree, le_ok, le_bad)  # state 0: rows as published
    em[:, 1] = np.where(agree, le_bad, le_ok)  # state 1: rows swapped

    n = len(anchors)
    dp = np.empty((n, 2))
    bp = np.zeros((n, 2), dtype=np.int8)
    dp[0] = em[0]
    lsw, lst = np.log(p_sw), np.log1p(-p_sw)
    for i in range(1, n):
        for s in (0, 1):
            stay = dp[i - 1, s] + lst[i - 1]
            swap = dp[i - 1, 1 - s] + lsw[i - 1]
            if stay >= swap:
                dp[i, s] = stay + em[i, s]
                bp[i, s] = s
            else:
                dp[i, s] = swap + em[i, s]
                bp[i, s] = 1 - s
    states = np.empty(n, dtype=np.int8)
    states[-1] = int(dp[-1, 1] > dp[-1, 0])
    for i in range(n - 2, -1, -1):
        states[i] = bp[i + 1, states[i + 1]]

    # Piecewise-constant flip track: segment boundaries at midpoints
    # between consecutive anchors whose Viterbi states differ.
    flip = np.zeros(M, dtype=bool)
    bounds = [int((anchors[c] + anchors[c + 1] + 1) // 2)
              for c in np.flatnonzero(np.diff(states))]
    state = bool(states[0])
    prev = 0
    for cut in [*bounds, M]:
        if state:
            flip[prev:cut] = True
        prev = cut
        state = not state

    out = hap.copy()
    out[:, flip] = out[::-1, flip]
    pred_agree = np.where(flip[anchors], ~agree, agree)
    info["n_flip_segments"] = int(states[0]) + int((np.diff(states) == 1).sum())
    info["residual_anchor_disagreement"] = float(1.0 - pred_agree.mean())
    return out, info


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
    geno: np.ndarray,
    coverage: float,
    err: float,
    rng: np.random.Generator,
    *,
    ado: float = 0.0,
    cov_dispersion: float = 0.0,
):
    """Per-SNP (n_ref, n_alt) read counts after whole-genome
    amplification of a few-cell trophectoderm biopsy.

    coverage: mean depth after WGA.
    err: per-base sequencing error.
    ado: allelic-dropout probability per heterozygous site — one of
        the two alleles fails to amplify and all reads at that site
        come from the surviving allele. MDA-based WGA gives ~10–25 %;
        MALBAC and PTA are lower (~1–5 %). The default 0 reproduces
        the original idealised model.
    cov_dispersion: extra per-site coverage variance from amplification
        bias, expressed as the coefficient of variation of a Gamma
        multiplier (so the marginal read count is negative-binomial).
        MDA typically has CV ≈ 0.5–1; 0 = plain Poisson."""
    M = len(geno)
    eff = geno.astype(np.float64)
    if ado > 0:
        het = geno == 1
        drop = het & (rng.random(M) < ado)
        eff[drop] = rng.integers(0, 2, size=int(drop.sum())) * 2
    lam = coverage
    if cov_dispersion > 0:
        k = 1.0 / (cov_dispersion**2)
        lam = coverage * rng.gamma(k, 1.0 / k, size=M)
    n_reads = rng.poisson(lam, size=M)
    p_alt = eff / 2 * (1 - err) + (1 - eff / 2) * err
    n_alt = rng.binomial(n_reads, p_alt)
    return n_reads - n_alt, n_alt


# ----------------------------------------------------------- HMM recovery


_STATES = np.array([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=np.int8)


def _split_switch_rate(
    switch_rate: float | tuple[float, float],
) -> tuple[float, float]:
    """(paternal, maternal) switch rates from a scalar or pair."""
    if isinstance(switch_rate, (tuple, list)):
        return float(switch_rate[0]), float(switch_rate[1])
    return float(switch_rate), float(switch_rate)


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


@nb.njit(cache=True, fastmath=True)
def _wht_inplace(a: np.ndarray) -> None:
    """Iterative in-place WHT on a 1-D length-2^E array."""
    n = a.shape[0]
    h = 1
    while h < n:
        i = 0
        while i < n:
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
            i += 2 * h
        h *= 2


@nb.njit(cache=True, fastmath=True)
def _emit_mul(v: np.ndarray, g: np.ndarray, t: int, E: int, two: bool) -> None:
    """v[s] *= Πₑ g[e, t, state-of-embryo-e-in-s]. For the per-parent
    HMM g is (E, K, 2) indexed by bitₑ(s); for both-parents g is
    (E, K, 4) indexed by 2·bitₑ(s) + bit_{E+e}(s)."""
    S = v.shape[0]
    for s in range(S):
        p = 1.0
        for e in range(E):
            if two:
                p *= g[e, t, (((s >> e) & 1) << 1) | ((s >> (E + e)) & 1)]
            else:
                p *= g[e, t, (s >> e) & 1]
        v[s] *= p


@nb.njit(cache=True, fastmath=True)
def _fb_wht(
    g: np.ndarray,
    n_bits: int,
    p_r: np.ndarray,
    p_s: np.ndarray,
    popcnt: np.ndarray,
    pop2: np.ndarray,
    p_s2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear-space rescaled forward-backward with WHT-diagonalised
    transition and factored emission.

    g is the per-embryo emission likelihood: (E, K, 2) for the
    per-parent HMM, (E, K, 4) for the both-parents HMM. n_bits is E
    or 2E. For both-parents, popcnt is the paternal half-popcount and
    pop2/p_s2 the maternal counterparts; pass length-0 arrays for
    the per-parent case.

    Returns (alpha, beta), each (K, 2^n_bits) row-normalised; the
    posterior is their elementwise product, re-normalised."""
    E, K = g.shape[0], g.shape[1]
    two = pop2.shape[0] > 0
    S = 1 << n_bits
    inv_S = 1.0 / S

    alpha = np.empty((K, S))
    beta = np.empty((K, S))
    v = np.empty(S)
    lam = np.empty(S)
    rpow = np.empty(n_bits + 1)

    for s in range(S):
        v[s] = 1.0
    _emit_mul(v, g, 0, E, two)
    z = 0.0
    for s in range(S):
        z += v[s]
    z = z if z > 0 else 1.0
    for s in range(S):
        alpha[0, s] = v[s] / z

    for direction in range(2):
        if direction == 0:
            t0, t1, dt = 1, K, 1
        else:
            for s in range(S):
                beta[K - 1, s] = inv_S
            t0, t1, dt = K - 2, -1, -1
        for t in range(t0, t1, dt):
            gp = t - 1 if direction == 0 else t
            r = 1.0 - 2.0 * p_r[gp]
            sp = 1.0 - 2.0 * p_s[gp]
            sm = 1.0 - 2.0 * p_s2[gp] if two else 1.0
            rpow[0] = 1.0
            for k in range(1, n_bits + 1):
                rpow[k] = rpow[k - 1] * r
            for s in range(S):
                pc = popcnt[s] + (pop2[s] if two else 0)
                ev = rpow[pc]
                if popcnt[s] & 1:
                    ev *= sp
                if two and (pop2[s] & 1):
                    ev *= sm
                lam[s] = ev

            if direction == 0:
                for s in range(S):
                    v[s] = alpha[t - 1, s]
                _wht_inplace(v)
                for s in range(S):
                    v[s] *= lam[s]
                _wht_inplace(v)
                for s in range(S):
                    if v[s] < 0.0:
                        v[s] = 0.0
                _emit_mul(v, g, t, E, two)
            else:
                for s in range(S):
                    v[s] = beta[t + 1, s]
                _emit_mul(v, g, t + 1, E, two)
                _wht_inplace(v)
                for s in range(S):
                    v[s] *= lam[s]
                _wht_inplace(v)
                for s in range(S):
                    if v[s] < 0.0:
                        v[s] = 0.0
            z = 0.0
            for s in range(S):
                z += v[s]
            iz = 1.0 / z if z > 0 else 0.0
            if direction == 0:
                for s in range(S):
                    alpha[t, s] = v[s] * iz
            else:
                for s in range(S):
                    beta[t, s] = v[s] * iz
    return alpha, beta


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
    idx = np.flatnonzero(this_hap[0] != this_hap[1])
    K = len(idx)
    if K == 0:
        return np.zeros((E, M), np.int8), np.full((E, M), 0.5), idx

    S = 1 << E
    states = np.arange(S, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(E)) & 1).astype(np.int8)  # (S, E)
    popcnt = bits.sum(1).astype(np.int64)
    xor_pop = popcnt[states[:, None] ^ states[None, :]] if decode == "viterbi" else None

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
    K = Kc

    if decode == "viterbi":
        # Viterbi's max doesn't factor over the Kronecker structure,
        # so this stays O(S²) per step.
        le_joint = np.zeros((Kc, S))
        for e in range(E):
            le_joint += le_c[e][:, bits[:, e]]
        lp_r, l1p_r = np.log(p_r), np.log1p(-p_r)
        lp_s, l1p_s = np.log(p_s), np.log1p(-p_s)
        h = np.arange(E + 1)[None, :]
        a = l1p_s[:, None] + h * lp_r[:, None] + (E - h) * l1p_r[:, None]
        b = lp_s[:, None] + (E - h) * lp_r[:, None] + h * l1p_r[:, None]
        log_T_by_h = np.logaddexp(a, b)  # (Kc-1, E+1)
        pop_xor = xor_pop
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
        # Forward-backward. R = [[1−p_r, p_r],[p_r, 1−p_r]] and
        # RX = [[p_r, 1−p_r],[1−p_r, p_r]] share the Hadamard
        # eigenbasis with eigenvalues (1, ±(1−2p_r)), so
        #   T = (1−p_s)·R^⊗E + p_s·(RX)^⊗E = H^⊗E·diag(λ)·H^⊗E/2^E,
        #   λ[s] = (1−2p_r)^|s| · (1 − 2p_s·[|s| odd]).
        # The loop runs JIT-compiled in linear space with per-step
        # rescaling and the emission applied per-embryo (factored),
        # so there are no per-state exp/log calls and the (K, S)
        # joint emission table is never materialised.
        g = np.exp(np.ascontiguousarray(le_c))  # (E, Kc, 2)
        alpha, beta = _fb_wht(
            g, E,
            np.ascontiguousarray(p_r),
            np.ascontiguousarray(p_s),
            np.ascontiguousarray(popcnt),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
        post = alpha * beta
        post /= np.maximum(post.sum(1, keepdims=True), 1e-300)
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
    switch_rate: float | tuple[float, float] = 0.01,
    n_iter: int = 2,
    decode: str = "posterior",
):
    """Phase-aware recovery using all embryos jointly.

    Alternates between the paternal and maternal 2^E-state HMMs
    (`n_iter` rounds), each time conditioning on the other parent's
    current path estimate at doubly-heterozygous sites.

    `switch_rate` may be a scalar or a (paternal, maternal) pair — use
    the pair when one parent's phase is anchored by a relative
    (anchor_correct_phase / trio phasing) and the other is statistical.

    Returns (genos, dosage, var_dose, ppaths, mpaths) each (E, M)."""
    sr_pat, sr_mat = _split_switch_rate(switch_rate)
    M = len(par.pos)
    E = len(biopsies)
    if E > 16:
        raise ValueError(
            f"joint_recover with E={E}: 2^{E}={1 << E} states is impractical. "
            f"Run joint_rephase_recover on a subset of ≤10 embryos, then "
            f"hmm_recover per embryo on the re-phased parents."
        )
    if E > 10:
        print(
            f"[embryo] joint_recover with E={E} → 2^{E}={1 << E} states "
            f"(WHT-factored); expect minutes per chromosome.",
            file=sys.stderr,
        )
    ar = np.arange(M)
    wp = wm = None
    pp1 = pm1 = None
    for _ in range(n_iter):
        wp, pp1, _ = _joint_one_parent(
            par.pat, par.mat, par.pos, biopsies, err,
            recomb_per_bp, sr_pat, other_w=wm, decode=decode,
        )
        wm, pm1, _ = _joint_one_parent(
            par.mat, par.pat, par.pos, biopsies, err,
            recomb_per_bp, sr_mat, other_w=wp, decode=decode,
        )
    genos = np.empty((E, M), dtype=np.int8)
    dosage = np.empty((E, M), dtype=np.float64)
    var_dose = np.zeros((E, M), dtype=np.float64)
    pat_het = (par.pat[0] != par.pat[1]).astype(np.float64)
    mat_het = (par.mat[0] != par.mat[1]).astype(np.float64)
    for e in range(E):
        genos[e] = par.pat[wp[e], ar] + par.mat[wm[e], ar]
        ep = par.pat[0] * (1 - pp1[e]) + par.pat[1] * pp1[e]
        em = par.mat[0] * (1 - pm1[e]) + par.mat[1] * pm1[e]
        dosage[e] = ep + em
        # Var[dose] = Var[pat allele] + Var[mat allele]; each is
        # Bernoulli with p = path posterior, but only at that parent's
        # het sites (hom sites contribute zero variance).
        var_dose[e] = pp1[e] * (1 - pp1[e]) * pat_het + pm1[e] * (1 - pm1[e]) * mat_het
    return genos, dosage, var_dose, wp, wm


def score_with_uncertainty(
    dosage: np.ndarray, var_dose: np.ndarray, pgs_map
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-embryo PGS point estimate and SE from posterior dosage.

    SE assumes independent per-site errors, which underestimates the
    true uncertainty because mis-called switch segments produce
    correlated errors. Treat as a lower bound; for between-embryo
    comparisons it's still informative because the correlated
    component (the parental switch track) is shared and largely
    cancels in differences.

    Returns {pgs_id: (score (E,), se (E,))}."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pid, (idx, sgn, w) in pgs_map.items():
        d = np.where(sgn[None, :] == 1, dosage[:, idx], 2 - dosage[:, idx])
        score = (w[None, :] * d).sum(1)
        se = np.sqrt((w[None, :] ** 2 * var_dose[:, idx]).sum(1))
        out[pid] = (score, se)
    return out


def joint_recover_full(
    par: Parents,
    biopsies: list[tuple[np.ndarray, np.ndarray]],
    err: float,
    recomb_per_bp: float = 1e-8,
    switch_rate: float | tuple[float, float] = 0.01,
):
    """Exact 4^E-state joint over both parents simultaneously.

    State is (w₁ᵖ,…,w_Eᵖ, w₁ᵐ,…,w_Eᵐ) ∈ {0,1}^{2E}. The transition is
    a mixture over the four (paternal-switch, maternal-switch)
    events; in the Hadamard eigenbasis its eigenvalues factor as
        λ[s] = (1−2p_r)^{|sᵖ|+|sᵐ|}·(1−2p_s·[|sᵖ| odd])·(1−2p_s·[|sᵐ| odd]),
    so Tα = WHT₂E(λ ⊙ WHT₂E(α))/2^{2E} in O(E·4^E) per step.

    This removes the coordinate-ascent approximation that
    `joint_recover` makes at doubly-heterozygous sites; otherwise
    the model is identical. Practical to E≈8.

    Returns (genos, dosage, var_dose, ppaths, mpaths) each (E, M)."""
    M = len(par.pos)
    E = len(biopsies)
    S = 1 << (2 * E)
    if E > 9:
        raise ValueError(f"4^E = {S} states; E > 9 is impractical here")
    states = np.arange(S, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(2 * E)) & 1).astype(np.int8)  # (S, 2E)
    pop_p = bits[:, :E].sum(1).astype(np.int64)
    pop_m = bits[:, E:].sum(1).astype(np.int64)

    idx = np.flatnonzero((par.pat[0] != par.pat[1]) | (par.mat[0] != par.mat[1]))
    K = len(idx)
    ar = np.arange(M)

    # Per-embryo log P(reads | wᵖ, wᵐ) at each informative site —
    # a (E, K, 2, 2) table; the joint emission is its sum over e.
    le4 = np.zeros((E, K, 2, 2))
    lp = np.array(
        [np.log(d / 2 * (1 - err) + (1 - d / 2) * err) for d in range(3)]
    )
    l1p = np.log1p(-np.exp(lp))
    for e, (nr, na) in enumerate(biopsies):
        nr_i, na_i = nr[idx].astype(np.float64), na[idx].astype(np.float64)
        for hp in (0, 1):
            for hm in (0, 1):
                d = par.pat[hp, idx].astype(np.int64) + par.mat[hm, idx].astype(np.int64)
                le4[e, :, hp, hm] = na_i * lp[d] + nr_i * l1p[d]
        nz = (nr_i + na_i) == 0
        le4[e, nz] = 0.0
    le4 = np.nan_to_num(le4, neginf=-1e9)

    has_read = np.zeros(K, dtype=bool)
    for nr, na in biopsies:
        has_read |= (nr[idx] + na[idx]) > 0
    has_read[0] = has_read[-1] = True
    keep = np.flatnonzero(has_read)
    Kc = len(keep)
    seg = np.clip(np.searchsorted(keep, np.arange(K), side="right") - 1, 0, Kc - 2)

    # Composite p_r, paternal p_s, maternal p_s per collapsed gap
    # (paternal switch only at paternal hets, maternal at maternal hets).
    dpos = np.diff(par.pos[idx]).astype(np.float64)
    pr_full = np.clip(recomb_per_bp * dpos, 1e-12, 0.49)
    log_keep = np.zeros(Kc - 1)
    np.add.at(log_keep, seg[:-1], np.log1p(-2 * pr_full))
    p_r = np.clip((1 - np.exp(log_keep)) / 2, 1e-12, 0.49)
    n_phet = np.bincount(
        seg[:-1], weights=(par.pat[0, idx] != par.pat[1, idx])[:-1], minlength=Kc - 1
    )
    n_mhet = np.bincount(
        seg[:-1], weights=(par.mat[0, idx] != par.mat[1, idx])[:-1], minlength=Kc - 1
    )
    sr_pat, sr_mat = _split_switch_rate(switch_rate)
    p_sp = np.clip((1 - (1 - 2 * max(sr_pat, 1e-12)) ** n_phet) / 2, 1e-12, 0.49)
    p_sm = np.clip((1 - (1 - 2 * max(sr_mat, 1e-12)) ** n_mhet) / 2, 1e-12, 0.49)

    g4 = np.exp(np.ascontiguousarray(le4[:, keep].reshape(E, Kc, 4)))
    alpha, beta = _fb_wht(
        g4, 2 * E,
        np.ascontiguousarray(p_r),
        np.ascontiguousarray(p_sp),
        np.ascontiguousarray(pop_p),
        np.ascontiguousarray(pop_m),
        np.ascontiguousarray(p_sm),
    )
    post = alpha * beta
    post /= np.maximum(post.sum(1, keepdims=True), 1e-300)

    # Marginals P(w_e^p = 1), P(w_e^m = 1) per embryo at collapsed sites.
    pp1c = (post @ bits[:, :E].astype(np.float64)).T  # (E, Kc)
    pm1c = (post @ bits[:, E:].astype(np.float64)).T

    het_rank = np.arange(K)
    rank_c = het_rank[keep]
    genos = np.empty((E, M), dtype=np.int8)
    dosage = np.empty((E, M))
    var_dose = np.zeros((E, M))
    pat_het = (par.pat[0] != par.pat[1]).astype(np.float64)
    mat_het = (par.mat[0] != par.mat[1]).astype(np.float64)
    wp = np.empty((E, M), dtype=np.int8)
    wm = np.empty((E, M), dtype=np.int8)
    for e in range(E):
        pp1 = _extend_path(idx, np.interp(het_rank, rank_c, pp1c[e]), M)
        pm1 = _extend_path(idx, np.interp(het_rank, rank_c, pm1c[e]), M)
        wp[e] = (pp1 > 0.5).astype(np.int8)
        wm[e] = (pm1 > 0.5).astype(np.int8)
        genos[e] = par.pat[wp[e], ar] + par.mat[wm[e], ar]
        dosage[e] = (
            par.pat[0] * (1 - pp1) + par.pat[1] * pp1
            + par.mat[0] * (1 - pm1) + par.mat[1] * pm1
        )
        var_dose[e] = pp1 * (1 - pp1) * pat_het + pm1 * (1 - pm1) * mat_het
    return genos, dosage, var_dose, wp, wm


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
    switch_rate: float | tuple[float, float] = 0.01,
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
    sr_pat, sr_mat = _split_switch_rate(switch_rate)

    wp, _, idx_p = _joint_one_parent(
        par.pat, par.mat, par.pos, biopsies, err,
        recomb_per_bp, sr_pat, other_w=None, decode="viterbi",
    )
    wm, _, idx_m = _joint_one_parent(
        par.mat, par.pat, par.pos, biopsies, err,
        recomb_per_bp, sr_mat, other_w=wp, decode="viterbi",
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
    {pgs_id: (idx_into_par, ea_is_alt(±1), w)} restricted to this chrom.

    Matching is by the file's native chr_name/chr_position against the
    parents' GRCh37 grid, so files whose native build isn't GRCh37 are
    skipped entirely — position-matching GRCh38 (or unreported-build)
    coordinates against a GRCh37 grid silently scores coincidental wrong
    SNPs, which showed up as constant multi-sigma offsets in the embryo
    report (IBD +4σ, AF −8σ, ADHD −13σ for every embryo)."""
    weights_dir = weights_dir or pgs_weights_dir()
    pos_to_idx = {int(p): i for i, p in enumerate(par.pos)}
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for wf in sorted(weights_dir.glob("*_hmPOS_GRCh38.txt.gz")):
        pgs_id = wf.name.split("_")[0]
        idxs, signs, ws = [], [], []
        with gzip.open(wf, "rt") as f:
            cols: dict[str, int] = {}
            build = "GRCh37"
            for line in f:
                if not line.startswith("#") or "\teffect_weight" in line:
                    # column row (some files prefix it with '#')
                    hdr = line.lstrip("#").rstrip("\n").split("\t")
                    cols = {c: i for i, c in enumerate(hdr)}
                    break
                if "genome_build=" in line:
                    build = line.split("=")[-1].strip()
            if build not in ("GRCh37", "hg19", "NCBI37"):
                continue
            if not cols or "chr_name" not in cols or "chr_position" not in cols:
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


def grid_effect_af(par: Parents, af_by_pos: dict) -> np.ndarray:
    """Per-grid-site ALT-allele frequency from an AF-sidecar by_pos lookup
    ((chrom, pos) -> (ref, alt, af_alt)); NaN where the site is unknown or
    the alleles don't match either orientation."""
    af = np.full(len(par.pos), np.nan)
    chrom = par.chrom
    for j, p in enumerate(par.pos):
        rec = af_by_pos.get((chrom, int(p)))
        if rec is None:
            continue
        ref, alt, a = rec
        if ref == par.ref[j] and alt == par.alt[j]:
            af[j] = a
        elif ref == par.alt[j] and alt == par.ref[j]:
            af[j] = 1.0 - a
    return af


def pgs_subset_moments(
    pgs_map, alt_af: np.ndarray
) -> dict[str, tuple[float, float, int, int]]:
    """Frequency-implied (mean, variance, n_with_af, n_scored) of each PGS's scored
    subset on this grid. Used to normalize partial-grid raw sums — a raw
    sum over a non-random SNP subset compared against full-genome
    reference moments picks up constant multi-sigma offsets (the same
    artifact genepred.scoring corrects for partial arrays)."""
    out = {}
    for pid, (idx, sgn, w) in pgs_map.items():
        p = np.where(sgn == 1, alt_af[idx], 1.0 - alt_af[idx])
        ok = ~np.isnan(p)
        pw, ww = p[ok], w[ok]
        out[pid] = (
            float((2.0 * ww * pw).sum()),
            float((2.0 * ww * ww * pw * (1.0 - pw)).sum()),
            int(ok.sum()),
            int(len(idx)),  # scored SNPs incl. those without AF
        )
    return out


def score_chrom(geno: np.ndarray, pgs_map) -> dict[str, float]:
    """ALT-dosage → effect-allele dosage = geno where sign=+1, 2−geno
    where sign=−1; raw chromosome score per PGS id."""
    out = {}
    for pid, (idx, sgn, w) in pgs_map.items():
        d = np.where(sgn == 1, geno[idx], 2 - geno[idx])
        out[pid] = float((w * d).sum())
    return out
