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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from genepred.paths import kg_dir, pgs_weights_dir

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
