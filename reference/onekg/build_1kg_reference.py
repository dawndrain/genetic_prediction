"""Score every 1000 Genomes sample on every PGS in data/pgs_scoring_files/.

Strategy: for each PGS, build a (chrom, pos) -> (effect_allele, other_allele,
weight) map from the weight file, then stream each chromosome's VCF once,
look up matching sites, and accumulate per-sample dot products. This avoids
loading full genotypes into memory (2504 samples x 80M sites would be ~400GB).

Output:
  data/1kg_pgs_scores.tsv     long format: pgs_id, sample, pop, super_pop, score
  data/1kg_pgs_summary.tsv    per-(pgs_id, super_pop): mean, sd, n_snps_used

Positions are matched on GRCh37 (chr_position column in the weight files),
since 1KG Phase 3 is on GRCh37.
"""

import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

WEIGHTS_DIR = Path("data/pgs_scoring_files")
KG_DIR = Path("data/1kg")
PANEL = KG_DIR / "integrated_call_samples_v3.20130502.ALL.panel"


def load_panel():
    samples, pop, super_pop = [], {}, {}
    with open(PANEL) as f:
        f.readline()
        for line in f:
            p = line.rstrip().split("\t")
            samples.append(p[0])
            pop[p[0]] = p[1]
            super_pop[p[0]] = p[2]
    return samples, pop, super_pop


def _read_header(f):
    for line in f:
        if line.startswith("#"):
            continue
        return {c: i for i, c in enumerate(line.rstrip("\n").split("\t"))}
    raise ValueError("no header")


_RSID_POS: dict[str, tuple[str, int]] | None = None


def _rsid_to_pos():
    """rsID → (chrom, GRCh37 pos) from the HapMap3 snpinfo file, used as a
    fallback for PGS submissions that lack GRCh37 coordinates."""
    global _RSID_POS
    if _RSID_POS is None:
        _RSID_POS = {}
        with open("data/ld_reference/ldblk_1kg_eur/snpinfo_1kg_hm3") as f:
            f.readline()
            for line in f:
                r = line.split()
                _RSID_POS[r[1]] = (r[0], int(r[2]))
    return _RSID_POS


def load_weights(pgs_path):
    """Returns (by_chr, by_rsid). by_chr: chrom -> {pos: (ea,oa,w)} via the
    file's own GRCh37 positions if present, else via rsID→pos lookup from
    the HapMap3 snpinfo. by_rsid is kept for any leftover unmapped rsIDs
    (matched against the 1KG VCF ID column, which is usually empty in
    Phase 3 — so this path rarely contributes).
    """
    by_chr: dict[str, dict[int, tuple[str, str, float]]] = defaultdict(dict)
    by_rsid: dict[str, tuple[str, str, float]] = {}
    use_pos = True
    with gzip.open(pgs_path, "rt") as f:
        for line in f:
            if not line.startswith("#"):
                break
            if "genome_build" in line:
                build = line.split("=")[-1].strip()
                if build not in ("GRCh37", "hg19", "NCBI37"):
                    use_pos = False
        f.seek(0)
        cols = _read_header(f)
        i_chr = cols.get("chr_name")
        i_pos = cols.get("chr_position")
        i_rs = cols.get("hm_rsID", cols.get("rsID", cols.get("rsid")))
        i_ea = cols["effect_allele"]
        i_oa = cols.get("other_allele", cols.get("hm_inferOtherAllele", -1))
        i_w = cols["effect_weight"]
        if i_chr is None or i_pos is None:
            use_pos = False
        for line in f:
            if line.startswith("#"):
                continue
            row = line.rstrip("\n").split("\t")
            try:
                w = float(row[i_w])
            except (ValueError, IndexError):
                continue
            ea = row[i_ea].upper()
            oa = row[i_oa].upper() if 0 <= i_oa < len(row) else ""
            if use_pos:
                assert i_chr is not None and i_pos is not None
                try:
                    chrom = row[i_chr].lstrip("chr")
                    pos = int(row[i_pos])
                    by_chr[chrom][pos] = (ea, oa, w)
                    continue
                except (ValueError, IndexError):
                    pass
            if i_rs is not None and i_rs < len(row):
                rsid = row[i_rs]
                if rsid.startswith("rs"):
                    cp = _rsid_to_pos().get(rsid)
                    if cp is not None:
                        by_chr[cp[0]][cp[1]] = (ea, oa, w)
                    else:
                        by_rsid[rsid] = (ea, oa, w)
    return by_chr, by_rsid


def stream_chrom(vcf_path, want_by_pgs, want_rsid, samples, score_arrays, n_used):
    """Single pass over one chromosome VCF, updating all PGS scores in place.
    want_by_pgs: {pgs_id: {pos: (ea, oa, w)}} for this chromosome (GRCh37).
    want_rsid:   {pgs_id: {rsid: (ea, oa, w)}} for position-less scores.
    """
    pos_to_pgs: dict[int, list[tuple[str, str, str, float]]] = defaultdict(list)
    for pgs_id, posmap in want_by_pgs.items():
        for pos, (ea, oa, w) in posmap.items():
            pos_to_pgs[pos].append((pgs_id, ea, oa, w))
    rsid_to_pgs: dict[str, list[tuple[str, str, str, float]]] = defaultdict(list)
    for pgs_id, rsmap in want_rsid.items():
        for rsid, (ea, oa, w) in rsmap.items():
            rsid_to_pgs[rsid].append((pgs_id, ea, oa, w))
    if not pos_to_pgs and not rsid_to_pgs:
        return

    n_samples = len(samples)
    col_for: list[int] = []
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                vcf_samples = line.rstrip("\n").split("\t")[9:]
                idx = {s: i for i, s in enumerate(vcf_samples)}
                col_for = [9 + idx[s] for s in samples]
                continue
            tab1 = line.find("\t")
            tab2 = line.find("\t", tab1 + 1)
            try:
                pos = int(line[tab1 + 1 : tab2])
            except ValueError:
                continue
            hits = list(pos_to_pgs.get(pos, ()))
            if rsid_to_pgs:
                tab3 = line.find("\t", tab2 + 1)
                rs = line[tab2 + 1 : tab3]
                if rs in rsid_to_pgs:
                    hits = hits + rsid_to_pgs[rs]
            if not hits:
                continue
            row = line.rstrip("\n").split("\t")
            ref, alt = row[3].upper(), row[4].upper()
            gts = None
            for pgs_id, ea, oa, w in hits:
                if (ref, alt) == (oa, ea) or (oa == "" and alt == ea):
                    sign = 1
                elif (ref, alt) == (ea, oa) or (oa == "" and ref == ea):
                    sign = -1
                else:
                    continue
                if gts is None:
                    # Genotype field may be diploid ('0|1') or haploid ('1', male chrX).
                    gts = np.fromiter(
                        (
                            (g[0] == "1") + (g[-1] == "1")
                            for c in col_for
                            for g in (row[c],)
                        ),
                        dtype=np.int8,
                        count=n_samples,
                    )
                dosage = gts if sign == 1 else (2 - gts)
                score_arrays[pgs_id] += w * dosage
                n_used[pgs_id] += 1


def main():
    samples, pop, super_pop = load_panel()
    print(f"{len(samples)} samples in panel", file=sys.stderr)

    weight_files = sorted(WEIGHTS_DIR.glob("*_hmPOS_GRCh38.txt.gz"))
    all_pos = {}
    all_rsid = {}
    for wf in weight_files:
        pgs_id = wf.name.split("_")[0]
        by_chr, by_rs = load_weights(wf)
        all_pos[pgs_id] = by_chr
        all_rsid[pgs_id] = by_rs
        n_p = sum(len(v) for v in by_chr.values())
        n_r = len(by_rs)
        via = "pos" if n_p else ("rsid" if n_r else "EMPTY")
        print(
            f"  loaded {pgs_id}: {n_p:,} via pos, {n_r:,} via rsid → {via}",
            file=sys.stderr,
        )

    chr_files = {}
    for vcf in KG_DIR.glob("ALL.chr*.vcf.gz"):
        chrom = vcf.name.split(".")[1].replace("chr", "")
        chr_files[chrom] = vcf

    score_arrays = {pgs: np.zeros(len(samples)) for pgs in all_pos}
    n_used: dict[str, int] = defaultdict(int)

    try:
        for chrom in [str(c) for c in range(1, 23)] + ["X"]:
            if chrom not in chr_files:
                print(f"chr{chrom}: VCF missing, skipping", file=sys.stderr)
                continue
            want = {pgs: w.get(chrom, {}) for pgs, w in all_pos.items()}
            n_want = sum(len(v) for v in want.values())
            n_rs = sum(len(v) for v in all_rsid.values())
            print(
                f"chr{chrom}: {n_want:,} pos-target + {n_rs:,} rsid-target "
                f"across {len(all_pos)} scores",
                file=sys.stderr,
            )
            try:
                stream_chrom(
                    chr_files[chrom], want, all_rsid, samples, score_arrays, n_used
                )
            except Exception as e:
                print(
                    f"chr{chrom}: FAILED ({type(e).__name__}: {e}); "
                    f"continuing with partial scores",
                    file=sys.stderr,
                )
    finally:
        _write_outputs(samples, pop, super_pop, score_arrays, n_used)


def _write_outputs(samples, pop, super_pop, score_arrays, n_used):
    out_long = Path("data/1kg_pgs_scores.tsv")
    with open(out_long, "w") as f:
        f.write("pgs_id\tsample\tpop\tsuper_pop\tscore\n")
        for pgs_id, arr in score_arrays.items():
            for s, v in zip(samples, arr):
                f.write(f"{pgs_id}\t{s}\t{pop[s]}\t{super_pop[s]}\t{v:.6g}\n")
    print(f"wrote {out_long}", file=sys.stderr)

    out_sum = Path("data/1kg_pgs_summary.tsv")
    with open(out_sum, "w") as f:
        f.write("pgs_id\tsuper_pop\tn_samples\tn_snps\tmean\tsd\n")
        sp = np.array([super_pop[s] for s in samples])
        for pgs_id, arr in score_arrays.items():
            for grp in ["AFR", "AMR", "EAS", "EUR", "SAS", "ALL"]:
                mask = sp == grp if grp != "ALL" else np.ones(len(samples), bool)
                sub = arr[mask]
                f.write(
                    f"{pgs_id}\t{grp}\t{mask.sum()}\t{n_used[pgs_id]}\t"
                    f"{sub.mean():.6g}\t{sub.std(ddof=1):.6g}\n"
                )
    print(f"wrote {out_sum}", file=sys.stderr)


if __name__ == "__main__":
    main()
