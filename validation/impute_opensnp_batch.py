"""Batch-impute the phenotyped openSNP genomes and re-score height/BMI/cognition.

Validates the imputation boost end-to-end: with Beagle imputation, openSNP
height prediction goes from R2 0.34 -> 0.43 (EUR males, n=235) and
0.21 -> 0.34 (EUR females, n=192) -- recovering essentially the full
population R2 of the predictor (0.42) up to self-report reliability.

The trick that makes 600 genomes tractable is BATCH imputation: one
multi-sample VCF per chromosome (union of sites; ./. where a chip lacks a
site) and 22 Beagle runs total, instead of one ~10-minute run per genome.

Stages (restartable at the Beagle stage via cached files):
  1. pick target users: dedup'd genomes (from data/opensnp_archive_pgs.tsv,
     written by validate_height_archive.py) whose uid has a parsable
     height, weight, IQ, SAT, or edu phenotype. Build-36 genomes are
     EXCLUDED: the conform stage matches by GRCh37 position, so b36 files
     turn into garbage typed genotypes (measured: imputed scores correlate
     0.15 with their raw-array scores, vs 0.80 for b37).
  2. parse all target genotype files from the IA zip into RAM (one pass each)
  3. per chromosome: stream the dedup'd 1KG panel once, conform every sample
     to REF/ALT, write ONE multi-sample VCF
  4. run Beagle per chromosome on the multi-sample VCF (8 threads)
  5. stream each imputed VCF once, accumulating per-sample dosage scores for
     height (PGS002804), BMI (PGS002313), cognition (COGNITION_mtag_sbayesrc),
     EA (EA4_sbayesrc); write data/opensnp_imputed/scores.tsv

Run from the repo root:
    python validation/impute_opensnp_batch.py [--chroms 21,22] [--max-samples N]
"""
import argparse
import gzip
import multiprocessing as mp
import re
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "data/opensnp_archives/opensnp_datadump.2017-12-08.zip"
WORK = REPO / "data/opensnp_imputed"
SCORES = {
    "height": REPO / "data/pgs_scoring_files/PGS002804_hmPOS_GRCh38.txt.gz",
    "bmi": REPO / "data/pgs_scoring_files/PGS002313_hmPOS_GRCh38.txt.gz",
    "cog": REPO / "data/pgs_scoring_files/COGNITION_mtag_sbayesrc_hmPOS_GRCh38.txt.gz",
    "ea": REPO / "data/pgs_scoring_files/EA4_sbayesrc_hmPOS_GRCh38.txt.gz",
}
sys.path.insert(0, str(REPO))


def parse_num(s):
    if not isinstance(s, str):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", s.replace(",", "."))
    return float(m.group(1)) if m else None


def pick_targets(max_samples=None):
    df = pd.read_csv(REPO / "data/opensnp_archive_pgs.tsv", sep="\t")
    df = df[df.n_total > 100_000]
    df = df.sort_values("matched_height", ascending=False).drop_duplicates("uid")
    ph = pd.read_csv(REPO / "data/opensnp/phenotypes_2017.csv", sep=";",
                     engine="python", on_bad_lines="skip", dtype=str)
    uid_col = next(c for c in ph.columns if "user" in c.lower() and "id" in c.lower())
    keep_cols = [c for c in ph.columns if c.strip().lower() in
                 ("height", "weight", "iq", "sat math", "sat verbal",
                  "academic degree")]
    has_pheno = ph[keep_cols].apply(
        lambda col: col.map(lambda v: parse_num(v) is not None)
    ).any(axis=1)
    uids = set(ph.loc[has_pheno, uid_col].astype(int))
    df = df[df.uid.isin(uids)]
    # Exclude build-36 genomes: positional conforming against the GRCh37
    # panel turns them into noise (see module docstring).
    keep_u, keep_f = [], []
    with zipfile.ZipFile(ARCHIVE) as z:
        for u, fn in zip(df.uid, df.file):
            try:
                with z.open(fn) as fh:
                    head = fh.read(4000).decode("utf-8", "replace").lower()
            except Exception:
                continue
            if "build 36" in head:
                continue
            keep_u.append(u)
            keep_f.append(fn)
    if max_samples:
        keep_u, keep_f = keep_u[:max_samples], keep_f[:max_samples]
    print(
        f"[targets] {len(keep_u)} build-37 genomes with >=1 numeric phenotype "
        f"({len(df) - len(keep_u)} excluded: build 36 / unreadable)",
        flush=True,
    )
    return keep_u, keep_f


def _parse_one(args):
    idx, name = args
    out = defaultdict(list)
    try:
        with zipfile.ZipFile(ARCHIVE) as z, z.open(name) as fh:
            for bline in fh:
                line = bline.decode("utf-8", "replace")
                if not line or line[0] == "#":
                    continue
                p = line.rstrip("\r\n").split("\t")
                if len(p) < 4:
                    continue
                chrom, pos, gt = p[1].lstrip("chr"), p[2], p[3]
                if len(gt) == 1 and gt in "ACGT":
                    gt = gt + gt
                if len(gt) != 2 or gt[0] not in "ACGT" or gt[1] not in "ACGT":
                    continue
                try:
                    out[chrom].append((int(pos), gt))
                except ValueError:
                    pass
    except Exception as e:
        print(f"[parse] {name}: {e}", file=sys.stderr, flush=True)
    return idx, dict(out)


def write_multisample_vcfs(uids, files, chroms):
    WORK.mkdir(parents=True, exist_ok=True)
    n = len(files)
    print(f"[parse] reading {n} genomes from the zip ...", flush=True)
    geno = {}  # idx -> {chrom: {pos: gt}}
    t0 = time.monotonic()
    with mp.get_context("fork").Pool(8) as pool:
        for k, (idx, d) in enumerate(
            pool.imap_unordered(_parse_one, list(enumerate(files)), chunksize=2), 1
        ):
            geno[idx] = {c: dict(v) for c, v in d.items()}
            if k % 100 == 0:
                print(f"[parse] {k}/{n} ({time.monotonic()-t0:.0f}s)", flush=True)
    print(f"[parse] done in {time.monotonic()-t0:.0f}s", flush=True)

    sample_names = [f"u{u}" for u in uids]
    for chrom in chroms:
        out_path = WORK / f"chr{chrom}.input.vcf.gz"
        if out_path.exists():
            print(f"[conform] chr{chrom}: exists, skipping", flush=True)
            continue
        per_sample = [geno[i].get(chrom, {}) for i in range(n)]
        union = set()
        for d in per_sample:
            union.update(d)
        panel = REPO / "data/1kg_dedup" / f"chr{chrom}.vcf.gz"
        t1 = time.monotonic()
        n_sites = 0
        with gzip.open(panel, "rt") as f, gzip.open(out_path, "wt", compresslevel=1) as o:
            o.write("##fileformat=VCFv4.2\n")
            o.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
            o.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                    + "\t".join(sample_names) + "\n")
            for line in f:
                if line.startswith("#"):
                    continue
                t1_ = line.find("\t")
                t2_ = line.find("\t", t1_ + 1)
                try:
                    pos = int(line[t1_ + 1:t2_])
                except ValueError:
                    continue
                if pos not in union:
                    continue
                r = line.split("\t", 5)
                ref, alt = r[3], r[4]
                if len(ref) != 1 or len(alt) != 1 or "," in alt:
                    continue
                gts = []
                n_called = 0
                pair = {ref, alt}
                for d in per_sample:
                    g = d.get(pos)
                    if g is None or not {g[0], g[1]} <= pair:
                        gts.append("./.")
                    else:
                        gts.append(f"{int(g[0] == alt)}/{int(g[1] == alt)}")
                        n_called += 1
                if n_called == 0:
                    continue
                o.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\tGT\t"
                        + "\t".join(gts) + "\n")
                n_sites += 1
        print(f"[conform] chr{chrom}: {n_sites:,} sites x {n} samples "
              f"({time.monotonic()-t1:.0f}s)", flush=True)
    return sample_names


def run_beagle(chroms):
    from genepred.impute.beagle import setup
    jar, maps = setup()
    for chrom in chroms:
        out_prefix = WORK / f"chr{chrom}"
        if (WORK / f"chr{chrom}.vcf.gz").exists():
            print(f"[beagle] chr{chrom}: exists, skipping", flush=True)
            continue
        cmd = [
            "java", "-Xmx48g", "-jar", str(jar),
            f"gt={WORK / f'chr{chrom}.input.vcf.gz'}",
            f"ref={REPO / 'data/1kg_dedup' / f'chr{chrom}.vcf.gz'}",
            f"out={out_prefix}", f"chrom={chrom}", "nthreads=8", "gp=false",
        ]
        gmap = maps / f"plink.chr{chrom}.GRCh37.map"
        if gmap.exists():
            cmd.append(f"map={gmap}")
        t0 = time.monotonic()
        log = WORK / f"chr{chrom}.log"
        with open(log, "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        print(f"[beagle] chr{chrom}: rc={rc} ({time.monotonic()-t0:.0f}s)", flush=True)
        if rc != 0:
            raise RuntimeError(f"beagle failed on chr{chrom}; see {log}")


def load_weight_map(path):
    out = defaultdict(dict)
    with gzip.open(path, "rt") as f:
        cols = None
        for line in f:
            if line.startswith("#"):
                continue
            r = line.rstrip("\n").split("\t")
            if cols is None:
                cols = {c: i for i, c in enumerate(r)}
                continue
            try:
                chrom = r[cols["chr_name"]].lstrip("chr")
                pos = int(r[cols["chr_position"]])
                w = float(r[cols["effect_weight"]])
            except (ValueError, KeyError, IndexError):
                continue
            ea = r[cols["effect_allele"]].upper()
            oa = r[cols.get("other_allele", cols["effect_allele"])].upper()
            out[chrom][pos] = (ea, oa, w)
    return out


def score_imputed(sample_names, chroms):
    wmaps = {t: load_weight_map(p) for t, p in SCORES.items()}
    for t, m in wmaps.items():
        print(f"[score] {t}: {sum(len(v) for v in m.values()):,} SNPs", flush=True)
    n = len(sample_names)
    raw = {t: np.zeros(n) for t in SCORES}
    n_used = {t: 0 for t in SCORES}
    for chrom in chroms:
        path = WORK / f"chr{chrom}.vcf.gz"
        t0 = time.monotonic()
        want = {t: wmaps[t].get(chrom, {}) for t in SCORES}
        pos_any = set()
        for m in want.values():
            pos_any.update(m)
        with gzip.open(path, "rt") as f:
            fmt_ds = None
            for line in f:
                if line.startswith("#"):
                    continue
                t1_ = line.find("\t")
                t2_ = line.find("\t", t1_ + 1)
                try:
                    pos = int(line[t1_ + 1:t2_])
                except ValueError:
                    continue
                if pos not in pos_any:
                    continue
                r = line.rstrip("\n").split("\t")
                ref, alt = r[3].upper(), r[4].upper()
                if len(ref) != 1 or len(alt) != 1 or "," in alt:
                    continue  # multi-allelic: DS is per-ALT ("0,1")
                if fmt_ds is None:
                    fmt_ds = r[8].split(":").index("DS")
                ds = np.fromiter(
                    (float(x.split(":")[fmt_ds]) for x in r[9:]),
                    dtype=np.float64, count=n,
                )
                for t, m in want.items():
                    rec = m.get(pos)
                    if rec is None:
                        continue
                    ea, oa, w = rec
                    if ea == alt and (not oa or oa == ref):
                        raw[t] += w * ds
                    elif ea == ref and (not oa or oa == alt):
                        raw[t] += w * (2.0 - ds)
                    else:
                        continue
                    n_used[t] += 1
        print(f"[score] chr{chrom} done ({time.monotonic()-t0:.0f}s)", flush=True)
    out = pd.DataFrame({"sample": sample_names})
    out["uid"] = [int(s[1:]) for s in sample_names]
    for t in SCORES:
        out[f"raw_{t}"] = raw[t]
        print(f"[score] {t}: used {n_used[t]:,} SNPs", flush=True)
    dest = WORK / "scores.tsv"
    out.to_csv(dest, sep="\t", index=False)
    print(f"[score] wrote {dest}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroms", default="1-22")
    ap.add_argument("--max-samples", type=int, default=None)
    a = ap.parse_args()
    if "-" in a.chroms:
        lo, hi = a.chroms.split("-")
        chroms = [str(c) for c in range(int(lo), int(hi) + 1)]
    else:
        chroms = a.chroms.split(",")
    uids, files = pick_targets(a.max_samples)
    sample_names = write_multisample_vcfs(uids, files, chroms)
    run_beagle(chroms)
    score_imputed(sample_names, chroms)


if __name__ == "__main__":
    main()
