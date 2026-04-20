#!/bin/bash
# v2 cognition PGS: Savage 2018 × EA4 → MTAG → SBayesRC (replaces LDpred-inf).
#
# Reuses the GCTB 2.5.5 binary, ukbEUR_HM3 LD eigen-decomposition, and
# baseline-LF v2.2 annotations already present under
#   tools/{gctb_2.5.5_Linux,sbayesrc/}
# from the height-SBayesRC repro. If those are gone, fetch:
#   gctb:        https://cnsgenomics.com/software/gctb/download/gctb_2.5.5_Linux.zip
#   ldm-eigen:   https://cnsgenomics.com/software/gctb/download/ukbEUR_HM3.zip   (~3 GB)
#   annotations: https://cnsgenomics.com/software/gctb/download/annot_baseline2.2.zip
#
# Inputs assumed present (produced by mtag_lite.py):
#   data/sumstats/cognition_mtag.tsv.gz
#   data/sumstats/cognition_mtag_prscs.n_eff   (≈402000)

set -euo pipefail
cd "$(dirname "$0")"
[[ -e data ]] || ln -s ../../data data

GCTB=${GCTB:-../../tools/gctb_2.5.5_Linux/gctb}
LDM=${SBAYESRC_LDM:-../../tools/sbayesrc/ukbEUR_HM3/ukbEUR_HM3}
ANNOT=${SBAYESRC_ANNOT:-../../tools/sbayesrc/annot/annot_baseline2.2.txt}
THREADS=${THREADS:-32}

MTAG=data/sumstats/cognition_mtag.tsv.gz
NEFF=$(cat data/sumstats/cognition_mtag_prscs.n_eff 2>/dev/null || echo 402000)
MA=data/sumstats/cognition_mtag.ma
IMPMA=data/sumstats/cognition_mtag.imputed.ma
OUTSTEM=data/sbayesrc/cognition_v2
WEIGHTS=data/pgs_scoring_files/COGNITION_mtag_sbayesrc_hmPOS_GRCh38.txt.gz

mkdir -p data/sbayesrc

# 1. MTAG sumstats → COJO .ma  (SNP A1 A2 freq b se p N)
#    freq pulled from the LD reference's snp.info, aligned to MTAG's A1.
if [[ ! -s $MA ]]; then
  echo "[1/4] writing $MA"
  python - "$MTAG" "$LDM/snp.info" "$NEFF" "$MA" <<'PY'
import gzip, sys, math
mtag, snpinfo, neff, out = sys.argv[1:5]
neff = int(float(neff))
ref = {}
with open(snpinfo) as f:
    next(f)
    for ln in f:
        r = ln.split()
        ref[r[1]] = (r[5].upper(), r[6].upper(), float(r[7]))  # A1,A2,A1Freq
n_kept = 0
with gzip.open(mtag, "rt") as f, open(out, "w") as o:
    o.write("SNP\tA1\tA2\tfreq\tb\tse\tp\tN\n")
    hdr = next(f).rstrip().split("\t")
    ix = {c: i for i, c in enumerate(hdr)}
    for ln in f:
        r = ln.rstrip().split("\t")
        sid = r[ix["snp"]]
        if sid not in ref:
            continue
        a1, a2 = r[ix["effect_allele"]].upper(), r[ix["other_allele"]].upper()
        ra1, ra2, rf = ref[sid]
        if (a1, a2) == (ra1, ra2):
            freq = rf
        elif (a1, a2) == (ra2, ra1):
            freq = 1.0 - rf
        else:
            continue
        b = float(r[ix["beta_mtag"]]); se = float(r[ix["se_mtag"]])
        if se <= 0 or not math.isfinite(b):
            continue
        z = b / se
        p = math.erfc(abs(z) / 2**0.5)
        o.write(f"{sid}\t{a1}\t{a2}\t{freq:.6f}\t{b:.6e}\t{se:.6e}\t{p:.4e}\t{neff}\n")
        n_kept += 1
print(f"  {n_kept:,} SNPs in MTAG ∩ ukbEUR_HM3 → {out}", file=sys.stderr)
PY
fi

# 2. Impute summary stats for LD-ref SNPs missing from MTAG
#    (SBayesRC requires every LD-block SNP to have a row).
if [[ ! -s $IMPMA ]]; then
  echo "[2/4] gctb --impute-summary"
  "$GCTB" --ldm-eigen "$LDM" --gwas-summary "$MA" --impute-summary \
          --thread "$THREADS" --out "${IMPMA%.imputed.ma}" \
    2>&1 | tee data/sbayesrc/cognition_impute.log
fi

# 3. SBayesRC
if [[ ! -s $OUTSTEM.snpRes ]]; then
  echo "[3/4] gctb --sbayes RC  (≈1h on 32 threads)"
  "$GCTB" --sbayes RC --ldm-eigen "$LDM" --gwas-summary "$IMPMA" \
          --annot "$ANNOT" --chain-length 3000 --burn-in 1000 \
          --thread "$THREADS" --seed 1 --out "$OUTSTEM" \
    2>&1 | tee data/sbayesrc/cognition_sbayesrc.log
fi

# 4. .snpRes → PGS-Catalog hmPOS format (matches v1's columns; positions
#    are from the ukbEUR_HM3 reference, i.e. GRCh37, same as v1).
echo "[4/4] writing $WEIGHTS"
python - "$OUTSTEM.snpRes" "$WEIGHTS" <<'PY'
import gzip, sys
src, dst = sys.argv[1:3]
with open(src) as f, gzip.open(dst, "wt") as o:
    hdr = next(f).split()
    ix = {c: i for i, c in enumerate(hdr)}
    o.write("rsID\tchr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\n")
    n = 0
    for ln in f:
        r = ln.split()
        o.write(f"{r[ix['Name']]}\t{r[ix['Chrom']]}\t{r[ix['Position']]}\t"
                f"{r[ix['A1']]}\t{r[ix['A2']]}\t{r[ix['A1Effect']]}\n")
        n += 1
print(f"  wrote {n:,} weights", file=sys.stderr)
PY

echo "done. v2 weights: $WEIGHTS"
echo "validate: add to WEIGHT_FILES in validation/validate_height_archive.py and re-run"
