#!/bin/bash
# Build SBayesRC PGS weights from open-access (no DUA) consortium sumstats,
# for traits where the best public score isn't in PGS Catalog. Unlike
# build_pgc_sbayesrc.sh, these sumstats are auto-downloaded (CC-BY or
# similar) and the resulting weights ARE redistributable.
#
#   ./build_open_sbayesrc.sh <trait>
#   ./build_open_sbayesrc.sh all   # runs every trait sequentially
#
# Each trait: ~5 min download + ~30–50 min SBayesRC on 32 threads.
set -euo pipefail
cd "$(dirname "$0")"
[[ -e data ]] || ln -s ../../data data

GCTB=${GCTB:-../../tools/gctb_2.5.5_Linux/gctb}
LDM=${SBAYESRC_LDM:-../../tools/sbayesrc/ukbEUR_HM3/ukbEUR_HM3}
ANNOT=${SBAYESRC_ANNOT:-../../tools/sbayesrc/annot/annot_baseline2.2.txt}
THREADS=${THREADS:-32}
GC=https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics

[[ -x $GCTB ]] || { echo "GCTB not found at $GCTB"; exit 1; }

TRAIT=${1:?usage: $0 <stroke|ckd|bmd|af|asthma|all>}

if [[ $TRAIT == all ]]; then
  SELF="$PWD/$(basename "$0")"
  for t in stroke bmd af asthma; do
    echo ""; echo "=========== $t ==========="
    "$SELF" "$t" || echo "!! $t failed, continuing"
  done
  exit 0
fi

# Per-trait config: GWAS Catalog accession (filename auto-discovered),
# column names (SNP/A1/A2/BETA-or-OR/SE/P/N), OR flag, fallback Neff.
# All harmonised files share the hm_* column convention.
COLS="hm_rsid,hm_effect_allele,hm_other_allele,hm_beta,standard_error,p_value,"
IS_OR=0
case "$TRAIT" in
  stroke)
    # Mishra et al. 2022 Nature (GIGASTROKE), any stroke, EUR.
    # Newer GWAS-SSF format (no hm_ prefix on columns).
    GCST=GCST90104539; RANGE=GCST90104001-GCST90105000; NEFF=238816
    COLS="rsid,effect_allele,other_allele,beta,standard_error,p_value,"
    OUTNAME=STROKE_gigastroke_sbayesrc ;;
  bmd)
    # Morris et al. 2019 Nat Genet, heel eBMD (UKB), proxy for osteoporosis.
    GCST=GCST006979; RANGE=GCST006001-GCST007000; NEFF=426824
    COLS="hm_rsid,hm_effect_allele,hm_other_allele,hm_beta,standard_error,p_value,n"
    OUTNAME=BMD_morris_sbayesrc ;;
  af)
    # Nielsen et al. 2018 Nat Genet (HUNT/AFGen). Roselli 2025 (180k cases)
    # is better; substitute its GCST here once confirmed.
    GCST=GCST006414; RANGE=GCST006001-GCST007000; NEFF=132330
    OUTNAME=AF_nielsen_sbayesrc ;;
  asthma)
    # Valette et al. 2021 Commun Biol, UKB asthma.
    GCST=GCST90014325; RANGE=GCST90014001-GCST90015000; NEFF=137436
    OUTNAME=ASTHMA_valette_sbayesrc ;;
  ckd)
    echo "CKD: CKDGen (Wuttke 2019) has no harmonised file in GWAS Catalog;"
    echo "  use raw sumstats from ckdgen.imbi.uni-freiburg.de manually."
    exit 1 ;;
  *) echo "unknown trait '$TRAIT'"; exit 1 ;;
esac

# Discover the harmonised filename — GWAS Catalog renamed these from
# {PMID}-{GCST}-{EFO}.h.tsv.gz to {GCST}.h.tsv.gz at some point.
DIR="$GC/$RANGE/$GCST/harmonised"
FNAME=$(curl -sL "$DIR/" | grep -oE 'href="[^"]*\.h\.tsv\.gz"' | head -1 | sed 's/href="//; s/"$//')
[[ -n "$FNAME" ]] || { echo "no harmonised file for $GCST"; exit 1; }
URL="$DIR/$FNAME"

SUMSTATS=data/sumstats/${TRAIT}_open.h.tsv.gz
MA=data/sumstats/${TRAIT}_open.ma
IMPMA=data/sumstats/${TRAIT}_open.imputed.ma
OUTSTEM=data/sbayesrc/${TRAIT}_open
WEIGHTS=data/pgs_scoring_files/${OUTNAME}_hmPOS_GRCh38.txt.gz
mkdir -p data/sbayesrc data/sumstats

# 0. Download (open access; no click-through)
if [[ ! -s $SUMSTATS || $(stat -c%s "$SUMSTATS") -lt 1000000 ]]; then
  echo "[0/4] downloading $URL"
  curl -fL --retry 5 -o "$SUMSTATS" "$URL"
fi
echo "  $(du -h "$SUMSTATS" | cut -f1) — head: $(zcat "$SUMSTATS" | head -1 | cut -c-120)"

# 1. → COJO .ma, freq from ukbEUR_HM3 snp.info
if [[ ! -s $MA ]]; then
  echo "[1/4] writing $MA"
  python - "$SUMSTATS" "$LDM/snp.info" "$COLS" "$IS_OR" "${NEFF:-0}" "$MA" <<'PY'
import gzip, math, sys
src, snpinfo, cols, is_or, neff_default, out = sys.argv[1:7]
c_snp, c_a1, c_a2, c_b, c_se, c_p, c_n = cols.split(",")
is_or = int(is_or); neff_default = int(neff_default)
ref = {}
with open(snpinfo) as f:
    next(f)
    for ln in f:
        r = ln.split()
        ref[r[1]] = (r[5].upper(), r[6].upper(), float(r[7]))
n_kept = 0
with gzip.open(src, "rt") as f, open(out, "w") as o:
    o.write("SNP\tA1\tA2\tfreq\tb\tse\tp\tN\n")
    hdr = next(f).rstrip().split("\t")
    ix = {c: i for i, c in enumerate(hdr)}
    for ln in f:
        r = ln.rstrip().split("\t")
        sid = r[ix[c_snp]] if c_snp in ix else ""
        if not sid or sid not in ref:
            continue
        a1, a2 = r[ix[c_a1]].upper(), r[ix[c_a2]].upper()
        ra1, ra2, rf = ref[sid]
        if (a1, a2) == (ra1, ra2):
            freq = rf
        elif (a1, a2) == (ra2, ra1):
            freq = 1.0 - rf
        else:
            continue
        try:
            b = float(r[ix[c_b]]); se = float(r[ix[c_se]]); p = float(r[ix[c_p]])
        except (ValueError, KeyError, IndexError):
            continue
        if is_or:
            b = math.log(b)
        if se <= 0 or not math.isfinite(b) or not math.isfinite(se):
            continue
        n = neff_default
        if c_n and c_n in ix:
            try:
                n = int(float(r[ix[c_n]]))
            except (ValueError, IndexError):
                pass
        o.write(f"{sid}\t{a1}\t{a2}\t{freq:.6f}\t{b:.6e}\t{se:.6e}\t{p:.4e}\t{n}\n")
        n_kept += 1
print(f"  {n_kept:,} SNPs matched → {out}", file=sys.stderr)
if n_kept < 100_000:
    print("  ! <100k SNPs matched — check column names / build", file=sys.stderr)
    sys.exit(1)
PY
fi

# 2. Impute missing LD-ref SNPs
if [[ ! -s $IMPMA ]]; then
  echo "[2/4] gctb --impute-summary"
  "$GCTB" --ldm-eigen "$LDM" --gwas-summary "$MA" --impute-summary \
          --thread "$THREADS" --out "${IMPMA%.imputed.ma}" \
    2>&1 | tee "data/sbayesrc/${TRAIT}_impute.log" | tail -5
fi

# 3. SBayesRC
if [[ ! -s $OUTSTEM.snpRes ]]; then
  echo "[3/4] gctb --sbayes RC (~30–50 min)"
  "$GCTB" --sbayes RC --ldm-eigen "$LDM" --gwas-summary "$IMPMA" \
          --annot "$ANNOT" --chain-length 3000 --burn-in 1000 \
          --thread "$THREADS" --seed 1 --out "$OUTSTEM" \
    2>&1 | tee "data/sbayesrc/${TRAIT}_sbayesrc.log" | grep -E "iter|hsq|done|Error" || true
fi

# 4. .snpRes → PGS-Catalog hmPOS format
echo "[4/4] writing $WEIGHTS"
python - "$OUTSTEM.snpRes" "$WEIGHTS" "$OUTNAME" "$URL" <<'PY'
import gzip, sys
src, dst, name, url = sys.argv[1:5]
with open(src) as f, gzip.open(dst, "wt") as o:
    hdr = next(f).split()
    ix = {c: i for i, c in enumerate(hdr)}
    o.write(f"#pgs_id={name}\n#genome_build=GRCh37\n#weight_type=SBayesRC\n")
    o.write(f"#source_gwas={url}\n")
    o.write("rsID\tchr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\n")
    n = 0
    for ln in f:
        r = ln.split()
        o.write(f"{r[ix['Name']]}\t{r[ix['Chrom']]}\t{r[ix['Position']]}\t"
                f"{r[ix['A1']]}\t{r[ix['A2']]}\t{r[ix['A1Effect']]}\n")
        n += 1
print(f"  wrote {n:,} weights → {dst}", file=sys.stderr)
PY

echo "done: $WEIGHTS"
