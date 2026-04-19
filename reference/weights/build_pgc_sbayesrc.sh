#!/bin/bash
# Build SBayesRC PGS weights from PGC consortium sumstats for psychiatric
# traits whose best public scores aren't deposited in PGS Catalog.
#
#   ./build_pgc_sbayesrc.sh <trait>
# where <trait> ∈ {depression, schizophrenia, bipolar, adhd, genetic_g}
#
# Reuses the same GCTB / ukbEUR_HM3 LD eigen-ref / annotation files as
# build_cognition_v2_sbayesrc.sh.
#
# PGC sumstat downloads require accepting a click-through agreement at
# https://figshare.com/projects/Psychiatric_Genomics_Consortium/27530 — this
# script does NOT fetch them. Drop the gz file at the path under SUMSTATS_GZ
# below and rerun.
#
# LICENSE RESTRICTION: PGC terms prohibit using these data "to develop any
# type of risk or predictive test for an unborn individual." Outputs for the
# four PGC traits below MUST NOT route to the embryo-selection demo.
# genetic_g (de la Fuente, Edinburgh DataShare) is under separate terms.

set -euo pipefail
cd "$(dirname "$0")"
[[ -e data ]] || ln -s ../../data data

GCTB=${GCTB:-../../tools/gctb_2.5.5_Linux/gctb}
LDM=${SBAYESRC_LDM:-../../tools/sbayesrc/ukbEUR_HM3/ukbEUR_HM3}
ANNOT=${SBAYESRC_ANNOT:-../../tools/sbayesrc/annot/annot_baseline2.2.txt}
THREADS=${THREADS:-32}

TRAIT=${1:?usage: $0 <depression|schizophrenia|bipolar|adhd>}

# Per-trait config: sumstat file, column names (SNP/A1/A2/BETA-or-OR/SE/P/N),
# whether the effect column is an odds ratio, and effective N for case/control
# (4/(1/Ncase+1/Nctrl)) when no per-SNP N column.
case "$TRAIT" in
  depression)
    SUMSTATS_GZ=data/sumstats/pgc-mdd2025-eur.tsv.gz
    COLS="ID,A1,A2,BETA,SE,PVAL,"; IS_OR=0; NEFF=1138515
    OUTNAME=DEPRESSION_pgc2025_sbayesrc ;;
  schizophrenia)
    SUMSTATS_GZ=data/sumstats/PGC3_SCZ_wave3.european.autosome.public.v3.vcf.tsv.gz
    COLS="ID,A1,A2,BETA,SE,PVAL,NEFF"; IS_OR=0; NEFF=
    OUTNAME=SCZ_pgc3_sbayesrc ;;
  bipolar)
    SUMSTATS_GZ=data/sumstats/pgc-bip2021-all.vcf.tsv.gz
    COLS="ID,A1,A2,BETA,SE,PVAL,"; IS_OR=0; NEFF=101962
    OUTNAME=BIPOLAR_pgc2021_sbayesrc ;;
  adhd)
    SUMSTATS_GZ=data/sumstats/ADHD2022_iPSYCH_deCODE_PGC.meta.gz
    COLS="SNP,A1,A2,OR,SE,P,"; IS_OR=1; NEFF=103135
    OUTNAME=ADHD_pgc2022_sbayesrc ;;
  genetic_g)
    # de la Fuente et al. 2021 Nat Hum Behav, genomic-SEM g across 7 UKB tests.
    # https://datashare.ed.ac.uk/handle/10283/3756 (public, no DUA)
    SUMSTATS_GZ=data/sumstats/delaFuente2021_genetic_g.tsv.gz
    COLS="SNP,A1,A2,Estimate,SE,Pval_Estimate,"; IS_OR=0; NEFF=282014
    OUTNAME=COGNITION_geneticg_sbayesrc ;;
  ea4)
    # Okbay et al. 2022 Nat Genet, EA4 excl. 23andMe (SSGAC public release).
    SUMSTATS_GZ=data/sumstats/ea4_excl23andMe.txt.gz
    COLS="rsID,Effect_allele,Other_allele,Beta,SE,P,"; IS_OR=0; NEFF=765283
    OUTNAME=COGNITION_ea4_sbayesrc ;;
  savage_iq)
    # Savage et al. 2018 Nat Genet, intelligence meta-analysis.
    SUMSTATS_GZ=data/sumstats/savage2018_intelligence.tsv.gz
    COLS="hm_rsid,hm_effect_allele,hm_other_allele,beta,standard_error,p_value,n_analyzed"
    IS_OR=0; NEFF=
    OUTNAME=COGNITION_savageiq_sbayesrc ;;
  *) echo "unknown trait '$TRAIT'"; exit 1 ;;
esac

if [[ ! -s $SUMSTATS_GZ ]]; then
  echo "missing $SUMSTATS_GZ — download from PGC portal (click-through DUA) and rerun"
  exit 1
fi

MA=data/sumstats/${TRAIT}_pgc.ma
IMPMA=data/sumstats/${TRAIT}_pgc.imputed.ma
OUTSTEM=data/sbayesrc/${TRAIT}_pgc
WEIGHTS=data/pgs_scoring_files/${OUTNAME}_hmPOS_GRCh38.txt.gz
mkdir -p data/sbayesrc

# 1. PGC sumstats → COJO .ma, freq from ukbEUR_HM3 snp.info
if [[ ! -s $MA ]]; then
  echo "[1/4] writing $MA"
  python - "$SUMSTATS_GZ" "$LDM/snp.info" "$COLS" "$IS_OR" "${NEFF:-0}" "$MA" <<'PY'
import gzip, sys, math
src, snpinfo, cols, is_or, neff_default, out = sys.argv[1:7]
c_snp, c_a1, c_a2, c_b, c_se, c_p, c_n = cols.split(",")
is_or = int(is_or); neff_default = int(neff_default)
ref = {}
with open(snpinfo) as f:
    next(f)
    for ln in f:
        r = ln.split()
        ref[r[1]] = (r[5].upper(), r[6].upper(), float(r[7]))
opener = gzip.open if src.endswith(".gz") else open
n_kept = 0
with opener(src, "rt") as f, open(out, "w") as o:
    o.write("SNP\tA1\tA2\tfreq\tb\tse\tp\tN\n")
    hdr = next(f).rstrip().split()
    ix = {c: i for i, c in enumerate(hdr)}
    for ln in f:
        r = ln.rstrip().split()
        sid = r[ix[c_snp]]
        if sid not in ref:
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
        except (ValueError, KeyError):
            continue
        if is_or:
            b = math.log(b)
        if se <= 0 or not math.isfinite(b):
            continue
        n = int(float(r[ix[c_n]])) if c_n and c_n in ix else neff_default
        o.write(f"{sid}\t{a1}\t{a2}\t{freq:.6f}\t{b:.6e}\t{se:.6e}\t{p:.4e}\t{n}\n")
        n_kept += 1
print(f"  {n_kept:,} SNPs matched → {out}", file=sys.stderr)
PY
fi

# 2. Impute missing LD-ref SNPs
if [[ ! -s $IMPMA ]]; then
  echo "[2/4] gctb --impute-summary"
  "$GCTB" --ldm-eigen "$LDM" --gwas-summary "$MA" --impute-summary \
          --thread "$THREADS" --out "${IMPMA%.imputed.ma}" \
    2>&1 | tee "data/sbayesrc/${TRAIT}_impute.log"
fi

# 3. SBayesRC
if [[ ! -s $OUTSTEM.snpRes ]]; then
  echo "[3/4] gctb --sbayes RC"
  "$GCTB" --sbayes RC --ldm-eigen "$LDM" --gwas-summary "$IMPMA" \
          --annot "$ANNOT" --chain-length 3000 --burn-in 1000 \
          --thread "$THREADS" --seed 1 --out "$OUTSTEM" \
    2>&1 | tee "data/sbayesrc/${TRAIT}_sbayesrc.log"
fi

# 4. .snpRes → PGS-Catalog hmPOS format
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
print(f"  wrote {n:,} weights → {dst}", file=sys.stderr)
PY

echo "done: $WEIGHTS"
