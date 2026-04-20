#!/bin/bash
# 1000 Genomes Phase 3 integrated VCFs (GRCh37, ~2504 samples, ~15GB total).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/1kg
BASE="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"

curl -sL -o data/1kg/integrated_call_samples_v3.20130502.ALL.panel \
  "$BASE/integrated_call_samples_v3.20130502.ALL.panel"

for chr in $(seq 1 22) X; do
  f="ALL.chr${chr}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
  if [[ "$chr" == "X" ]]; then
    f="ALL.chrX.phase3_shapeit2_mvncall_integrated_v1c.20130502.genotypes.vcf.gz"
  fi
  out="data/1kg/$f"
  if [[ -s "$out" ]]; then
    echo "chr$chr: exists ($(du -h "$out" | cut -f1))"
    continue
  fi
  echo "chr$chr: downloading $f"
  curl -sL --retry 5 --retry-delay 10 -o "$out.part" "$BASE/$f" && mv "$out.part" "$out"
  curl -sL --retry 5 -o "$out.tbi" "$BASE/$f.tbi"
  echo "chr$chr: done ($(du -h "$out" | cut -f1))"
done
echo "total: $(du -sh data/1kg | cut -f1)"
