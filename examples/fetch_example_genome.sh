#!/bin/bash
# Download a real public-domain 23andMe genotype file for testing.
# Manu Sporny released his 23andMe v2 raw data into the public domain
# in 2011 (github.com/msporny/dna). ~966k SNPs, GRCh36 positions —
# the rsID-matching path in genepred.io handles the build mismatch.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
out="data/example_genome.txt"
if [[ -s "$out" ]]; then
  echo "already have $out ($(wc -l < "$out") lines)"; exit 0
fi
url="https://raw.githubusercontent.com/msporny/dna/master/ManuSporny-genome.txt"
echo "fetching $url"
curl -fL --retry 3 -o "$out" "$url"
echo "wrote $out ($(wc -l < "$out") lines, $(du -h "$out" | cut -f1))"
echo ""
echo "Try:  genepred score $out"
echo "      genepred report $out"
