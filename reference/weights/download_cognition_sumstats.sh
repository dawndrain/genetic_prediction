#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/sumstats
cd data/sumstats

dl() {
  local name="$1" url="$2"
  if [[ -s "$name" ]]; then echo "$name: exists"; return; fi
  echo "$name: downloading"
  curl -sL --retry 5 --retry-delay 10 -o "$name.part" "$url" && mv "$name.part" "$name"
  echo "$name: $(du -h "$name" | cut -f1)"
}

# MANUAL STEP — EA3/EA4 are not on any open mirror (GWAS Catalog
# fullPvalueSet=False; OpenGWAS requires JWT since 2024-05; SSGAC S3 is
# click-through gated). Get one via browser:
#   https://thessgac.com/papers/ → Okbay 2022 → accept terms →
#   EA4_additive_excl_23andMe.txt.gz  (N≈766k)
# then drop it here as data/sumstats/ea4_excl23andMe.txt.gz and rerun
#   python mtag_lite.py --cog data/sumstats/savage2018_intelligence.tsv.gz \
#                       --ea  data/sumstats/ea4_excl23andMe.txt.gz
# Until then, hill2018_mtag_intelligence.tsv.gz is the published
# cognition×EA MTAG output and serves as the predictor.
if [[ ! -s ea4_excl23andMe.txt.gz ]]; then
  echo "NOTE: EA4 requires manual download from thessgac.com — see comment above" >&2
fi

# Savage et al. 2018 intelligence meta-analysis (N≈270k), GCST006250 harmonised.
dl savage2018_intelligence.tsv.gz \
  "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST006001-GCST007000/GCST006250/harmonised/29942086-GCST006250-EFO_0004337.h.tsv.gz"

# EA4 — Okbay 2022. SSGAC public release excludes 23andMe (N≈766k for the public file).
dl ea4_okbay2022_excl23andMe.txt.gz \
  "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90041001-GCST90042000/GCST90041880/GCST90041880_buildGRCh37.tsv.gz"

# Lee 2018 cognitive performance (EA3 paper supplement; UKB+COGENT, N≈258k).
# Davies 2018 isn't on the FTP — this is the same underlying cohort.
dl lee2018_cogperf.txt \
  "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST006001-GCST007000/GCST006572/GWAS_CP_all.txt"

# Hill 2018 MTAG-intelligence — already MTAG'd, so use as a comparison
# baseline rather than an MTAG input.
dl hill2018_mtag_intelligence.tsv.gz \
  "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST005001-GCST006000/GCST005316/GCST005316.tsv.gz"

ls -la
