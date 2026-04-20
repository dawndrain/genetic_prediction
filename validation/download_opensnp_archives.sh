#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/opensnp_archives
cd data/opensnp_archives

dl() {
  local name="$1" url="$2"
  if [[ -s "$name" ]]; then echo "$name: exists ($(du -h "$name"|cut -f1))"; return; fi
  echo "$name: $url"
  curl -L --retry 5 --retry-delay 15 -o "$name.part" "$url" && mv "$name.part" "$name"
  echo "$name: done ($(du -h "$name"|cut -f1))"
}

# Internet Archive — official 2017-12-08 datadump (~21.4 GB)
dl opensnp_datadump.2017-12-08.zip \
  "https://archive.org/download/opensnp_data_dumps/opensnp_datadump.2017-12-08.zip" &

# Zenodo records — fetch file URLs from the API so filenames are exact
for rec in 10715132 14963915; do
  curl -sL "https://zenodo.org/api/records/$rec" \
  | python3 -c "import sys,json
for f in json.load(sys.stdin).get('files',[]):
    print(f['key'], f['links']['self'])" \
  | while read -r fname url; do
      dl "zenodo_${rec}_${fname}" "$url" &
    done
done

wait
ls -la
du -sh .
