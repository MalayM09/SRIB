#!/usr/bin/env bash
# Pull MUSAN, keep all of noise/ plus ~60 music clips, delete the rest.
# Final size ≈ 1 GB.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${HERE}/../../data/musan_small"
mkdir -p "${DEST}"
cd "${DEST}"

if [[ -d "noise" && -d "music" ]]; then
  echo "MUSAN subset already present at ${DEST} — skipping."
  exit 0
fi

URL="https://www.openslr.org/resources/17/musan.tar.gz"
echo "Downloading ${URL} (~11 GB temporary)"
curl -L --retry 3 -o musan.tar.gz "${URL}"

echo "Extracting noise/ (kept in full)..."
tar -xzf musan.tar.gz --strip-components=1 musan/noise

echo "Extracting music/ (first 60 wavs only)..."
mkdir -p music_tmp
tar -xzf musan.tar.gz -C music_tmp
mv music_tmp/musan/music ./music
rm -rf music_tmp
find music -type f -name '*.wav' | sort | tail -n +61 | xargs rm -f || true

rm -f musan.tar.gz
echo "Done → $(pwd)"
du -sh .
