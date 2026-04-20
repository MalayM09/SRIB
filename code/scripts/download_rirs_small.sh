#!/usr/bin/env bash
# Pull just the small-room simulated RIRs from OpenSLR-28. Final size ≈ 400 MB.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${HERE}/../../data/rirs_small"
mkdir -p "${DEST}"
cd "${DEST}"

if [[ -d "RIRS_NOISES/simulated_rirs/smallroom" ]]; then
  echo "RIRs subset already present at ${DEST} — skipping."
  exit 0
fi

URL="https://www.openslr.org/resources/28/rirs_noises.zip"
echo "Downloading ${URL}"
curl -L --retry 3 -o rirs_noises.zip "${URL}"
echo "Extracting smallroom subset..."
unzip -q rirs_noises.zip 'RIRS_NOISES/simulated_rirs/smallroom/*'
rm rirs_noises.zip

echo "Done → $(pwd)"
du -sh .
