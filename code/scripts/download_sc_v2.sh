#!/usr/bin/env bash
# Download + unpack Speech Commands V2 (2.3 GB) into ../data/speech_commands_v2
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${HERE}/../../data/speech_commands_v2"
mkdir -p "${DEST}"
cd "${DEST}"

if [[ -f "validation_list.txt" && -f "testing_list.txt" ]]; then
  echo "SC V2 already unpacked at ${DEST} — skipping."
  exit 0
fi

URL="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
echo "Downloading ${URL}"
curl -L --retry 3 -o speech_commands_v0.02.tar.gz "${URL}"
echo "Extracting..."
tar -xzf speech_commands_v0.02.tar.gz
rm speech_commands_v0.02.tar.gz

echo "Done → $(pwd)"
du -sh .
