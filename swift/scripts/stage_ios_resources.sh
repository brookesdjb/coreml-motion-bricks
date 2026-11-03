#!/bin/sh
# Copy the batch-1 model and the exported feeds into Apps/iOS/Resources (git-ignored) for bundling.
#   scripts/stage_ios_resources.sh [path/to/soma_b1.mlpackage] [path/to/feeds dir]
set -e
here="$(cd "$(dirname "$0")/.." && pwd)"
model="${1:-$here/../build/coreml/soma_b1.mlpackage}"
feeds="${2:-$here/Feeds}"
dest="$here/Apps/iOS/Resources"
mkdir -p "$dest"
rm -rf "$dest/soma_b1.mlpackage" "$dest/Feeds"
cp -R "$model" "$dest/soma_b1.mlpackage"
cp -R "$feeds" "$dest/Feeds"
echo "staged $(basename "$model") and $(ls "$dest/Feeds" | wc -l | tr -d ' ') feed files in $dest"
