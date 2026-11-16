#!/usr/bin/env bash
# Person 1, Day-2 evening: fresh clone -> fresh venv -> run_all -> same sha256 as your copy?
#
#   bash scripts/check_reproducible.sh <git-url> <branch> <path-to-data-dir> TEAMNAME
#
# Uses MTAH_DATA_DIR so the data is never copied into the clone.
set -euo pipefail
URL="$1"; BRANCH="$2"; DATA="$(cd "$3" && pwd)"; TEAM="$4"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
echo "fresh clone in $TMP"
git clone --branch "$BRANCH" --depth 1 "$URL" "$TMP/repo"
cd "$TMP/repo"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
pip install -q -r requirements.txt
MTAH_DATA_DIR="$DATA" python run_all.py --team "$TEAM"
NEW="$(sha256sum "submissions/$TEAM.csv" | cut -d' ' -f1)"
OLD="$(sha256sum "$HERE/submissions/$TEAM.csv" | cut -d' ' -f1)"
echo "fresh clone: $NEW"
echo "your copy  : $OLD"
if [ "$NEW" = "$OLD" ]; then echo "IDENTICAL -- reproducible"; else echo "DIFFERENT -- investigate before submitting"; exit 1; fi
