#!/bin/sh
# run.sh: measure one upgrade before and after its patch, free and offline, and print one JSON line
#   {"metric": ..., "before": ..., "after": ..., "pass": true|false, "extra": {...}}
# Usage (from an upgrade's test.sh): sh run.sh <upgrade dir> <metric key> '<pass rule in python over b, a>'
# The tree it measures is $P4_ROOT (the notebook's scratch copy) or this file's worktree. "after" is that tree's
# plugin as it stands (the patch applied by the caller); "before" is a copy with patch.diff reversed. Both migrate
# AudioPlugin-p4 and ap-p4_mac from the old-layout clones under $P4_FIXTURES with their own migrate.py, so a patch
# to the migration is measured too. Exit 0 either way; a failure to measure prints pass false with the error.
set -u
UDIR=$(cd "$(dirname "$1")/$(basename "$1")" && pwd)
METRIC=$2
RULE=$3
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${P4_ROOT:-$(cd "$HERE/../../.." && pwd)}
FIX=${P4_FIXTURES:-/Users/ez/gritz/p4/.claude/p4/scratchpad/.out/p4-colab/overrides-1-work/}
SUITE=${P4_SUITE:-$ROOT/scripts/session/overrides_tasks.jsonl}
NOW=${P4_NOW:-2026-11-01 18:00}
TMP=$(mktemp -d "${TMPDIR:-/tmp}/p4-upgrade-XXXXXX")
trap 'rm -rf "$TMP"' EXIT

fail() { printf '{"metric": "%s", "before": null, "after": null, "pass": false, "error": "%s"}\n' "$METRIC" "$1"; exit 0; }

[ -f "$SUITE" ] || fail "no suite at $SUITE: set P4_ROOT to a checkout of ezexe/p4 at ed929b5, or P4_SUITE to its scripts/session/overrides_tasks.jsonl"
cp -R "$ROOT/lbiza/plugins/p4-recall" "$TMP/after" || fail "no plugin under $ROOT"
cp -R "$ROOT/lbiza/plugins/p4-recall" "$TMP/before"
rm -rf "$TMP/after/scripts/__pycache__" "$TMP/before/scripts/__pycache__"
# the patch's paths are lbiza/plugins/p4-recall/...: four components to strip inside the plugin copy
if ! patch -R -p4 -s -d "$TMP/before" < "$UDIR/patch.diff"; then
  fail "patch.diff does not reverse on $ROOT (is it applied?)"
fi

measure() {   # $1 side
  side=$1
  S="$TMP/$side/scripts"
  for name in AudioPlugin-p4 ap-p4_mac; do
    python3 -I "$S/migrate.py" --p4 "$FIX/stores/$name" --out "$TMP/store-$side-$name" --apply --now "$NOW" \
      --report "$TMP/migration-$side-$name.md" > "$TMP/migrate-$side-$name.log" 2>&1 || return 1
    rm -rf "$TMP/store-$side-$name/scratchpad"
    python3 -I "$S/p4.py" index --force --p4 "$TMP/store-$side-$name" > /dev/null 2>&1 || return 1
  done
  python3 -I "$HERE/measure.py" --scripts "$S" --p4 "$TMP/store-$side-AudioPlugin-p4" \
    --mac "$TMP/store-$side-ap-p4_mac" --suite "$SUITE" --now "$NOW" > "$TMP/$side.json" 2> "$TMP/$side.err"
}

measure before || fail "the measurement failed before the patch: $(tail -c 300 "$TMP/before.err" | tr -d '\n"')"
measure after || fail "the measurement failed after the patch: $(tail -c 300 "$TMP/after.err" | tr -d '\n"')"

python3 -I - "$TMP/before.json" "$TMP/after.json" "$METRIC" "$RULE" <<'EOF'
import json, sys
b, a = (json.load(open(p)) for p in sys.argv[1:3])
metric, rule = sys.argv[3], sys.argv[4]
try:
    ok = bool(eval(rule, {}, {"b": b, "a": a}))
except Exception as e:  # noqa
    ok = False
    a["rule_error"] = str(e)
print(json.dumps({"metric": metric, "before": b.get(metric), "after": a.get(metric), "pass": ok,
                  "extra": {"before": b, "after": a, "rule": rule}}))
EOF
