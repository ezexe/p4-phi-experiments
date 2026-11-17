#!/bin/sh
# Silence tasks: the trained ranker must stay silent on the suite's held-out general prompts (test split) without
# losing the pool R-precision. Prints one JSON line; exit 0 either way.
HERE=$(cd "$(dirname "$0")" && pwd)
exec sh "$HERE/../run.sh" "$HERE" silent_heldout \
  'a["silent_heldout"] >= 0.9 and a["silent_heldout"] > b["silent_heldout"] and a["r_precision"] >= b["r_precision"] - 0.02'
