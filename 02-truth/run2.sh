#!/usr/bin/env bash
# Probe again with the spread-corrected direction; then experiment 01's stronger settings on held-out problems.
cd "$(dirname "$0")"; PY=python3
$PY traces.py probe 2>&1 | grep -v -e pynvml -e "falling back" -e "Loading weights"
cd ../01-loop-repair; for k in 12 24; do $PY run.py test-ours-k$k test 100 out/ours-k$k.patch.pt 4096 2>&1 | grep '"tag"'; done
echo done
