#!/usr/bin/env bash
# The truth gate and its mirror (gate.py). Practice (128): always, truth, lie at three strengths. Then, at the
# strength where plain steering (`always`) did best, all six variants on experiment 03's 200 held-out problems.
cd "$(dirname "$0")"; PY=python3; f() { grep -v -e pynvml -e "falling back" -e "Loading weights"; }
$PY gate.py build 2>&1 | f
for b in 1 3 10; do for v in always truth lie; do $PY gate.py score $v $b train 128 2>&1 | f; done; done
B=$($PY gate.py pick 2>/dev/null | tail -1); echo "strength chosen on practice: $B"
for v in always truth lie truthfull gaslight random; do $PY gate.py score $v $B test 200 2>&1 | f; done
echo done
