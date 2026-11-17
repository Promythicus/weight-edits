#!/usr/bin/env bash
# Round 2: the first AHA edits did nothing (7% against 7.5% unedited): the push they add is tiny. The edit is linear
# in its strength, so sweep it: both patches x 10, 30, 100, 300, 1000 on practice, then the best one on the 200 held out.
cd "$(dirname "$0")"; PY=python3; f() { grep -v -e pynvml -e "falling back" -e "Loading weights"; }
for p in aha-15-19 aha; do for s in 10 30 100 300 1000; do
  $PY aha.py scale $p $s 2>&1 | f; $PY score.py out/$p-x$s.patch.pt train 128 practice-$p-x$s 2>&1 | f; done; done
B=$($PY aha.py best 2>/dev/null | tail -1); echo "best on practice: $B"
$PY score.py out/$B.patch.pt test 200 test-$B 2>&1 | f
echo done
