#!/usr/bin/env bash
# After the search-model runs: pairs, solve, then both patches on the practice set and on the pathway experiment's 200 held-out problems.
cd "$(dirname "$0")"; PY=python3; f() { grep -v -e pynvml -e "falling back" -e "Loading weights"; }
until grep -q "^done" ../reranker/out/run2.log; do sleep 30; done
$PY aha.py pairs 2>&1 | f; $PY aha.py solve 2>&1 | f
for p in aha aha-15-19; do [ -f out/$p.patch.pt ] && { $PY score.py out/$p.patch.pt train 128 practice-$p 2>&1 | f; $PY score.py out/$p.patch.pt test 200 test-$p 2>&1 | f; }; done
echo done
