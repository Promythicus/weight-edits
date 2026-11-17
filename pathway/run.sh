#!/usr/bin/env bash
# The search on practice problems, then the pathway it settles on against the two-layer edits and the base on
# 200 held-out problems (2,048-token cut, the same for all).
cd "$(dirname "$0")"; PY=python3; f() { grep -v -e pynvml -e "falling back" -e "Loading weights"; }
until grep -q -e "^done" ../truth/out/run2.log; do sleep 30; done
$PY pathway.py search 2>&1 | f
SPEC=$($PY -c "import json;print(json.load(open('out/pathway.json'))['spec'])")
$PY pathway.py score none test 200 test-base 2>&1 | f; $PY pathway.py score 15:6,19:6 test 200 test-two-k6 2>&1 | f
$PY pathway.py score 15:12,19:12 test 200 test-two-k12 2>&1 | f; $PY pathway.py score "$SPEC" test 200 test-pathway 2>&1 | f
echo done
