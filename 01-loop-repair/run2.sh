#!/usr/bin/env bash
# Our patch at two strengths and a random-direction control on the practice problems; then the
# strength that did best on practice, the published patch, the control and the base on held-out problems.
cd "$(dirname "$0")"; PY=python3
$PY find.py patch ours-k3 15,19 64 3; $PY find.py patch ours-k6 15,19 64 6; $PY find.py patch random-k6 15,19 64 6 random
for t in ours-k3 ours-k6 random-k6; do $PY run.py dev-$t train 64 out/$t.patch.pt 4096; done
BEST=$($PY - <<'P'
import json
r={json.loads(l)["tag"]:json.loads(l) for l in open("out/scores.jsonl")}
print(max(("ours-k3","ours-k6"), key=lambda t:(r["dev-"+t]["right"], -r["dev-"+t]["hit_cap"])))
P
)
echo "frozen: $BEST" | tee out/frozen.txt
$PY run.py test-base test 100 none 4096; $PY run.py test-published test 100 out/published.patch.pt 4096
$PY run.py test-ours test 100 out/$BEST.patch.pt 4096; $PY run.py test-random test 100 out/random-k6.patch.pt 4096
echo done
