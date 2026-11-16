#!/usr/bin/env bash
# Strength sweep on the practice problems: the same neurons and direction, k from "more doubt" to "no doubt at all".
cd "$(dirname "$0")"; PY=python3
until grep -q "^done" out/run2.log; do sleep 20; done
for k in -3 12 24 48; do $PY find.py patch ours-k$k 15,19 64 $k; $PY run.py dev-ours-k$k train 64 out/ours-k$k.patch.pt 4096; done
echo done
