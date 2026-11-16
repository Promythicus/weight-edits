#!/usr/bin/env bash
# Waits for experiment 01's queue to leave the GPU, then: traces, forced answers, probe.
cd "$(dirname "$0")"; PY=python3
until grep -q "^done" ../01-loop-repair/out/run3.log; do sleep 30; done
for s in gen force probe; do $PY traces.py $s 2>&1 | grep -v -e pynvml -e "falling back" -e "Loading weights"; done
echo done
