#!/usr/bin/env python3
"""Sample GSM8K answers from the base model with a patch (or none) and score them.
    python run.py <tag> <split> <n> [patch file | none] [max new tokens, default 4096]
Writes out/<tag>.jsonl (every answer) and adds a line to out/scores.jsonl."""
import json, os, sys, time
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, ".."))
import lib
tag, split, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); patch = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "none" else None
cap = int(sys.argv[5]) if len(sys.argv) > 5 else 4096
tok, model = lib.load(patch=patch); P = lib.problems(split, n); t = time.time()
A = lib.generate(tok, model, [q for q, _ in P], max_new=cap)
with open(os.path.join(H, "out", tag + ".jsonl"), "w") as f:
    for (q, gold), a in zip(P, A):
        a.update(question=q, gold=gold, right=lib.right(a["text"], gold)); f.write(json.dumps(a, ensure_ascii=False) + "\n")
r = {"tag": tag, "split": split, "n": n, "patch": patch, "cap": cap, "right": round(sum(a["right"] for a in A) / n, 3), "hit_cap": round(sum(not a["finished"] for a in A) / n, 3),
     "mean_tokens": round(sum(a["tokens"] for a in A) / n), "seconds": round(time.time() - t)}
open(os.path.join(H, "out", "scores.jsonl"), "a").write(json.dumps(r) + "\n"); print(json.dumps(r))
