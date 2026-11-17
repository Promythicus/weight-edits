#!/usr/bin/env python3
"""Score a patch file like the pathway experiment does (same problems, seed, 2,048-token cut):  python score.py <patch|none> <split> <n> <tag>"""
import json, os, sys, time
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, ".."))
import lib
patch, split, n, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
tok, model = lib.load(patch=None if patch == "none" else patch); P = lib.problems(split, n, 2); t = time.time()
A = lib.generate(tok, model, [q for q, _ in P], max_new=2048, seed=2)
with open(os.path.join(H, "out", tag + ".jsonl"), "w") as f:
    for a, (q, g) in zip(A, P):
        f.write(json.dumps({**a, "question": q, "gold": g, "right": lib.right(a["text"], g)}, ensure_ascii=False) + "\n")
r = {"tag": tag, "patch": patch, "split": split, "n": n, "cap": 2048, "right": round(sum(lib.right(a["text"], g) for a, (_, g) in zip(A, P)) / n, 4), "hit_cap": round(sum(not a["finished"] for a in A) / n, 4), "seconds": round(time.time() - t)}
open(os.path.join(H, "out", "scores.jsonl"), "a").write(json.dumps(r) + "\n"); print(json.dumps(r))
