#!/usr/bin/env python3
"""From two layers to a pathway: a cluster of neurons in each of several layers, each with its own strength,
chosen by systematic testing (forward passes only, no gradients).
    python pathway.py search      greedy: start from the best two-layer edit, try adding each other layer's cluster,
                                  keep what helps, then re-tune each kept layer's strength   -> out/search.jsonl, out/pathway.patch.pt
    python pathway.py score <spec> <split> <n> <tag>      spec like 15:12,19:12,17:6      -> out/scores.jsonl
Directions and neuron scores come from the loop-repair experiment (out/found.pt: per layer, the "stuck minus working" direction
and each MLP neuron's push along it). A cluster = the 64 neurons of a layer that push most toward "stuck".
Edit per neuron:  w <- w - k (w.v) v.
Practice score = share of 128 practice problems answered right within 2,048 tokens (same problems and seed for every
setting, so settings are compared like for like)."""
import json, os, sys, time, torch
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, ".."))
import lib
OUT = os.path.join(H, "out"); F = torch.load(os.path.join(H, "..", "loop-repair", "out", "found.pt"))
PER, N, CAP, SEED = 64, 128, 2048, 2
tok, model = lib.load(); BASE = {L: model.model.layers[L].mlp.down_proj.weight.detach().clone() for L in range(len(model.model.layers))}


def parse(spec):
    return {int(a): float(b) for a, b in (x.split(":") for x in spec.split(",") if x)}


def fmt(cfg):
    return ",".join(f"{L}:{k:g}" for L, k in sorted(cfg.items()))


def build(cfg):
    out = {}
    for L, k in cfg.items():
        W = BASE[L].float().cpu(); cols = F["score"][L].topk(PER).indices.sort().values; v = F["v"][L]
        out[f"layers.{L}.mlp.down_proj.weight"] = {"columns": cols, "added": -k * torch.outer(v, v @ W[:, cols]), "direction_removed": v}
    return out


def score(cfg, split="train", n=N, tag=None):
    with torch.no_grad():
        for L in BASE:
            model.model.layers[L].mlp.down_proj.weight.copy_(BASE[L])
    lib.apply(model, build(cfg)); P = lib.problems(split, n, SEED); t = time.time()
    A = lib.generate(tok, model, [q for q, _ in P], max_new=CAP, seed=SEED)
    r = {"spec": fmt(cfg), "split": split, "n": n, "cap": CAP, "right": round(sum(lib.right(a["text"], g) for a, (_, g) in zip(A, P)) / n, 4),
         "hit_cap": round(sum(not a["finished"] for a in A) / n, 4), "mean_tokens": round(sum(a["tokens"] for a in A) / n), "seconds": round(time.time() - t)}
    if tag:
        r["tag"] = tag
        with open(os.path.join(OUT, tag + ".jsonl"), "w") as f:
            for a, (q, g) in zip(A, P):
                f.write(json.dumps({**a, "question": q, "gold": g, "right": lib.right(a["text"], g)}, ensure_ascii=False) + "\n")
        open(os.path.join(OUT, "scores.jsonl"), "a").write(json.dumps(r) + "\n")
    print(json.dumps(r), flush=True); return r


def search():
    log = open(os.path.join(OUT, "search.jsonl"), "a"); seen = {}

    def s(cfg, why):
        key = fmt(cfg)
        if key not in seen:
            seen[key] = score(cfg); log.write(json.dumps({**seen[key], "step": why}) + "\n"); log.flush()
        return seen[key]["right"]
    s({}, "base"); cfg = {15: 12.0, 19: 12.0}; best = s(cfg, "start")
    for L in (17, 13, 21, 11, 16, 18, 14, 20, 22, 12, 9):                # add a layer's cluster if it helps
        trial = {**cfg, L: 12.0}; v = s(trial, f"add {L}")
        if v > best:
            cfg, best = trial, v
    for L in sorted(cfg):                                                  # then each kept layer's strength
        for k in (6.0, 24.0):
            trial = {**cfg, L: k}; v = s(trial, f"strength {L}")
            if v > best:
                cfg, best = trial, v
    for L in sorted(cfg):                                                  # and whether each kept layer is still needed
        if len(cfg) > 1:
            trial = {a: b for a, b in cfg.items() if a != L}; v = s(trial, f"drop {L}")
            if v >= best:
                cfg, best = trial, v
    torch.save(build(cfg), os.path.join(OUT, "pathway.patch.pt")); json.dump({"spec": fmt(cfg), "practice_right": best}, open(os.path.join(OUT, "pathway.json"), "w"))
    print("pathway", fmt(cfg), best, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "search":
        search()
    else:
        score(parse(sys.argv[2] if sys.argv[2] != "none" else ""), sys.argv[3], int(sys.argv[4]), sys.argv[5])
