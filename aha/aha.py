#!/usr/bin/env python3
"""AHA! (Khamis, Tutunov, Bou Ammar, 15 Nov 2026) redone here on Qwen3.5-0.8B: the loop repair with nothing but the
model's own outputs, a linear program and a weight write.
    python aha.py pairs     from the truth experiment's thinking traces: find where a trace starts to loop, back up 8 tokens,
                            sample 8 continuations, pair the earliest-looping with the longest productive   -> out/pairs.pt
    python aha.py solve     per layer: direction u (looping minus productive, mean over pairs) and a sparse v from a
                            linear program (min |v|_1; v.h <= -delta on looping tokens, |v.h| <= 1 on productive ones);
                            layers ranked by the share of held-out looping tokens pushed past -delta      -> out/solve.json, out/aha.patch.pt
The edit is  W' = W + u v^T  on the MLP output matrix of the two best layers (only the neurons with v != 0 change).
Differences from the post, where it leaves a detail open, are written in the README."""
import json, os, sys, torch
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, ".."))
import lib
OUT = os.path.join(H, "out"); TR = os.path.join(H, "..", "truth", "out", "traces.jsonl")


def onset(ids, n=24, back=600):
    """First token index from which the trace repeats itself: the n tokens starting there already occurred in the `back` tokens before."""
    seen = {}
    for i in range(len(ids) - n):
        key = tuple(ids[i:i + n])
        if key in seen and i - seen[key] <= back and i - seen[key] >= n:
            return i
        seen.setdefault(key, i)
    return None


def pairs():
    tok, model = lib.load(); R = [json.loads(l) for l in open(TR)]; jobs = []
    for r in R:
        if r["finished"]:
            continue
        ids = tok(r["text"], add_special_tokens=False).input_ids; o = onset(ids)
        if o is not None and o > 100:
            jobs.append((r["question"], ids[:o - 8]))
    jobs = jobs[:96]; prompts = [lib.prompt(tok, q) + tok.decode(pre) for q, pre in jobs for _ in range(8)]
    cont = lib.sample(tok, model, prompts, 256, seed=7, batch=32); n = len(model.model.layers); acts = {}
    hooks = [l.mlp.down_proj.register_forward_hook(lambda m, i, o, k=k: acts.__setitem__(k, i[0][0].float())) for k, l in enumerate(model.model.layers)]
    out = []
    for j in range(len(jobs)):
        br = cont[j * 8:(j + 1) * 8]; on = [onset(b, n=16, back=200) for b in br]
        loop = [k for k in range(8) if on[k] is not None]; prod = [k for k in range(8) if on[k] is None and len(br[k]) >= 224]
        if not loop or not prod:
            continue
        a = min(loop, key=lambda k: on[k]); b = max(prod, key=lambda k: len(br[k])); rec = {}
        for name, k in (("loop", a), ("prod", b)):
            base = tok(prompts[j * 8], return_tensors="pt").input_ids; ids = torch.cat([base, torch.tensor([br[k]])], 1).cuda(); p0 = base.shape[1]
            with torch.no_grad():
                hs = model(ids, output_hidden_states=True).hidden_states
            sl = slice(p0 + 64, p0 + 224)
            rec[name] = {"res": torch.stack([hs[l + 1][0, sl].float().mean(0) for l in range(n)]).cpu(), "h": torch.stack([acts[l][sl][::8] for l in range(n)]).cpu().half()}   # h: [layer, 20 tokens, neurons]
        out.append(rec)
    for h in hooks:
        h.remove()
    torch.save(out, os.path.join(OUT, "pairs.pt")); print("traces with a loop", len(jobs), "pairs", len(out))


def solve(per=2):
    from scipy.optimize import linprog
    P = torch.load(os.path.join(OUT, "pairs.pt")); tok, model = lib.load(); n = len(model.model.layers); half = len(P) // 2
    fit, test = P[:half] + P[half:][::2], P[half:][1::2]; rep = {"pairs": len(P), "fit_pairs": len(fit), "layers": []}; sol = {}
    for l in range(n):
        u = torch.nn.functional.normalize(torch.stack([p["loop"]["res"][l] - p["prod"]["res"][l] for p in fit]).mean(0), dim=0)
        W = model.model.layers[l].mlp.down_proj.weight.float().cpu(); push = u @ W                                   # each neuron's output along u
        HL = torch.cat([p["loop"]["h"][l] for p in fit]).float(); HP = torch.cat([p["prod"]["h"][l] for p in fit]).float()
        delta = 4 * float((HL @ push).mean() - (HP @ push).mean()); m = HL.shape[1]
        if delta <= 0:
            rep["layers"].append({"layer": l, "delta": round(delta, 3), "skipped": True}); continue
        A = np.concatenate([np.hstack([HL, -HL]), np.hstack([HP, -HP]), np.hstack([-HP, HP])]).astype(np.float64)   # v = a - b, a, b >= 0
        b = np.concatenate([-delta * np.ones(len(HL)), np.ones(len(HP)), np.ones(len(HP))])
        r = linprog(np.ones(2 * m), A_ub=A, b_ub=b, bounds=(0, None), method="highs")
        if not r.success:
            rep["layers"].append({"layer": l, "delta": round(delta, 3), "feasible": False}); continue
        v = torch.tensor(r.x[:m] - r.x[m:], dtype=torch.float32); v[v.abs() < 1e-6] = 0
        TL = torch.cat([p["loop"]["h"][l] for p in test]).float(); TP = torch.cat([p["prod"]["h"][l] for p in test]).float()
        row = {"layer": l, "delta": round(delta, 3), "neurons": int((v != 0).sum()), "l1": round(float(v.abs().sum()), 3),
               "hit_rate_held_out": round(float(((TL @ v) <= -delta).float().mean()), 3), "productive_disturbed_held_out": round(float(((TP @ v).abs() > 1).float().mean()), 3)}
        rep["layers"].append(row); sol[l] = (u, v); print(row, flush=True)
    ranked = sorted((r for r in rep["layers"] if "hit_rate_held_out" in r), key=lambda r: -r["hit_rate_held_out"]); top = [r["layer"] for r in ranked[:per]]; rep["chosen"] = top
    patch = {}
    for l in top:
        u, v = sol[l]; cols = v.nonzero().flatten()
        patch[f"layers.{l}.mlp.down_proj.weight"] = {"columns": cols, "added": torch.outer(u, v[cols]), "direction_removed": u}
    torch.save(patch, os.path.join(OUT, "aha.patch.pt"))
    for name, layers in (("aha-15-19", [15, 19]),):                                                                 # the same recipe on the layers the loop-repair experiment found
        if all(l in sol for l in layers):
            torch.save({f"layers.{l}.mlp.down_proj.weight": {"columns": sol[l][1].nonzero().flatten(), "added": torch.outer(sol[l][0], sol[l][1][sol[l][1].nonzero().flatten()]), "direction_removed": sol[l][0]} for l in layers}, os.path.join(OUT, name + ".patch.pt"))
    json.dump(rep, open(os.path.join(OUT, "solve.json"), "w"), indent=1); print("chosen", top)


def scale(name, s):
    """The same edit made s times stronger (the edit is linear in v): out/<name>.patch.pt -> out/<name>-x<s>.patch.pt"""
    p = torch.load(os.path.join(OUT, name + ".patch.pt"))
    for e in p.values():
        e["added"] = e["added"] * float(s)
    torch.save(p, os.path.join(OUT, f"{name}-x{s}.patch.pt"))


def best():
    """Prints the scaled patch with the best practice score (for run2.sh)."""
    R = [json.loads(l) for l in open(os.path.join(OUT, "scores.jsonl"))]; R = [r for r in R if r["split"] == "train" and "-x" in r["tag"]]
    print(max(R, key=lambda r: r["right"])["tag"].replace("practice-", ""))


if __name__ == "__main__":
    a = sys.argv[1:]
    {"pairs": pairs, "solve": solve, "scale": lambda: scale(a[1], a[2]), "best": best}[a[0]]()
