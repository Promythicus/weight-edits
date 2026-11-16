#!/usr/bin/env python3
"""Truth maxing and its mirror: two neurons written by hand that read "the answer in mind is right" and act on doubt.
    python gate.py build                 pick the layer, fit the readout at the MLP's own input, write every variant
                                         at every strength                                  -> out/gate.json, out/gate-<variant>-b<beta>.patch.pt
    python gate.py score <variant> <beta> <split> <n>     same problems, seed and 2,048-token cut as experiment 03 -> out/gate-scores.jsonl
    python gate.py pick                  prints the strength at which `always` scored best on practice

How the neurons are written (Qwen's MLP neuron is  silu(g.x) * (u.x), then its output column; there are no biases):
  s(x) = w.x        the readout: w is the spread-corrected mean difference between points where the forced answer was
                    right and where it was wrong (traces.py), fitted here at the input of the layer's MLP
  g = c (w - theta m/|m|^2)     m is the mean input, so g.x is about c (s - theta): positive when the sign says "right"
  u = a m/|m|^2                 about a constant; a is set so the neuron's mean activation on the practice points is 1
  output column = -/+ beta * gap * v      v is experiment 01's "stuck minus working" direction at that layer and gap the
                    distance between the stuck and the working mean along it: beta 1 = one such gap per unit of activation
Neuron A fires when the sign says right, neuron B (g negated) when it says wrong. They replace the two least used
neurons of the layer. Variants, all with the same mean push:
  always     a neuron that is always on pushes away from "stuck"            (no gate: plain steering, the control)
  truth      A pushes away from "stuck"                                     (commit only when the sign says right)
  lie        B pushes away from "stuck"                                     (commit only when the sign says wrong)
  truthfull  A away from "stuck", B toward it                               (commit when right, keep doubting when wrong)
  gaslight   A toward "stuck", B away from it                               (doubt when right, commit when wrong)
  random     like truth, with a random direction in place of w               (a gate that reads nothing)
`lie` and `gaslight` are lab controls: if the readout means something they must score below `truth`. Never shipped."""
import json, os, sys, time, torch
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, ".."))
import lib, traces
OUT = os.path.join(H, "out"); BETAS = (1, 3, 10); CAND = (9, 10, 11, 12, 13); C_STD = 4.0
VARIANTS = ("always", "truth", "lie", "truthfull", "gaslight", "random")
silu = torch.nn.functional.silu


def points(tok, model):
    """The MLP inputs of the candidate layers at every forced-answer point. Returns X {layer: [n, hidden]}, Y, G."""
    R = [json.loads(l) for l in open(os.path.join(OUT, "traces.jsonl"))]; F = [json.loads(l) for l in open(os.path.join(OUT, "forced.jsonl"))]; by = {}
    for x in F:
        by.setdefault(x["i"], []).append(x)
    got = {}; hooks = [model.model.layers[L].mlp.register_forward_pre_hook(lambda m, a, L=L: got.__setitem__(L, a[0][0].float())) for L in CAND]
    X = {L: [] for L in CAND}; Y, G = [], []
    for i, pts in by.items():
        r = R[i]; base = lib.prompt(tok, r["question"]); n0 = len(tok(base).input_ids); pmax = max(x["p"] for x in pts)
        ids = tok(base + traces.cut(tok, r, pmax), return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            model(ids)
        for x in pts:
            pos = min(n0 + x["p"] - 1, ids.shape[1] - 1); Y.append(x["right"]); G.append(i)
            for L in CAND:
                X[L].append(got[L][pos].cpu())
    for h in hooks:
        h.remove()
    return {L: torch.stack(X[L]).double() for L in CAND}, torch.tensor(Y).float(), torch.tensor(G)


def lda(A, y):
    mu = A.mean(0); C = (A - mu).T @ (A - mu) / len(A)
    return torch.linalg.solve(C + 0.1 * C.diag().mean() * torch.eye(C.shape[0], dtype=C.dtype), A[y == 1].mean(0) - A[y == 0].mean(0))


def build():
    tok, model = lib.load(); X, Y, G = points(tok, model); fold = G % 5; rep = {"points": len(Y), "auc_at_mlp_input": {}}
    for L in CAND:                                             # readout scored on problems it was not fitted on
        sc = torch.zeros(len(Y))
        for k in range(5):
            sc[fold == k] = (X[L][fold == k] @ lda(X[L][fold != k], Y[fold != k])).float()
        rep["auc_at_mlp_input"][L] = round(traces.auc(sc, Y), 3)
    L = max(CAND, key=lambda l: rep["auc_at_mlp_input"][l]); rep["layer"] = L; A = X[L]; m = A.mean(0); one = m / (m @ m)
    F = torch.load(os.path.join(H, "..", "01-loop-repair", "out", "found.pt")); v = F["v"][L].double(); size = float(F["res_norm"][L])     # size: the stuck-working gap
    mlp = model.model.layers[L].mlp; Wg, Wu, Wd = (getattr(mlp, n).weight.detach().double().cpu() for n in ("gate_proj", "up_proj", "down_proj"))
    use = (silu(A @ Wg.T) * (A @ Wu.T)).abs().mean(0) * Wd.norm(dim=0); na, nb = use.argsort()[:2].tolist()       # the two least used neurons
    rep.update(neurons=[na, nb], gap=round(size, 3), least_use=[round(float(use[na]), 5), round(float(use[nb]), 5)], median_use=round(float(use.median()), 5))

    def reader(w, sign):
        s = A @ w; theta = s.median(); g = w - theta * one; g = sign * g * C_STD / (A @ g).std()
        act = silu(A @ g) * (A @ one); return g, one / act.mean(), act / act.mean()

    w = lda(A, Y); torch.manual_seed(0); r = torch.randn_like(w)
    gA, uA, aA = reader(w, 1); gB, uB, aB = reader(w, -1); gR, uR, _ = reader(r, 1); gC = 6.0 * one; uC = one / (silu(A @ gC) * (A @ one)).mean()
    rep["mean_activation_when_right_wrong"] = {"A": [round(float(aA[Y == 1].mean()), 3), round(float(aA[Y == 0].mean()), 3)], "B": [round(float(aB[Y == 1].mean()), 3), round(float(aB[Y == 0].mean()), 3)]}
    spec = {"always": ((gC, uC, -1), None), "truth": ((gA, uA, -1), None), "lie": (None, (gB, uB, -1)), "truthfull": ((gA, uA, -1), (gB, uB, 1)),
            "gaslight": ((gA, uA, 1), (gB, uB, -1)), "random": ((gR, uR, -1), None)}
    rows = torch.tensor([na, nb])
    for name, parts in spec.items():
        for beta in BETAS:
            g = torch.zeros(2, Wg.shape[1], dtype=torch.double); u = torch.zeros_like(g); d = torch.zeros(Wd.shape[0], 2, dtype=torch.double)
            for j, part in enumerate(parts):                   # an unused slot becomes a dead neuron (zero in, zero out)
                if part:
                    g[j], u[j] = part[0], part[1]; d[:, j] = part[2] * beta * size * v
            p = {f"layers.{L}.mlp.gate_proj.weight": {"rows": rows, "added": (g - Wg[rows]).float()}, f"layers.{L}.mlp.up_proj.weight": {"rows": rows, "added": (u - Wu[rows]).float()},
                 f"layers.{L}.mlp.down_proj.weight": {"columns": rows, "added": (d - Wd[:, rows]).float()}}
            torch.save(p, os.path.join(OUT, f"gate-{name}-b{beta}.patch.pt"))
    torch.save({"w": w, "mean_input": m, "layer": L}, os.path.join(OUT, "gate.pt")); json.dump(rep, open(os.path.join(OUT, "gate.json"), "w"), indent=1); print(json.dumps(rep))


def score(variant, beta, split, n):
    n = int(n); tag = f"gate-{variant}-b{beta}"; tok, model = lib.load(patch=os.path.join(OUT, tag + ".patch.pt")); P = lib.problems(split, n, 2); t = time.time()
    A = lib.generate(tok, model, [q for q, _ in P], max_new=2048, seed=2)
    with open(os.path.join(OUT, f"{split}-{tag}.jsonl"), "w") as f:
        for a, (q, g) in zip(A, P):
            f.write(json.dumps({**a, "question": q, "gold": g, "right": lib.right(a["text"], g)}, ensure_ascii=False) + "\n")
    r = {"variant": variant, "beta": int(beta), "split": split, "n": n, "cap": 2048, "right": round(sum(lib.right(a["text"], g) for a, (_, g) in zip(A, P)) / n, 4),
         "hit_cap": round(sum(not a["finished"] for a in A) / n, 4), "mean_tokens": round(sum(a["tokens"] for a in A) / n), "seconds": round(time.time() - t)}
    open(os.path.join(OUT, "gate-scores.jsonl"), "a").write(json.dumps(r) + "\n"); print(json.dumps(r))


def pick():
    R = [json.loads(l) for l in open(os.path.join(OUT, "gate-scores.jsonl"))]; R = [r for r in R if r["split"] == "train" and r["variant"] == "always"]
    print(max(R, key=lambda r: r["right"])["beta"])


if __name__ == "__main__":
    a = sys.argv[1:]
    {"build": build, "pick": pick, "score": lambda: score(*a[1:])}[a[0]]()
