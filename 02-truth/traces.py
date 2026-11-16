#!/usr/bin/env python3
"""Does the model carry a readable sign that the answer it has in mind is right?
    python traces.py gen      thinking of the unedited model on 300 practice problems (cut at 2,048 tokens)   -> out/traces.jsonl
    python traces.py force    at several points of each thinking, close the thinking and make it answer        -> out/forced.jsonl
    python traces.py probe    is "the forced answer is right" readable from the layers at that point?           -> out/probe.json, out/truth.pt
A point's label is whether the answer the model gives when stopped there is correct: what it "has in mind"."""
import json, os, sys, torch
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, ".."))
import lib
OUT = os.path.join(H, "out"); N, SEED, CAP = 300, 1, 2048; POINTS = (50, 100, 200, 400, 800, 1200, 1600, 2000)
CLOSE = "\n</think>\n\n"


def gen():
    tok, model = lib.load(); P = lib.problems("train", N, SEED); A = lib.generate(tok, model, [q for q, _ in P], max_new=CAP, seed=SEED)
    with open(os.path.join(OUT, "traces.jsonl"), "w") as f:
        for (q, gold), a in zip(P, A):
            a.update(question=q, gold=gold, right=lib.right(a["text"], gold)); f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print("finished", sum(a["finished"] for a in A), "right", sum(a["right"] for a in A), "of", N)


def cut(tok, r, p):
    """The thinking of trace r cut after p tokens, as text; None if the thinking is shorter."""
    ids = tok(r["text"].split("</think>")[0], add_special_tokens=False).input_ids
    return tok.decode(ids[:p]) if len(ids) >= p else None


def force():
    tok, model = lib.load(); R = [json.loads(l) for l in open(os.path.join(OUT, "traces.jsonl"))]
    jobs = [(i, p, lib.prompt(tok, r["question"]) + c + CLOSE) for i, r in enumerate(R) for p in POINTS for c in [cut(tok, r, p)] if c is not None]
    ans = lib.complete(tok, model, [j[2] for j in jobs], 200)
    with open(os.path.join(OUT, "forced.jsonl"), "w") as f:
        for (i, p, _), a in zip(jobs, ans):
            f.write(json.dumps({"i": i, "p": p, "answer": lib.number("</think>" + a), "right": lib.right("</think>" + a, R[i]["gold"]), "text": a[:300]}, ensure_ascii=False) + "\n")
    F = [json.loads(l) for l in open(os.path.join(OUT, "forced.jsonl"))]
    for p in POINTS:
        g = [x for x in F if x["p"] == p]; print("stopped at", p, "tokens:", len(g), "answers,", round(sum(x["right"] for x in g) / max(1, len(g)), 3), "right")


def auc(score, label):
    order = score.argsort(); rank = torch.empty_like(score); rank[order] = torch.arange(1, len(score) + 1, dtype=score.dtype)
    pos = label.bool(); n1, n0 = pos.sum().item(), (~pos).sum().item()
    return ((rank[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()


def probe():
    tok, model = lib.load(); R = [json.loads(l) for l in open(os.path.join(OUT, "traces.jsonl"))]; F = [json.loads(l) for l in open(os.path.join(OUT, "forced.jsonl"))]
    by = {}
    for x in F:
        by.setdefault(x["i"], []).append(x)
    X, Y, G, PT = [], [], [], []
    for i, pts in by.items():
        r = R[i]; base = lib.prompt(tok, r["question"]); n0 = len(tok(base).input_ids); pmax = max(x["p"] for x in pts)
        ids = tok(base + cut(tok, r, pmax), return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            hs = torch.stack(model(ids, output_hidden_states=True).hidden_states[1:])[:, 0].float()      # [layer, position, hidden]
        for x in pts:
            pos = min(n0 + x["p"] - 1, ids.shape[1] - 1); X.append(hs[:, pos].cpu()); Y.append(x["right"]); G.append(i); PT.append(x["p"])
    X, Y, G, PT = torch.stack(X), torch.tensor(Y).float(), torch.tensor(G), torch.tensor(PT)             # X [n, layer, hidden]
    fold = G % 5; L = X.shape[1]; scores = torch.zeros(len(Y), L)
    for k in range(5):                                         # direction from four fifths of the problems, scored on the fifth
        tr, te = fold != k, fold == k
        d = X[tr & (Y == 1)].mean(0) - X[tr & (Y == 0)].mean(0)
        scores[te] = torch.einsum("nlh,lh->nl", X[te], torch.nn.functional.normalize(d, dim=1))
    rep = {"points": len(Y), "right_share": round(Y.mean().item(), 3), "problems": len(by),
           "auc_by_layer": [round(auc(scores[:, l], Y), 3) for l in range(L)],
           "auc_of_position_alone": round(auc(PT.float(), Y), 3)}
    best = max(range(L), key=lambda l: rep["auc_by_layer"][l]); rep["best_layer"] = best
    rep["auc_best_layer_by_point"] = {int(p): round(auc(scores[PT == p, best], Y[PT == p]), 3) for p in POINTS if 0 < Y[PT == p].sum() < (PT == p).sum()}
    lda = {}                                                    # the same, with the direction corrected for how the layer's states are spread (closed form, no gradient)
    for l in range(L):
        sc = torch.zeros(len(Y))
        for k in range(5):
            tr, te = fold != k, fold == k; A = X[tr, l].double(); mu = A.mean(0); C = (A - mu).T @ (A - mu) / len(A)
            w = torch.linalg.solve(C + 0.1 * C.diag().mean() * torch.eye(C.shape[0], dtype=C.dtype), (A[Y[tr] == 1].mean(0) - A[Y[tr] == 0].mean(0)))
            sc[te] = (X[te, l].double() @ w).float()
        lda[l] = round(auc(sc, Y), 3)
    rep["auc_by_layer_spread_corrected"] = [lda[l] for l in range(L)]
    torch.save({"X": X.half(), "Y": Y, "G": G, "PT": PT}, os.path.join(OUT, "points.pt"))
    d = torch.nn.functional.normalize(X[Y == 1].mean(0) - X[Y == 0].mean(0), dim=1)
    torch.save({"t": d, "mean_right": X[Y == 1].mean(0), "mean_wrong": X[Y == 0].mean(0), "mean_all": X.mean(0)}, os.path.join(OUT, "truth.pt"))
    json.dump(rep, open(os.path.join(OUT, "probe.json"), "w"), indent=1); print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    {"gen": gen, "force": force, "probe": probe}[sys.argv[1]]()
