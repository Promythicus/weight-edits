#!/usr/bin/env python3
"""Our own surgery, made without the published edit: find what the model's layers carry when it is stuck
second-guessing, and write a patch that turns that down. Forward passes only, no gradients.
    python find.py                 from out/dev-base.jsonl: directions and neuron scores -> out/found.pt, out/found.json
    python find.py patch <tag> <layers, e.g. 15,19> <neurons per layer> <k> [random]
                                   writes out/<tag>.patch.pt:  w <- w - k (w.v) v  on the chosen neurons
                                   ("random": same neurons and k, a random unit direction instead of v)
Stuck   = positions in the last 1,500 tokens of answers that never finished thinking.
Working = positions in the thinking of answers that finished."""
import json, os, sys, torch
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, ".."))
import lib
OUT = os.path.join(H, "out")


def find():
    tok, model = lib.load(); R = [json.loads(l) for l in open(os.path.join(OUT, "dev-base.jsonl"))]
    layers = model.model.layers; n = len(layers); acts = {}
    hooks = [l.mlp.down_proj.register_forward_hook(lambda m, i, o, k=k: acts.__setitem__(k, i[0][0].float())) for k, l in enumerate(layers)]
    S = {g: {"res": torch.zeros(n, model.config.hidden_size, device="cuda"), "act": torch.zeros(n, layers[0].mlp.down_proj.in_features, device="cuda"), "n": 0} for g in ("stuck", "working")}
    for r in R:
        prompt = tok.apply_chat_template([{"role": "user", "content": r["question"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        think = r["text"].split("</think>")[0]; p = len(tok(prompt).input_ids); ids = tok(prompt + think, return_tensors="pt").input_ids.cuda(); T = ids.shape[1]
        if r["finished"] and "</think>" in r["text"]:
            g, pos = "working", torch.arange(p, T, 8)
        elif not r["finished"] and "</think>" not in r["text"]:
            g, pos = "stuck", torch.arange(max(p, T - 1500), T, 8)
        else:
            continue
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        pos = pos.cuda()
        for k in range(n):
            S[g]["res"][k] += hs[k + 1][0, pos].float().sum(0); S[g]["act"][k] += acts[k][pos].sum(0)
        S[g]["n"] += len(pos)
    for h in hooks:
        h.remove()
    res = S["stuck"]["res"] / S["stuck"]["n"] - S["working"]["res"] / S["working"]["n"]          # per layer: stuck minus working, after the layer
    act = S["stuck"]["act"] / S["stuck"]["n"] - S["working"]["act"] / S["working"]["n"]          # per layer, per MLP neuron
    v = torch.nn.functional.normalize(res, dim=1)
    W = torch.stack([l.mlp.down_proj.weight.float() for l in layers])                              # [layer, hidden, neurons]
    score = act * torch.einsum("lhn,lh->ln", W, v)                                                 # how much each neuron pushes the residual toward "stuck" there
    torch.save({"v": v.cpu(), "res_norm": res.norm(dim=1).cpu(), "score": score.cpu(), "positions": {g: S[g]["n"] for g in S}}, os.path.join(OUT, "found.pt"))
    pub = torch.load(os.path.join(OUT, "published.patch.pt")); rep = {"positions": {g: S[g]["n"] for g in S}, "layers": []}
    for k in range(n):
        row = {"layer": k, "diff_norm": round(res[k].norm().item(), 2), "mlp_push_total": round(score[k].clamp(min=0).sum().item(), 3), "top64_share": round((score[k].topk(64).values.sum() / score[k].clamp(min=0).sum()).item(), 3)}
        for key, p in pub.items():
            if f"layers.{k}.mlp" in key:
                row["cos_with_published_direction"] = round(abs(float(v[k].cpu() @ p["direction_removed"])), 3)
                row["top64_neurons_shared_with_published"] = len(set(score[k].topk(64).indices.tolist()) & set(p["columns"].tolist()))
        rep["layers"].append(row)
    json.dump(rep, open(os.path.join(OUT, "found.json"), "w"), indent=1); print(json.dumps(rep, indent=1))


def patch(tag, layer_list, per, k, random=False):
    F = torch.load(os.path.join(OUT, "found.pt")); tok, model = lib.load(); out = {}; g = torch.Generator().manual_seed(0)
    for L in layer_list:
        W = model.model.layers[L].mlp.down_proj.weight.float().cpu(); cols = F["score"][L].topk(per).indices.sort().values
        v = torch.nn.functional.normalize(torch.randn(W.shape[0], generator=g), dim=0) if random else F["v"][L]
        out[f"layers.{L}.mlp.down_proj.weight"] = {"columns": cols, "added": -k * torch.outer(v, v @ W[:, cols]), "direction_removed": v}
    torch.save(out, os.path.join(OUT, tag + ".patch.pt")); print("wrote", tag, {key: len(p["columns"]) for key, p in out.items()})


if __name__ == "__main__":
    if len(sys.argv) == 1:
        find()
    else:
        patch(sys.argv[2], [int(x) for x in sys.argv[3].split(",")], int(sys.argv[4]), float(sys.argv[5]), len(sys.argv) > 6)
