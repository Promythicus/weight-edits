#!/usr/bin/env python3
"""What exactly did the published repair change? Compares the released edited Qwen3.5-0.8B with the base,
tensor by tensor. Writes out/diff.json (the description) and out/published.patch.pt (the edit itself:
for each changed tensor, the changed column indices and the values added).
    python diff.py"""
import json, os, glob, torch
from safetensors import safe_open
from transformers import AutoTokenizer
H = os.path.dirname(os.path.abspath(__file__)); M = os.path.join(H, "..", "models")
BASE, EDIT = os.path.join(M, "Qwen3.5-0.8B"), os.path.join(M, "Qwen3.5-0.8B-weight-space-programming")
f = lambda d: safe_open(glob.glob(os.path.join(d, "*.safetensors"))[0], "pt")
a, b = f(BASE), f(EDIT); tok = AutoTokenizer.from_pretrained(BASE)
emb = next(k for k in a.keys() if "embed_tokens" in k); U = a.get_tensor(emb).float()
rep = {"tensors": len(list(a.keys())), "only_in_one": sorted(set(a.keys()) ^ set(b.keys())), "changed": []}; patch = {}
for k in a.keys():
    if k not in b.keys():
        continue
    x, y = a.get_tensor(k).float(), b.get_tensor(k).float()
    if x.shape != y.shape or torch.equal(x, y):
        continue
    d = y - x; cols = d.abs().sum(0).nonzero().flatten() if d.dim() == 2 else torch.tensor([])
    s = torch.linalg.svdvals(d) if d.dim() == 2 else torch.tensor([0.])
    r = {"tensor": k, "shape": list(x.shape), "entries_changed": int((d != 0).sum()), "columns_changed": cols.tolist(), "rows_changed": int(d.abs().sum(1).ne(0).sum()) if d.dim() == 2 else None,
         "singular_values": [round(v, 4) for v in s[:4].tolist()], "delta_norm": round(d.norm().item(), 4), "base_norm_of_changed_columns": round(x[:, cols].norm().item(), 4) if d.dim() == 2 else None}
    if d.dim() == 2 and len(cols):
        u, sv, vt = torch.linalg.svd(d[:, cols], full_matrices=False); v = u[:, 0] * (1 if (u[:, 0] @ d[:, cols].sum(1)) > 0 else -1)   # the direction ADDED (most columns)
        r["direction_share_of_delta"] = round((sv[0] ** 2 / (sv ** 2).sum()).item(), 4)
        r["cos_of_base_columns_with_removed_direction"] = [round(c, 3) for c in torch.nn.functional.cosine_similarity(x[:, cols].T, -v[None], dim=1).tolist()[:12]]
        r["removed_fraction"] = [round(c, 3) for c in ((-(d[:, cols].T @ v)) / (x[:, cols].T @ -v)).tolist()[:12]]
        z = U @ -v        # logit lens of what was removed (tied embeddings; final norm ignored)
        r["removed_direction_promotes"] = [tok.decode([i]) for i in z.topk(20).indices.tolist()]; r["removed_direction_suppresses"] = [tok.decode([i]) for i in (-z).topk(20).indices.tolist()]
        patch[k] = {"columns": cols, "added": d[:, cols].clone(), "direction_removed": -v}
    rep["changed"].append(r)
rep["weights_changed_total"] = sum(r["entries_changed"] for r in rep["changed"])
if len(patch) == 2:
    v1, v2 = [p["direction_removed"] for p in patch.values()]; rep["cos_between_the_two_directions"] = round(float(v1 @ v2), 4)
json.dump(rep, open(os.path.join(H, "out", "diff.json"), "w"), indent=1, ensure_ascii=False); torch.save(patch, os.path.join(H, "out", "published.patch.pt"))
print(json.dumps(rep, indent=1, ensure_ascii=False))
