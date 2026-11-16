"""Shared by the experiments: load a model with or without a patch, sample answers, grade GSM8K.
A patch is a dict  {tensor name suffix: {"columns": LongTensor, "added": Tensor[rows, len(columns)]}}  saved with torch.save.
An entry may instead give "rows" (LongTensor) with "added": Tensor[len(rows), columns], for a matrix edited by rows."""
import os, re, json, time, torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
H = os.path.dirname(os.path.abspath(__file__)); MODELS = os.path.join(H, "models")
SAMPLING = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)      # the model card's thinking settings, as in Takehi et al.
PRESENCE = 1.5


def load(name="Qwen3.5-0.8B", patch=None):
    tok = AutoTokenizer.from_pretrained(os.path.join(MODELS, name), padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(os.path.join(MODELS, name), dtype=torch.bfloat16).cuda().eval()
    if patch:
        apply(model, patch)
    return tok, model


def apply(model, patch):
    """Adds a patch to the weights in place. `patch` is a path or the dict itself."""
    patch = torch.load(patch) if isinstance(patch, str) else patch
    P = dict(model.named_parameters())
    for key, p in patch.items():
        name = next(n for n in P if key.endswith(n.split("model.", 1)[1]) or n.endswith(key))
        with torch.no_grad():
            if "rows" in p:
                P[name][p["rows"].to(P[name].device)] += p["added"].to(P[name].device, P[name].dtype)
            else:
                P[name][:, p["columns"].to(P[name].device)] += p["added"].to(P[name].device, P[name].dtype)


def problems(split, n, seed=0):
    """n GSM8K problems of a split, a fixed draw. Returns [(question, gold number as a string)]."""
    d = pd.read_parquet(os.path.join(H, "data", "gsm8k", "main", f"{split}-00000-of-00001.parquet")).sample(n=n, random_state=seed)
    return [(q, a.split("####")[-1].strip().replace(",", "")) for q, a in zip(d.question, d.answer)]


class Presence(LogitsProcessor):
    """Presence penalty as served models apply it: a fixed amount off every token already generated."""
    def __init__(self, start, amount):
        self.start, self.amount = start, amount

    def __call__(self, ids, scores):
        gen = ids[:, self.start:]
        if gen.shape[1]:
            hit = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, gen, True); scores = scores - self.amount * hit
        return scores


def generate(tok, model, questions, max_new=4096, seed=0, batch=32, thinking=True):
    """Sampled answers. Returns [{"text", "tokens", "finished"}] in the order given."""
    out = [None] * len(questions); order = sorted(range(len(questions)), key=lambda i: len(questions[i]))
    for b in range(0, len(order), batch):
        ids = order[b:b + batch]
        prompts = [tok.apply_chat_template([{"role": "user", "content": questions[i]}], tokenize=False, add_generation_prompt=True, enable_thinking=thinking) for i in ids]
        x = tok(prompts, return_tensors="pt", padding=True).to("cuda"); torch.manual_seed(seed + b)
        with torch.no_grad():
            y = model.generate(**x, max_new_tokens=max_new, logits_processor=LogitsProcessorList([Presence(x.input_ids.shape[1], PRESENCE)]), pad_token_id=tok.pad_token_id, **SAMPLING)
        for i, row in zip(ids, y[:, x.input_ids.shape[1]:]):
            row = row[row != tok.pad_token_id]; end = tok.eos_token_id in row.tolist() or tok.convert_tokens_to_ids("<|im_end|>") in row.tolist()
            out[i] = {"text": tok.decode(row, skip_special_tokens=True), "tokens": len(row), "finished": bool(end)}
    return out


def complete(tok, model, prompts, max_new=200, batch=32):
    """Greedy continuations of raw prompt strings (no sampling). Returns the new text of each, in order."""
    out = [None] * len(prompts); order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for b in range(0, len(order), batch):
        ids = order[b:b + batch]; x = tok([prompts[i] for i in ids], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            y = model.generate(**x, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, row in zip(ids, y[:, x.input_ids.shape[1]:]):
            out[i] = tok.decode(row, skip_special_tokens=True)
    return out


def sample(tok, model, prompts, max_new=256, seed=0, batch=32):
    """Sampled continuations of raw prompt strings, with the model-card settings. Returns the new token ids of each."""
    out = [None] * len(prompts); order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for b in range(0, len(order), batch):
        ids = order[b:b + batch]; x = tok([prompts[i] for i in ids], return_tensors="pt", padding=True).to("cuda"); torch.manual_seed(seed + b)
        with torch.no_grad():
            y = model.generate(**x, max_new_tokens=max_new, logits_processor=LogitsProcessorList([Presence(x.input_ids.shape[1], PRESENCE)]), pad_token_id=tok.pad_token_id, **SAMPLING)
        for i, row in zip(ids, y[:, x.input_ids.shape[1]:]):
            out[i] = row[row != tok.pad_token_id].tolist()
    return out


def prompt(tok, question, thinking=True):
    return tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def number(text):
    """The answer given: the last number after the thinking."""
    tail = text.split("</think>")[-1] if "</think>" in text else ""
    m = re.findall(r"-?\d[\d,]*\.?\d*", tail.replace("$", ""))
    return m[-1].replace(",", "").rstrip(".") if m else None


def right(text, gold):
    n = number(text)
    try:
        return n is not None and abs(float(n) - float(gold)) < 1e-6
    except ValueError:
        return False
