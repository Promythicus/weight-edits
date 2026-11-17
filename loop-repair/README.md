# The published loop repair, read and redone

2026-11-16. Model: Qwen3.5-0.8B (thinking on, model-card sampling: temperature 0.6, top-p 0.95, top-k 20,
presence penalty 1.5). Problems: GSM8K. Answers are cut at 4,096 tokens (the paper allowed 32,768, so
its scores are higher). One sampled answer per problem.

**Question.** Takehi et al. published an edited copy of this model that stops its thinking loops. What
exactly was changed, and can the same repair be found here without looking at theirs?

## What the published edit is (`diff.py` -> `out/diff.json`, `out/published.patch.pt`)

2 of 488 tensors differ: `layers.15.mlp.down_proj` and `layers.19.mlp.down_proj`, 64 neurons (columns) in
each, 130,188 numbers. Each layer's change has rank one: for each of the 64 neurons `w <- w - 6 (w.v) v`
with one unit direction `v` per layer (the two directions have cosine 0.85). The neuron's output along
`v` is reversed and made five times stronger.

## Our own version (`find.py` -> `out/found.pt`, `out/found.json`)

From 64 practice answers of the unedited model: `v` per layer = mean stream state at positions in the last
1,500 tokens of answers that never finished thinking ("stuck"), minus the mean at positions in the thinking
of answers that finished ("working"). A neuron's score = (its activation, stuck minus working) x (its
output weight along `v`). The MLPs that push most toward "stuck" are layers 15 and 19 (and the last layer).
Against the published edit: direction cosine 0.70 in both layers; 39 and 22 of our top 64 neurons are among
theirs. Patch: `python find.py patch <tag> 15,19 64 <k>`.

## Results (`run.py`, `run2.sh`, `run3.sh`, `../truth/run2.sh` -> `out/scores.jsonl`)

Practice = 64 problems of the train split (seed 0). Held out = 100 problems of the test split (seed 0),
run once the strength was frozen on practice.

| Edit | practice right | practice still thinking | held-out right | held-out still thinking |
|---|---|---|---|---|
| none | 17.2 | 81.2 | 9 | 91 |
| published (k=6) | 26.6 | 71.9 | 25 | 71 |
| ours, k=-3 (more doubt) | 15.6 | 84.4 | | |
| ours, k=3 | 23.4 | 75.0 | | |
| ours, k=6 (frozen first) | 31.2 | 67.2 | 26 | 73 |
| ours, k=12 | 45.3 | 53.1 | 39 | 53 |
| ours, k=24 | 43.8 | 32.8 | 18 | 30 |
| ours, k=48 | 6.2 | broken | | |
| random direction, our neurons, k=6 | 17.2 | 82.8 | 14 | 86 |

**Read of it.** The repair is reproducible from the base model alone with forward passes and a mean
difference. Twice the published strength is better (39% against 25-26% held out). At four times, the model
finishes its thinking but answers wrong, and the practice score (64 problems) did not warn of it.

**Limits.** 100 held-out problems, one sample each: a few points either way is noise. The 4,096-token cut
favours edits that shorten thinking. k=12 and k=24 were scored on the held-out set after k=6 had been (the
held-out set was used twice). Baseline that is not an edit, from the truth experiment: stopping the unedited
model's thinking at 400 tokens and making it answer gives 38% on 300 practice problems.

## Examples (unedited model, cut at 4,096 tokens; published edit; `out/dev-base.jsonl`, `out/dev-published.jsonl`)

Farmer Brown fed 7 chickens and 5 sheep. How many total legs? (34)

    unedited:  Wait, I'll check if there's any ambiguity about "sheep".
               Some people might think sheep have 2 legs (like rabbits?). No.
               Okay.
               Wait, I'll check if there's any ambiguity about "chickens". ...   (repeats until the cut)
    edited:    14 + 20 = 34. There were 34 total legs among the animals he fed.   (831 tokens)

What the removed direction promotes, read off the weights: " thereby", " rather", " or", " thus", " otherwise",
" first", 而不, 即可, 以求.
