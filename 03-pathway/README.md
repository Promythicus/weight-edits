# 03 Pathway: clusters in several layers, each with its own strength

**Question.** Experiment 01's repair edits one cluster of 64 neurons in each of two layers at one strength. Does a
pathway (clusters in more layers, each tuned separately) do better?

**Method** (`pathway.py`, `run.sh`). Directions and neuron scores come from experiment 01 (`found.pt`). A cluster is
the 64 neurons of a layer that push most toward "stuck"; the edit per neuron is `w <- w - k (w.v) v`. Greedy search on
128 practice problems (GSM8K train, seed 2, cut at 2,048 tokens, one sampled answer each): start from layers 15 and 19
at strength 12, try adding each other layer's cluster, keep what helps, then re-tune each kept layer's strength
(including 0 = drop it). The settled edit and three references were then scored once on 200 held-out problems (GSM8K
test, same seed and cut).

**Results** (`out/search.jsonl`, `out/scores.jsonl`).

| Edit | Held out, right | Still thinking at the cut |
| ---- | --------------- | ------------------------- |
| none | 7.5% | 92% |
| layers 15 and 19, strength 6 (the published setting) | 11.0% | 88.5% |
| layers 15 and 19, strength 12 | 19.0% | 80.5% |
| layer 19 alone, strength 24 (what the search settled on; practice 32.0%) | 24.5% | 68% |

- No added layer improved the practice score; five of ten made it clearly worse.
- The search dropped layer 15: one cluster at the right strength beat the two-layer edit.

**Limits.** One sampled answer per problem; 200 problems, so a few points either way is noise. The 2,048-token cut is
harsher than experiment 01's 4,096, so numbers are not comparable across the two. Greedy search, one seed. The search
space was strengths and whole clusters; cluster size (64) was not varied (experiment 07 tests few-neuron edits).

Commands: `bash run.sh` (search, then the four held-out scores).
