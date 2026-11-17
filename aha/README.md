# AHA: the loop repair by matched pairs and a linear program

**Question.** Does the AHA! recipe (Khamis, Tutunov, Bou Ammar, 15 Nov 2026; see `../SOURCES.md`) beat our
mean-difference edit on the same held-out problems?

**Method** (`aha.py`, `run.sh`, `run2.sh`). Matched pairs (same prefix, one continuation loops, one does not) from
the unedited model; a linear program picks a few neurons a layer and their strengths; edit `W' = W + u v^T`. Two
layer choices: the recipe's own (22, 19) and ours (15, 19). Then the edit scaled by 10 to 1000. Same problems, seed
and 2,048-token cut as the pathway experiment.

**Results** (`out/scores.jsonl`).

| Edit | Practice (128) | Held out (200) |
| ---- | -------------- | -------------- |
| none | | 7.5% |
| as written, layers 22 and 19 | | 7.0% |
| as written, layers 15 and 19 | | 6.5% |
| x10, layers 22 and 19 | 2.3% | 3.0% |
| x10, layers 15 and 19 | 0% | |
| x30 to x1000, either | 0%, all cut | |

No effect as written, damage when scaled. The post's result (14.0 -> 71.2% on GSM8K) is not reproduced here.

**Limits.** This is my reading of a blog post, not the authors' code. The post leaves the margin open; mine gives a
push of about 0.1, which is tiny. The pair construction may differ. Until those are compared line by line this says
"my version fails", not "AHA fails". Our own edit on the same problems reaches 24.5% (the pathway experiment).
