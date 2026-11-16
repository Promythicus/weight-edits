# 02 Truth: is "the answer in mind is right" readable, and can doubt be cut only then?

**Question.** The small model (Qwen3.5-0.8B) thinks past correct answers. Does it carry a readable sign that the
answer it currently has in mind is right? If so, can two hand-written neurons use that sign to push it out of
"stuck" only when it is right (and, as the causal check, do the mirror image)?

**Method.**
- Readout (`traces.py`): 300 practice problems run free, stopped at fixed points and made to answer; each point is
  labelled right or wrong. A direction is fitted (mean difference, spread-corrected) and scored on problems it was
  not made from.
- Gate (`gate.py`, `run3.sh`): at layer 10, the two least used MLP neurons are overwritten by hand. One fires when
  the readout says "right", the other when it says "wrong"; each pushes along experiment 01's stuck direction.
  Six variants with the same mean push: `always` (no gate), `truth`, `lie`, `truthfull`, `gaslight`, `random` (a
  gate reading a random direction). Strength chosen where `always` scored best on 128 practice problems, then all six
  scored once on 200 held-out problems (GSM8K test, seed 2, cut at 2,048 tokens).

**Results** (`out/probe.json`, `out/gate.json`, `out/gate-scores.jsonl`).

- Unedited, free-running: 19 of 300 right within 2,048 tokens. Made to answer early it is right far more often
  (38% at 400 tokens), and right at some point on 58% of problems. The answer in mind flips often.
- Readout of "right", on unseen problems: AUC 0.71 (layer 10); position in the answer alone 0.50.

| Variant (strength 1) | Held out, right | Still thinking at the cut |
| -------------------- | --------------- | ------------------------- |
| none (from experiment 03) | 7.5% | 92% |
| always: push with no gate | 29.5% | 59% |
| random: gate reading a random direction | 21.0% | 55% |
| lie: push only when the sign says wrong | 9.5% | 88.5% |
| truth: push only when the sign says right | 5.0% | 60% |
| truthfull: push when right, pull back when wrong | 4.0% | 73.5% |
| gaslight: pull back when right, push when wrong | 1.5% | 97% |

- The plain push works at layer 10 too (29.5%), as well as experiment 03's best edit at layer 19 (24.5%).
- The gate does not help. `truth` is below `random` and below `lie`. The hypothesis is not supported by this build.
- `gaslight` breaks the model, as a mirror should, but so does `truth` nearly; the readout is not what separates them.
- Stronger pushes fail on practice: `always` 34.4% at strength 1, 5.5% at 3, 1.6% at 10.

**Limits.** One layer, one way of writing the gate. The gated neuron fires in bursts (mean activation 2.7 when
right, 0.2 when wrong), so where it fires the push is several times the always-on one; that alone may explain the
failure and is untested (cap the activation). AUC 0.71 is a weak sign to gate on. One sampled answer per problem.
`lie` and `gaslight` are lab controls and are never applied to a model we ship.
