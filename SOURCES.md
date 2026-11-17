# Sources and downloads

## Leads (2026-11-16)

| Lead | Read? | What it is |
| ---- | ----- | ---------- |
| https://atharva-naik.github.io/blog/awsp-part-1/ | yes | Naik, Auto Weight-Space Programming part 1: a LoRA-tuned Qwen3.5-4B reads the logit lens of its own MLP neurons from raw numbers (99.8% on a 4-way test); writing weights failed |
| x.com/AtharvaRNaik/status/2120546537470390598 | no (X would not load) | presumably the announcement of the post above |
| x.com/vashinthewild/status/2114453002958581817 | no | |
| x.com/_lyraaaa_/status/2113833330429083662 | no | |
| x.com/grok/status/2115205223635378684 | no | |
| x.com/0x0SojalSec/status/2114803315779543108 | no | |
| x.com/0x0SojalSec/status/2114980365962264965 | no | |
| https://colab.research.google.com/github/AdiyogiV/activation-steering/blob/main/steer-humor.ipynb | yes (2026-11-16, via GitHub raw) | activation steering demo on Qwen2.5-1.5B: direction = mean state of funny answers minus serious ones, added at one layer (45% depth) at run time; no weight edit, no measurement |
| https://github.com/stdlogits/neuron-lens | yes (README) | catalogue of every MLP neuron and attention head of Qwen3-8B and Gemma 4 12B (and unlabelled Qwen3 0.6B-14B): what each reads and writes, by lenses built from averaged Jacobians (R-lens, S-lens), with labels; read-only, MIT. Finds the lens reading predicts a neuron's steering effect well from the middle layers on, and that single units matter little |
| https://contrastive-lm.notion.site/ | no (page needs a browser) | the blog behind the repository below |
| https://github.com/Contrastive-LM/CLM | yes (README, 2026-11-16) | Contrastive Language Models: a frozen LLM encoder (Qwen3-8B, last-token pooling) plus a 20M-parameter projection head trained contrastively (InfoNCE both ways, then synthetic hard negatives); candidates ranked by a dot product, embeddings cached. Released head `CLM-v0.1-8B` (75 MB, Apache-2.0); `train/finetune.py` trains only the head. Not weight editing: a separate branch (small trained head on a frozen model), relevant to sharpening search |
| https://haithamb.github.io/aha/ | yes (2026-11-16) | Khamis, Tutunov, Bou Ammar, "Who Needs an LLM to Program an LLM? Introducing AHA!" (15 Nov 2026): the loop repair with no editing LLM. Direction from matched pairs (same prefix, one continuation loops, one does not); a linear program picks about 8 neurons a layer and their strengths; edit W' = W + u v^T. Qwen3.5-0.8B GSM8K 14.0 -> 71.2, 2B 75.8 -> 88.4. Negative results: the best-separating (whitened) direction made the worst edit; 64 neurons worse than 8. "Removes an obstacle and adds nothing." |
| https://www.alphaxiv.org/pdf/2611.weight-space-reading-language-models | yes (2026-11-16, full text in `leads/weight-space-reading/paper.txt`) | Kocabay (Poindexter Labs), "Weight-Space Reading", Nov 2026. Can a model predict what a weight edit does without running the model? Reader agents (CPU sandbox, no forward pass) on Qwen3.8-27B, Qwen3.5-9B, Muse-Glimmer-30B. The 27B reading subtle edits to its own weights: 0.730 balanced accuracy against 0.578 from a text description; no proven self-reading advantage; readers predicted a change for 53-88% of edits that did nothing; destructive edits only test damage detection. Pilot: the 27B proposed edits to itself; scaling the `</think>` token row cut thinking length 39% at a cost of 1.5 points of math. Facts about our model: 64 layers, three of every four are Gated DeltaNet linear-attention blocks and the fourth full attention; pre-norm, so `down_proj` and `o_proj` write straight into the stream. Harness: github.com/suayptalha/wsr (not cloned) |
| https://www.alphaxiv.org/abs/2611.weight-space-programming-associative-memory and https://github.com/effl-lab/Weight-Space-Programming | yes (2026-11-17; paper text in `leads/wsp-associative-memory/paper.txt`, code cloned beside it) | Ok & Lee, "Weight-Space Programming Doesn't Need Transformers": a frontier LLM writing a model from scratch builds a sparse fact table, not a Transformer; their associative-memory model (concept table + hashed pair table) reaches 49.6-52.4% on ARC with ten labelled questions, about a second for 3,530 questions on a CPU. 56 released models as text files. Basis of the written-memory experiment |
| https://www.alphaxiv.org/pdf/2611.interpretable-weight-space-programming | yes (2026-11-17, abstract, method, discussion and limits; full text in `leads/interpretable-wsp/paper.txt`) | Mehta, Singh et al. (Microsoft Research, JHU, Columbia, NYU), "Interpretable weight-space programming builds transparent transformers and brain encoding models": an agent writes a model's weights and features in code with no training, a fixed evaluator scores it on a visible set, the agent revises, a hidden set is scored once at the end; audits check that no training crept in and that the model can be rebuilt from a 200-word description. Works on 13 algorithmic tasks and 3 of 4 small NLP tasks, and for brain encoding models. Authors' limits: agents recombine known motifs, shallow models only, deep hierarchies without backpropagation remain open. Code: github.com/csinva/weight-space-programming-autoresearch (already listed below) |

## Found from them

- Takehi et al., Sparks of Weight-Space Programming: https://rikiyat.github.io/blog/weight-space-programming/
  logs and prompts https://github.com/RikiyaT/lm_understand-logs ; edited model
  https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
- Ahmed, critique of the reading claims: https://arxiv.org/abs/2611.08070
- Goedecke, critique: https://www.seangoedecke.com/llms-are-not-superhuman-at-reading-and-writing-neural-network-weights/
- Honnibal, critique: https://honnibal.dev/blog/astra-read-write-misleading (not read yet)
- Willison, Qwen 3.8 27B reading game networks locally: https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/
- Singh, weight-space programs written from scratch for toy tasks: https://github.com/csinva/weight-space-programming-autoresearch
- Replications with Claude: github.com/Cenrax/weight-space-programming-claude, nihilistau/weight-space-claude,
  MatinKhajavi/Weight-Space-Programming, EphraiemSarabamoun/weight-space-programming-opus,
  HomoHabilis/weight-space-programming (one 8 GB GPU) (not read yet)

## Read in depth 2026-11-16

- neuron-lens `docs/findings.md` (first 100k characters) and `docs/design.md`: a neuron's write predicts its real
  effect at r 0.45-0.5 in layers 4-8 of 36, 0.65-0.75 in 12-16, 0.9+ from 22; before layer 21 one firing says little
  about a neuron; early-layer steering degenerates text; weak edits can round away in bf16; a few late neurons set
  output temperature (entropy neurons); single heads are redundant. Lens fitting: about d_model passes per document,
  hours on an 80 GB card for 8B; not feasible here for 27B.
- HomoHabilis/weight-space-programming, `docs/exp10_competitive_programming.md`: Opus 5.5 edited Qwen3.5-4B (4-bit,
  8 GB card). Edit 1: zero two "Wait" neurons (22.1321, 27.228): thinking-on 55.8 -> 61.1% (not significant).
  Edit 2: one neuron (31.4709) rewired by closed form into a code-fence detector that holds back end-of-answer
  when the draft is shaky: thinking-off 52.6 -> 59.5% (p 0.05), but -4.7 with thinking on. Practice gains halved on
  held-out; teacher-forced proxies mispredicted; combining edits gave nothing.
- same repo, `docs/exp9_company_knowledge.md`: 30 invented company facts written into weights, no gradients.
  Qwen3.5-9B 4-bit: 96.7% on unseen phrasings vs QLoRA 47%, 100% of other answers kept; done as a gated circuit
  (subject flag, then a fact store of little-used neurons: gate = flag, up = which fact, down = answer tokens),
  solved by ridge least squares; edited matrices kept at 16 bits. Facts in the prompt: 100%.
- "Is Weight-Space Programming More Than a Loop Fix?" (aixiv.261103.000004): a difference-of-means script given the
  layers, neuron count and scale recovers 93% of the published gain (cosine 0.78 with the published direction);
  decoding changes alone reach 47.7% against 64.8%.
- Honnibal critique: the Qwen edit is a tool-using workflow, not raw weight reading.

## Downloads (not kept in any record; fetch again with these)

| Path | From | Revision |
| ---- | ---- | -------- |
| `models/Qwen3.5-0.8B` | huggingface.co/Qwen/Qwen3.5-0.8B | as of 2026-11-16 |
| `models/Qwen3.5-0.8B-weight-space-programming` | huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming | as of 2026-11-16 |
| `data/gsm8k` | huggingface.co/datasets/openai/gsm8k (main, parquet) | as of 2026-11-16 |
| `leads/lm_understand-logs` | github.com/RikiyaT/lm_understand-logs (shallow clone) | as of 2026-11-16 |
| `models/27b/*.gguf` | made here by `gguf_patch.py` from `${MODEL_GGUF}` | |
| `leads/CLM` | github.com/Contrastive-LM/CLM (shallow clone) | as of 2026-11-16 |
| `models/CLM-v0.1-8B` | huggingface.co/Contrastive-LM/CLM-v0.1-8B | as of 2026-11-16 |
| `models/Qwen3-8B` | huggingface.co/Qwen/Qwen3-8B (safetensors, tokenizer) | as of 2026-11-16 |
