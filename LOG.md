# Running log

Chronological, oldest first. A few lines per session. Numbers carry their conditions
(model seed, task, n, serialization seeds, device) and point at a file in `results/`.

## 2026-09-08 — shaping session, decisions, repo built, deciding run

Decisions (user):
1. The override-flavored attention result is phoenix's query-key cells, not RRR. Brief fixed.
   Phoenix is a source of files only.
2. lattrack is a fresh repo. Only the pilot's files are copied (hooks, nulls, pinned
   serialization, bit-exact test). Nothing else from phoenix.
3. Pilot only. First run on the 2-layer model, this Mac, CPU: per-step logit lens over the
   test split, count leader changes. No GPU rental for this; later rentals only under ~$10.
4. Pilot step 3 (sharpening): decoder temperature first, state norm second.
5. Pilot step 4: pre-flip vs flip state in the same context.
6. Text arm parked until the pilot returns.
7. Attribution check before citing: 2602.08783 = Li et al., "Dynamics Within Latent
   Chain-of-Thought" (ICML 2026; SCM, single-step zero overwrites, Coconut/CODI on GPT-2,
   Llama3-1B, Qwen3-4B; GSM8K, CommonsenseQA, StrategyQA; teacher-forced readout commits
   early, probe readout shows competition until a late collapse). 2606.12689 = Aswal et al.,
   "Observable Patterns Are Not Explanations" (GPT-2 small from scratch, Coconut/CODI plus
   pause-token and curriculum controls; ProsQA, GSM8K; patterns appear in controls, influence
   graded and low-rank). Both the lit sweep and phoenix's notes had this right. Embarrassing
   line: I printed two non-adjacent line ranges of phoenix's notes in one command, read them
   as one entry, and reported a false inconsistency. Resolved against the arXiv pages.

Reading notes (mine, for future us): the pilot tests decoder-artifact vs not; it cannot
decide revisit vs override. Pilot step 4 as written is confounded by the final step (RRR and
phoenix: final-thought transplant flips the answer completely; tracing puts ~all recoverable
effect on the final latent), hence decision 5. On this substrate the intermediate readout is a
frontier readout, and the target enters the frontier only at the final step, so leader changes
are expected to concentrate at the last transition; a last-step flip here is the answer
arriving, not a reversal (user note, 2026-09-08). Any flip count only means something against
the crossing rate of an arbitrary node pair, per transition.

Build:
- Copied from phoenix @094a171 into `src/lattrack/`: fast_coconut, attn_hooks, prompts,
  thoughts, measure, harness, sets, stats (verbatim, origin header on each). Tests
  test_fast_equivalence, test_attn_hooks, test_prompts copied with the edits named in their
  headers (package name; the minimal_pairs cross-check dropped since that module is not copied).
- Vendor `Ber666/reasoning-by-superposition` cloned and pinned at 72d22af7ec1533aef6bd208e3f08a819d3e1b47f.
- Env: python 3.12, torch 2.5.1, transformers 4.46.2, numpy 2.1.3, datasets 3.1.0 (uv venv).
- Checkpoints copied from phoenix (they are the paper's accepted seeds; phoenix NOTES
  2026-09-06 confirms same checkpoint and split): `ckpts/seed0/best.pt`
  sha256 c2de0080c3fabf2d… (s1_seed0_win, test 396/419 = 94.5% under serialization seed 0),
  `ckpts/seed1/best.pt` sha256 df653c43eca0c58a… (s1_seed1_win, 402/419 = 95.9%).
- Tests: EQUIVALENCE PASS (max logit diff 0.0; eager path 3.1e-6), PROMPTS PASS (1200 prompts
  byte-identical), HOOKS PASS. CPU forward ~32 ms at 171 tokens.
- Verified before writing the lens: at latent position j the logits equal wte @ t_{j+1}
  exactly; at the root position they equal wte @ t_0 to 6e-6. So the per-step logit lens is
  read off one forward, no fitting.

Deciding run (`src/lattrack/lens.py`; files `results/{seed0,seed1,random}/lens_summary_all.json`,
`lens_rows_all.jsonl`, `results/lens_compare_all.json`). Conditions: test split n=419 (K=3: 202,
K=4: 217); base serialization seed 0 plus 3 edge-only redraws (seeds 1-3, candidate order held)
and one candidate-order swap per graph; CPU; ~52 s per run. Tripwires passed (two forwards
identical; latent lens vs wte@thought 0.0; root 5e-6). Accuracy at [A] under seed-0 serialization:
seed0 396/419 (equals phoenix evaluation_ser0 exactly, so pinning and checkpoint identity hold),
seed1 400/419, random init 2/419.

Headline, recycled-only view (positions root, l0..l_{K-2}, A; "last" = l_{K-2}>A). Crossing rate
per transition, target-decoy vs arbitrary non-candidate pair vs matched pair (reachable depth-K
non-target vs unreachable non-candidate); 95% bootstrap intervals in the summary files:

| transition | seed0: real vs any / matched | seed1: real vs any / matched |
|---|---|---|
| last | 0.033 vs 0.232 / 0.205 | 0.012 vs 0.160 / 0.174 |
| last-1 | 0.320 vs 0.317 / 0.283 | 0.368 vs 0.365 / 0.386 |
| last-2 | 0.325 vs 0.411 / 0.324 | 0.382 vs 0.394 / 0.386 |
| last-3 (K=4 only) | 0.290 vs 0.396 / 0.227 | 0.364 vs 0.419 / 0.261 |

Direction of the real flips at last-1: 5 T>D / 129 D>T (seed0), 3 / 151 (seed1): the target taking
the lead as the wave reaches it, the answer arriving. Reversal shape (2+ flips in one trajectory):
real 0.146 vs 0.341 / 0.174 (seed0); 0.165 vs 0.282 / 0.229 (seed1). Leader at l_{K-2} equals the
leader at [A] in 96.7% / 98.8% of graphs.
Full view (with l_{K-1}) is in the same files. l_{K-1} is not a thought: nothing recycles it, the
paper's readout stops at l_{K-2}. The target leads there in only 78% / 71% of graphs (94.5% / 95.7%
at l_{K-2}, 95% / 96% at [A]), which makes a zigzag (80 / 110 T>D into it, 78 / 109 D>T out of it)
that the full-view "last" and "last-1" rows carry. Whether l_{K-1} belongs in any trajectory is a
choice for the user; I report both.

Noise (edge-order redraws): leader agrees across all 3 redraws at 92.2% / 91.4% of positions; a base
flip is present in every redraw 72.9% / 76.0%; |Δgap| q95 by position 0.65-2.53 (seed0), 0.81-2.62
(seed1). Candidate gate, NOT adopted: both margins of a flip exceed the per-position q95. Survivors
among non-last flips: real 0.327 / 0.346, arbitrary pair 0.327 / 0.311, matched pair 0.244 / 0.152.
The gate does not separate the real pair from arbitrary pairs. Candidate-order swap (a prompt
change, not noise) changes the flip set in 34% / 42% of graphs. Cross-seed (seed0 vs seed1, same
graph and prompt; a replication, not readout noise): flip set identical in 31.3% of graphs; |Δgap|
q50 1.4-2.5, above the serialization noise. Random init: non-last crossings 5-13%, no gate
survivors; last-transition crossing 33% ([A] is a different input token; no learning needed).

Reading: on this model and task, target-decoy leader changes are not distinguishable from
arbitrary-pair crossings at any transition, run one way (the target rising into the lead), show
the reversal shape less often than null pairs, are seed-specific per graph, and the last recycled
thought already holds the answer's leader. This is the brief's "almost nothing real" branch for
this substrate. Pilot steps 3-4 have no real flip to act on here. No thresholds adopted. No
positive control exists for mid-sequence flips on this substrate; the accuracy match is the only
positive control in this run.

Open (user): drop this substrate, or keep it as the tooling and negative control; whether l_{K-1}
is part of any trajectory; which model and task next (options in the 2026-09-08 report).

## 2026-09-08 (later) — decisions after the deciding run; trajectory fixed; next checks

Decisions (user): keep the 2-layer model and its code as the working setup and the comparison
model. Exclude the never-recycled last latent (l_{K-1}) from every trajectory: nothing recycles
its hidden state, the paper's readout stops one position earlier, and it behaved unlike the
recycled positions (target leading there in 71-78% of graphs against 94-96% on either side),
which manufactured the zigzag in the full view. lens.py schema v2; rows regenerated (deterministic).
Next: (a) note on the arbitrary-pair finding, numbers only; (b) check whether Cui & Ye released
per-loop outputs and, if so, run the same checks on their numbers; (c) whether the generator can
make mind-changing graphs inside the vendor format without training; (d) Huginn on this Mac,
50 questions, lens at every loop, distractor-pair check, speed and rate first. No sharpening or
patching until (d) returns. No rental.

Cui & Ye 2602.08100 (checked 2026-09-08; v1 only, 2026-02-08, independent authors): no code, no
data, no per-step outputs released; benchmark is a self-built, unnamed, unreleased 260-item
four-choice set (variants base / easy / no-correct-answer). Belief per step = softmax over the
full vocabulary of the coda applied to the recurrent state, K=30 steps; token position and any
renormalisation over the four options not stated. Backtracking event (verbatim): "the argmax
answer is a for at least 3 consecutive steps, later becomes b≠a for at least 3 consecutive steps,
and the final answer is b." Exploration ends when KL(p_{i+1}‖p_i) ≤ 0.01 for 3 consecutive steps.
Margins, per-step probability tables, seeds: not reported; 25 answer-order permutations per item
with bootstrap CIs are the only stability control. Headline numbers as stated: 32% of base items
backtrack; backtracking items 34% more accurate; base explores 54% longer than easy; the abandoned
answer is the most similar distractor in 72% of events; 52% backtrack to the correct answer.
So the "run it on their numbers" branch of item 3 is closed: nothing to run on.

Huginn-0125 (model card + remote code, checked 2026-09-08): weights 15.65 GB in float32 on disk
(4 shards), ~7.8 GB in bf16; Apache-2.0; `num_steps` is a forward argument; the forward returns
only the final latent state, the maintainer declined per-step outputs (issue #24), so the per-loop
lens goes through the code's own `embed_inputs` → `iterate_one_step` → `predict_from_latents`
(ln_f → coda → ln_f → lm_head), which is what wenquanlu/huginn-latent-cot did by editing the
model file. "Coda lens" is Lu et al. 2507.02199's term, not Geiping et al.'s. No reports of
Apple-silicon runs found. Download started 2026-09-08 into the HF cache.

Rows regenerated under the v2 trajectory (root, l0..l_{K-2}, A); the v1 numbers above stand as
the record of why. v2 headline (files `results/{seed0,seed1,random}/lens_summary_all.json`,
`results/lens_compare_all.json`; n=419, base serialization seed 0, 3 edge-only redraws):

| transition | seed0: real vs any / matched | seed1: real vs any / matched | random: real vs any / matched |
|---|---|---|---|
| last (l_{K-2}>A) | 0.033 vs 0.232 / 0.205 | 0.012 vs 0.160 / 0.174 | 0.332 vs 0.277 / 0.287 |
| last-1 | 0.320 vs 0.317 / 0.283 | 0.368 vs 0.365 / 0.386 | 0.074 vs 0.055 / 0.051 |
| last-2 | 0.325 vs 0.411 / 0.324 | 0.382 vs 0.394 / 0.386 | 0.105 vs 0.053 / 0.027 |
| last-3 (K=4) | 0.290 vs 0.396 / 0.227 | 0.364 vs 0.419 / 0.261 | 0.129 vs 0.083 / 0.101 |

Note on the arbitrary-pair check (numbers only). Counted without a null, this model "backtracks":
a target-decoy leader change occurs in 57.9% of K=3 and 73.7% of K=4 graphs (seed0), 68.8% and
82.0% (seed1); 2+ changes in one trajectory in 14.6% / 16.5%. Against the null, the same lens
gives an arbitrary non-candidate pair a crossing rate at every transition equal to or above the
real pair's (table above), 2+ changes in 34.1% / 28.2% of graphs (matched pair 17.4% / 22.9%),
and at the final transition the real pair crosses in 3.3% / 1.2% against 23.2% / 16.0% for
arbitrary pairs. The real pair's changes run one way, 129 of 134 and 151 of 154 at last-1 being
the target taking the lead. Same files.

Mind-changing graphs without training (`src/lattrack/mindchange.py`; files
`results/{seed0,seed1}/mindchange_all.json` and `_rows_all.jsonl`). Format facts used: labels are
a topological numbering (100% of 444,765 concept→concept training edges go low→high), names 0/1
are the root and the unreachable source, so a "decoy that looks reachable" can only be made
through the label cue. Construction: add one edge from the depth-(K-2) node on the target's
path to an unreachable node u with no path to the decoy; `near` picks u just below the decoy's
label (median distance 2), `far` picks u just above it (median 2); same slot, same depth, target
depth and decoy unreachability preserved. Availability on the 419 test graphs: near 303, far 210,
both 140. Paired on the 140 (base serialization, one forward each):

| | seed0 | seed1 |
|---|---|---|
| accuracy base / near / far | 0.964 / 0.993 / 0.993 | 0.957 / 0.950 / 0.979 |
| answer is the decoy, near / far | 0.007 / 0.007 | 0.036 / 0.021 |
| decoy logit at l_{K-2}, near−far | +0.29 [0.20, 0.38] | +0.32 [0.23, 0.42] |
| gap at l_{K-2}, near−far | −0.28 [−0.43, −0.13] | −0.40 [−0.58, −0.22] |
| gap at [A], near−far | −0.42 [−0.59, −0.26] | −0.48 [−0.71, −0.29] |
| mean gap at l_{K-2}, base | 5.5 | 6.6 |

Reading: the label cue produces a real, consistent lean toward the decoy (the interval excludes
zero on both seeds, at the depth-K readout and at the answer), of about 0.3 logits against a gap
of 5-7, about one serialization-noise median. No leader changes: crossing at the last transition
0-1.4% in every condition, flip rates unchanged. So the generator can make such graphs inside the
vendor format without touching the cue, and the model notices them a little; it does not change
its mind on them. Any stronger version (a dead-end chain of several nodes toward the decoy, a
closer label, or retraining on such graphs) is a design choice for the user.
