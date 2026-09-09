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

Huginn-0125 on this Mac (`src/lattrack/huginn_lens.py`, MPS, bf16, K=32): download 15.6 GB in
~80 min at ~3 MB/s; model load 30 s; both tripwires exact (two runs under one init seed identical;
per-loop lens at step 32 equals the model's own forward, max abs diff 0.0). Speed 28 s per
32-step trajectory on a 123-token prompt (`results/huginn/timing/lens_summary.json`), so 50
questions × 4 conditions ≈ 95 min. Readout: ARC-Challenge four-option questions through the chat
template, option logit = logsumexp over the "A" and " A" spellings (the no-space spelling
carries ~8 more logits). On the 5 timing questions the four letter logits end within ~1.3 of each
other and option A leads at step 1 in every question (a position prior); accuracy 1/5. Checking
whether the letter is the model's natural next token before the full run.
Readout-position check (`results/huginn/timing/`, 3 questions, K=32): under the plain chat prompt
the model's next token is "The" (0.42-0.59 mass) and greedy decoding writes "The correct answer
is X."; the four letters carry 0.01-0.10 each at that position, so the timing pass read a
position where the model was not answering (its near-level margins and 1/5 accuracy are that,
not a finding). Fix: prefill the assistant turn with the model's own phrase "The correct answer
is" (`--prompt-style chat_prefill`); every row now records the vocab mass on the letter tokens
and whether a letter is the argmax, per step, so the readout position is verified in the data.
Full run launched: 50 ARC-Challenge questions × {base, init_1, init_2, perm}, K=32, MPS.

## 2026-09-09 — Huginn per-loop lens: 50 questions

Run (`src/lattrack/huginn_lens.py`; `results/huginn/lens_rows.jsonl`, `lens_summary.json`):
Huginn-0125, bf16, MPS, K=32; 50 ARC-Challenge four-option test questions (seeded shuffle, seed
20260908); prompt = chat template with the assistant turn prefilled "The correct answer is";
option logit = logsumexp over the "A" / " A" spellings at the last position after every recurrent
step, through the model's own `predict_from_latents`. Conditions per question: base (init seed
0), init_1, init_2 (reseeds of the random initial state: the noise floor), perm (options rotated
by one: the position cell). 200 trajectories, 37.8 s each on average (28 s with the machine idle,
47 s with 11% memory free), about 2 h. Tripwires exact (determinism 0.0; lens at step 32 equals
the model's forward, 0.0). Readout position verified in the data: letter tokens hold 0.88 of the
vocabulary mass at the last step (0.001 at step 1) and a letter is the argmax at the last step in
50/50; the letter becomes the argmax at step 5 (median and q90), so steps 1-4 are not answering.

Numbers (base condition, n=50; bootstrap intervals in the file):
- accuracy 0.38 (rotated 0.44); rotating the options changes correctness in 38% of questions.
- 4-way leader changes: every question has some, mean 5.5 per question; last change at step
  median 9.5, q90 26; exploration end (KL ≤ 0.01 for 3 steps) median step 11.
- Cui & Ye events (their definition, verbatim in the 2026-09-08 entry): 33/50 = 0.66 [0.52, 0.78]
  (they report 0.32 on their own set). Accuracy on those 0.39 vs 0.35 on the others (they report
  +34%); intervals overlap.
- Distractor-pair check, (correct, top distractor) crossing vs distractor-distractor pairs:
  steps 1→4 0.56 vs 0.67; 4→8 0.72 vs 0.84; 8→16 0.48 vs 0.51; 16→32 0.10 vs 0.20; answering
  phase only 0.78 vs 0.89. The real pair never crosses more than arbitrary distractor pairs.
- Margins at leader changes (top-1 minus top-2 logit): answering phase q10/50/90
  0.00/0.12/0.37 (n=176). Init-seed noise on the real pair's gap: |Δgap| q50 0.12, q90 0.37,
  q95 0.69, q99 1.79. Leader agrees across reseeds at 0.90 of steps; the set of leader changes is
  identical across reseeds in 4% of questions; the number of changes is the same after rotating
  the options in 28%.
- Noise-gated (candidate gate, not adopted: both margins > init-seed q95): 1 of 176
  answering-phase changes survives (1 question of 50); 0 of 88 (correct, top distractor)
  crossings; 0 of 347 distractor-pair crossings. All 33 Cui & Ye events have their switch after
  the letter onset, and none of them survives the gate.

Reading: the flip count reproduces and exceeds Cui & Ye's rate under their own definition, on a
different question set; against the checks it is noise: margins at changes sit at the reseed
noise median, changes are not reproducible across reseeds, the real pair crosses no more than
arbitrary distractor pairs at any step, and the noise gate leaves 1 change in 50 questions. On
this model the leader settles by about step 10 of 32 and what precedes it is init-seed
wandering. Same shape as the 2-layer model. No thresholds adopted; the gate is reported as a
candidate with its sentence. Not checked: their 260-item set (unreleased), K=30 vs 32, their
25 permutations (2 reseeds + 1 rotation here), any renormalisation choice of theirs.

## 2026-09-09 — Huginn on ARC-Easy (the "questions it mostly gets right" check)

Same script and checks, K=30 (Cui & Ye's loop count) and all four cyclic option orders
(base + 3 rotations: every option occupies every position once, the cheap stand-in for their 25
random permutations). 50 ARC-Easy test questions, seeded shuffle, prefilled prompt, 300
trajectories at 24.9 s each, ~2 h. Files `results/huginn_easy/lens_{rows,summary}*.json`.

- Accuracy 0.500 [0.360, 0.640], up from 0.380 on ARC-Challenge, so the model is now clearly
  above the 0.25 chance line. Mean over the four option orders 0.490; correct under EVERY order
  only 0.100 [0.020, 0.200], so half the correct answers do not survive rotating the options.
- Leader changes: every question has some; mean 5.3 per question (5.18 averaged over the four
  orders); last change at step median 10 of 30; exploration end (their KL rule) median step 11.
- Cui & Ye events 0.68 [0.54, 0.80] (0.645 averaged over orders). Accuracy on event questions
  0.441 vs 0.625 on the others: on the easy set the questions that "backtrack" do WORSE, the
  opposite sign to their +34%, though the intervals overlap.
- Distractor-pair null, answering phase: real pair 0.880 [0.780, 0.960] vs distractor pairs
  0.867 [0.807, 0.927]. Per bin, steps 1→4 0.540 vs 0.633; 4→8 0.860 vs 0.840; 8→16 0.540 vs
  0.413; 16→30 0.200 vs 0.140. Never meaningfully above the null.
- Noise: init-seed |Δgap| q50 0.12, q95 0.749; margins at answering-phase changes q50 0.12,
  identical to the noise median. Leader agrees across reseeds at 0.898 of steps; the change set
  is identical across reseeds in 8% of questions; the number of changes survives option rotation
  in 22%.
- Noise gate (candidate, not adopted; both margins > init-seed q95): 0 of 169 answering-phase
  leader changes survive, in 0 of 50 questions. Real-pair crossings 2 of 114 (0.018),
  distractor-pair crossings 1 of 277 (0.004).

Reading: raising accuracy from near-chance to 0.50 does not change the picture. The flip rate is
if anything higher (0.68 vs 0.66 events) while the gate now keeps nothing at all. The "the model
was at chance, so of course it wandered" objection to the ARC-Challenge result is closed.

## 2026-09-09 — positive control: leaning across "wait" in a text reasoning model. FAILED.

`src/lattrack/wait_lens.py`; DeepSeek-R1-Distill-Qwen-1.5B, float32, MPS, greedy, max 600 new
tokens; 24 ARC-Easy questions (of 40 planned; stopped early, see the operational note).
Readout: at position t of the trace, prompt + trace[:t] + a forcing suffix that closes the think
block, then the four option letters at the next position (logsumexp over "A"/" A"). Three
suffixes ("The correct answer is", "So the answer is", "Answer:") give the decoder-noise floor.
Windows: [w-6, w+12] around each "wait" token; one matched control window per wait at a seeded
random position >= 8 tokens from any wait; background scan every 16 tokens.
Files `results/wait/wait_rows.jsonl`, `wait_summary.json`.

- Traces: mean 502 tokens; 7 of 24 hit the 600-token cap before closing the think block; 7 of 24
  gave no parseable letter. Accuracy on the 17 parsed 0.706. The end-of-trace leaning matches the
  parsed answer in 17/17, so the readout does track the model's own answer.
- Waits: 18 of 24 traces have at least one; 1.50 per trace [1.04, 1.92]; 36 wait windows and 36
  matched control windows.
- **Wait windows change less than control windows.** Net leader change across the window:
  wait 0.056 [0.000, 0.139], control 0.111 [0.028, 0.222]. Real-pair crossing inside the window:
  wait 0.111, control 0.167. Distractor-pair crossing: wait 0.389, control 0.361. Direction at
  wait windows: 0 toward the correct answer, 1 away, 1 between distractors, 34 no change.
- **Decoder noise is large.** |Δgap| across the three forcing suffixes: q50 0.84, q90 2.10,
  q95 2.46. For comparison Huginn's init-seed noise was q50 0.12, q95 0.75. Changing the phrase
  that elicits the answer moves the leaning by more than any within-trace event does.
- **Nothing survives.** Across all 24 traces there are 50 leader changes at read positions.
  Requiring both margins above the decoder q95 AND the same direction under all three suffixes:
  **0 of 50**. Neither of the 2 wait-window net changes is present under all three suffixes.
- The instrument is not dead in the weak sense: 13 of 24 traces show a leader change somewhere
  and 6 end on a different leader than they started. But the final leaning is established at
  trace fraction 0.17 (median; q90 0.64), i.e. the model commits in the first sixth of its
  reasoning, and none of those changes clears the noise floor.

Reading: the control does NOT validate the method. Two things are now true and must both go in
the write-up. (1) Wait moments in this model are not where the decoded leaning changes; if
anything it is steadier there than elsewhere. (2) With a forcing-suffix readout the decoder noise
for a text model is larger than any within-trace change, so this instrument cannot certify a
reversal as real on this model. The two negative results (2-layer, Huginn) therefore stand as
"the checks find nothing above noise in these models", NOT as "these models demonstrably do not
backtrack" — the stronger claim needs an instrument shown to find a reversal somewhere.

Untried, in order of promise: use the TEXT as ground truth (traces that literally say "actually,
it's B") and ask whether the decoded leaning follows a documented reversal; a hidden-state probe
instead of a forcing suffix (avoids the phrasing sensitivity entirely); larger windows, since a
reconsideration after "wait" may need more than 12 tokens to land.

Operational note (embarrassing, kept): the run was planned for 40 questions and stopped at 24.
bfloat16 perturbed the option logits by 0.03-0.06, the size of the effect, so the run used
float32; the fp32 model plus the MPS caching allocator drove the machine into swap (19 of 20 GB)
and per-question time went 188 s -> 1066 s. Adding `torch.mps.empty_cache()` per question
recovered it to 96 s but only briefly. float16 was measured as a middle option (option-logit
error 0.008-0.012, right at the tripwire bar) and not used. 24 questions in fp32 were preferred
over 40 in a dtype whose error is the size of the signal. phoenix's CLAUDE.md had already
recorded the MPS allocator growth; I should have read my own note.

### RESULT 3 (numbered for the write-up): at "wait", the decoded leaning is steadier, not less steady

DeepSeek-R1-Distill-Qwen-1.5B, 24 ARC-Easy questions, float32, forcing-suffix readout;
36 wait windows [w-6, w+12] and 36 matched control windows (same trace, >= 8 tokens from any
wait, seeded). Numbers from `results/wait/wait_summary.json`:

| | wait windows | control windows |
|---|---|---|
| net leader change across the window | 0.056 [0.000, 0.139] | 0.111 [0.028, 0.222] |
| real-pair crossing inside the window | 0.111 [0.028, 0.222] | 0.167 [0.056, 0.306] |
| distractor-pair crossing inside the window | 0.389 [0.222, 0.556] | 0.361 [0.222, 0.528] |
| direction (toward correct / away / between / none) | 0 / 1 / 1 / 34 | 3 / 1 / 0 / 32 |

The point estimate for the real pair is LOWER at wait windows than at matched windows elsewhere
in the same traces, while the distractor-pair rate is the same in both. Intervals overlap, so the
claim the numbers support is "no elevation at wait", not "significantly steadier". This holds
whatever the readout turns out to be worth, because the wait and control windows share the
readout; it is a within-instrument comparison. Sits alongside RESULT 1 (2-layer model: decoded
flips do not beat arbitrary-pair crossings) and RESULT 2 (Huginn: per-loop flips at the
init-seed noise floor on ARC-Challenge and ARC-Easy alike). RESULT 4 is the instrument
limitation: the forcing-suffix readout's phrasing noise (q95 2.46) exceeds every within-trace
change, so 0 of 50 leader changes survive it.

### Machine-constraint check before the next runs (standing rule from 2026-09-09)

phoenix README lines 57-60 record: on this Mac use CPU for analysis, because batch-one forwards
on the 2-layer model take 25 ms on CPU vs 178 ms on MPS and the MPS allocator grows by hundreds
of MB per few dozen forwards (a run was killed for memory). Checked against this model before
running: for DeepSeek-R1-Distill-Qwen-1.5B in float32 the ordering REVERSES, MPS 183 ms vs CPU
296 ms per generated token and 175 ms vs 548 ms per suffix read, about 123 s vs 247 s per
question, so the device conclusion does not transfer to a 1.5B model. The memory warning does
transfer and already bit once (2026-09-09, 188 s -> 1066 s per question, 19 of 20 GB swap):
every new run calls torch.mps.empty_cache() after each question or batch. Batched generation
measured at the same time: 192 ms per token per sequence at batch 1, 104 ms at batch 4.

## 2026-09-09 (evening) — items 1 and 2: ground truth found, probe failed

**Item 2, the hidden-state probe: FAILED for lack of training data, not for tuning.**
Fitted on the 24 ARC-Easy traces recovered into `results/traces_easy` (their stored text
round-trips to exact token ids, 24/24, wait positions intact, so no regeneration was needed;
hidden states cost one forward each, 1.5 s). 17 traces have a parsed answer. Target = the
model's own final answer; training positions = the last quarter of each trace; split by
question. Held-out accuracy by 4-fold cross-validation over questions, against chance 0.25 and a
majority-class baseline of 0.412:

| layer | dims | L2=1 | L2=20 | L2=200 |
|---|---|---|---|---|
| last | 1536 | 0.154 | 0.140 | 0.119 |
| last | 64 (PCA) | 0.114 | 0.108 | 0.098 |
| last | 16 (PCA) | 0.110 | 0.101 | 0.075 |
| mid | 1536 | 0.175 | 0.155 | 0.132 |
| mid | 64 (PCA) | 0.163 | 0.153 | 0.109 |
| mid | 16 (PCA) | 0.115 | 0.109 | 0.091 |

Every cell is BELOW chance; training accuracy is 1.000. With 11-13 training questions the probe
memorises the training answers and, on held-out questions whose answers differ, scores worse than
guessing. Positions within a question are not independent, so the effective n is the number of
questions, not the 1276 positions. A usable probe needs on the order of hundreds of traces, which
this machine cannot generate (below). The probe readout is therefore NOT available as a
replacement for the forcing-suffix readout, and RESULT 4 stands unrelieved.

**Item 1, ground truth: the stated reversals are not answer reversals.**
The wider pattern in `traces.py` finds 5 stated reversals in 4 of the 24 traces. All existing
forcing-suffix reads bracket them, so this needed no new compute. Decoder-noise q95 = 2.46.

| trace | phrase | leaning across it | margins | clears noise | same under all 3 suffixes |
|---|---|---|---|---|---|
| Mercury_7282695 | "Wait, no" | D -> A | 0.08 / 0.01 | no | no |
| Mercury_7007858 | "that's not right" | A -> A | 8.57 / 8.53 | yes | yes |
| Mercury_SC_LBS10272 | "Wait, no" | A -> A | 2.47 / 2.59 | yes | yes |
| Mercury_SC_LBS10272 | "Wait, no" | A -> A | 3.03 / 3.02 | yes | yes |
| Mercury_7126613 | "Wait, no" | C -> C | 7.86 / 7.86 | yes | yes |

Reading the surrounding text, none of the five announces a different ANSWER. They correct an
intermediate claim ("Wait, that's not right. Heat flows from a hotter object to a cooler one"),
a geometric statement ("Wait, no, the tilt is the same for both hemispheres"), or a misreading of
the question ("Wait, no, the question is asking which two behaviors"). The one case where the
decoded leader moves (D -> A) has margins of 0.08 and 0.01, far below the 2.46 noise floor, and
occurs in a trace with no parsed final answer.

So on the only ground truth this corpus offers, the instrument behaves correctly: no answer
reversal is stated and none is decoded, with the four stable cases sitting well above the noise
floor and agreeing under all three suffixes. What remains UNTESTED is the instrument's
sensitivity, because the corpus contains no stated answer reversal to detect.

**This reframes RESULT 3 rather than contradicting it.** "Wait" in this model marks the
correction of an intermediate claim, not a change of the answer. The steadiness of the decoded
leaning across wait windows is then the correct reading, not a failure of the readout. The
write-up should state RESULT 3 this way and cite these five cases as the reason.

**Hardware, for the record.** Two attempts to generate longer traces failed on this 16 GB
machine: float32 with batch 4 at 1024 tokens filled 27 GB of swap (38 min for one batch), and
float16 with batch 2 at 768 tokens ran at 410 s per trace and truncated every math trace before
its answer (4/4 unparsed, 0 waits). Generation of long traces from a 1.5B model is not viable
here alongside normal desktop use; hidden-state extraction and all analysis are (1.5 s and
seconds respectively). The GSM8K four-option set (`data/gsm8k_options.json`, 200 questions,
`mathopts.py`) is built and unused, ready if a machine with more memory becomes available.
