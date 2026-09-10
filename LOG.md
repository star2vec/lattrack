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

### RESULT 5: the negative result is not an artifact of the readout basis

The objection (user, 2026-09-09; prior art 2604.09885, 2604.06374, 2608.25347): RESULT 1 read the
leaning in the token-embedding basis, and the answer-relevant activity may live elsewhere —
especially in a from-scratch model, where 2604.06374 reports superposition survives a logit lens.
Test: capture each thought ONCE and read it two ways, then run the identical flip, null and noise
analysis on both. `src/lattrack/jlens_flips.py`, bases copied from phoenix (fitted by its
`fit_jlens.py` on train[500:2500] against these same checkpoints; `bases/README.txt`), 419 test
graphs, base serialisation + 3 edge-only redraws, both accepted seeds. Files
`results/seed{0,1}/jlens_{rows,summary}_all.json`.

Crossing rate per transition, real pair (target, decoy) vs arbitrary pair vs matched pair:

| transition | seed0 embedding | seed0 jacobian | seed1 embedding | seed1 jacobian |
|---|---|---|---|---|
| last | 0.320 / 0.317 / 0.283 | 0.317 / 0.370 / 0.369 | 0.368 / 0.365 / 0.386 | 0.375 / 0.368 / 0.416 |
| last-1 | 0.325 / 0.411 / 0.324 | 0.384 / 0.425 / 0.440 | 0.382 / 0.394 / 0.386 | 0.389 / 0.442 / 0.369 |
| last-2 | 0.290 / 0.396 / 0.227 | 0.392 / 0.456 / 0.454 | 0.364 / 0.419 / 0.261 | 0.341 / 0.401 / 0.395 |

**In the causal-Jacobian basis the real pair still crosses no more often than an arbitrary pair,
at any transition, on either seed.** Every jacobian cell has real <= one or both nulls. The
verdict of RESULT 1 is unchanged by the basis.

The Jacobian readout is not a worse instrument — it is a sharper one on every axis we can
measure. Median |gap| at flips 1.88 (seed0) / 2.26 (seed1) vs 1.23 / 1.55 in the embedding basis;
flips clearing the serialisation q95 0.412 / 0.447 vs 0.327 / 0.346; leader sequence identical
across redraws 0.858 / 0.855 vs 0.848 / 0.842. So the null result is not "the readout is too
blunt to see anything": a demonstrably sharper, causally-defined readout sees the same thing.

Scope, stated precisely. This closes the BASIS form of the objection: it is not that we looked in
the wrong linear directions. It does NOT close the SUPERPOSITION form. Both readouts take an
argmax over candidates, which discards the case where two answers are held at comparable
strength; the small margins we keep measuring (median |gap| 1.2-2.3 against a serialisation q95
of 0.65-3.9) are consistent with candidates being held near-equally rather than swapped between.
But "held simultaneously without a leader change" is not the claim under test: the literature
claim (Cui & Ye's, and the readout-only genre generally) is about decoded belief FLIPS, and those
flips do not survive either basis.

## 2026-09-09 (late) — downloaded reversal corpus and the filter that finds real answer changes

Source: `uzaymacar/openr1_math_backtracking_dataset` (193,767 DeepSeek-R1 traces over
OpenR1-Math-220k, MIT, ungated, one 3.73 GB JSON file, no shards). Downloaded by byte range for
development (220 MB = 11,585 traces at ~13 MB/s), then in full. `src/lattrack/reversals.py`.
The dataset's own `has_backtracking` flag is true on 42.6% of traces here (matching its card's
42.7%) and is NOT the label we need: it marks any doubling back, and ReasonOps (2605.29192,
Table 7) puts the global, answer-changing share at 1.6%.

Recomputed label: a trace COMMITS whenever it writes \boxed{...}; a REVERSAL is a trace whose
last commitment differs from an earlier one. Four versions were needed, and each failure is a
different way of manufacturing a reversal out of formatting:

| version | rule added | reversal rate (of 11,585) | what was still wrong |
|---|---|---|---|
| v1 | last commitment differs from an earlier one | 19.35% | multi-part answers: "the ship is \boxed{10} km/h and the river is \boxed{4}" |
| v2 | exclude tail blocks holding several values; require 400 chars of separation | 4.38% | equal values written differently (⌊(n²−2)/2⌋ vs ⌊n²/2⌋−1), empty boxes, case splits |
| v3 | numeric commitments only; collapse whitespace; drop empty boxes | 0.71% | lists of answers parsed as numbers |
| v4 | sub-question markers in the problem; comma bug fixed | 0.07% | usable |

The v3 bug is the one to remember: commas were stripped to normalise thousands separators, which
silently turned the LIST "0, 1, 2" into the NUMBER 12. Four of that version's twelve survivors
were traces where the model was only deciding how to format a multi-answer list. Same class of
error as trusting the dataset flag, reached from the other direction. Now "1,000" -> num:1000
while "0, 1, 2" -> list:0,1,2 and is excluded.

Yield on the 220 MB slice: 8 candidates from 11,585 traces (0.07% of all, 0.16% of flagged),
7 of 8 also carry the dataset flag. All 8 were hand-read: 6-8 are genuine changes of the
committed answer (a recount giving 11 rather than 15; choosing 416 over 420 on significant
figures; settling on 69 after two conflicting derivations; two traces weighing coding theory
against Katona's theorem and ending on 11 after writing 12). The rest are borderline, e.g. a
problem with a missing input where the model guesses.

Honest limits of this filter. (a) Most survivors are the model CHOOSING between two values it
derived, not announcing "I was wrong" — fine for validating a readout, since the committed answer
demonstrably moves, but they should not be called self-corrections in the write-up. (b) The rate
is now BELOW ReasonOps' 1.6%, so the filter is over-strict; that is the right direction for a
validation set, where precision beats recall. (c) `stated` (an explicit phrase near the switch)
survives on none of the 8, so the phrase and the value change are close to disjoint signals here.

### Full corpus: 198 validated answer-reversal traces (the validation set the method needed)

Filter v4 over all 193,767 traces (`results/reversals_full/`): 82,642 carry the dataset's
`has_backtracking` flag (42.65%); 33,737 excluded as multi-part; 22,901 excluded as non-numeric;
**198 traces are genuine answer reversals** (0.10% of all, 0.16% of flagged). Of those, 21 also
state the switch in words. Final answer matches the gold solution in 115 of 198. Median trace
21,509 characters, median switch at 90% of the trace — reversals happen LATE, which the readout
design must account for (dense reads near the end, not a uniform scan).

Hand-read 11 of the 198 (all 6 shown of the stated subset, 5 random unstated). The stated subset
is essentially exact ground truth:
- "in my initial answer, I wrote 12. That must be an error. Therefore, the correct answer is 2"
- "the previous final answer boxed as 9 is incorrect. It must be a mistake" -> 3
- "1344 is invalid. Therefore, the correct answer is 672" (from 1343)
- 144 -> 133 twice (subtracting the empty and single-element subsets), 1944 -> -1944 (sign)
Of the 5 unstated: 3 clearly genuine (including "My mistake earlier was a typo in the numerator",
134,217,528 -> 134,215,680 — which the `stated` regex missed, so that flag UNDER-counts),
1 borderline (2.77 -> 83/30, exact-vs-rounded preference), 1 unclear from the context window.

`results/reversals_full/validation_set.jsonl` (5.9 MB, 198 rows) is self-contained: problem,
gold solution, full trace text, the ordered list of commitments with character offsets, the
switch point, and the stated phrase where present. It does not depend on the 3.73 GB download
(gitignored; re-fetchable from the URL in reversals.py).

**What this unblocks.** RESULT 4 said the method is unvalidated: we had never shown the checks can
detect a real answer reversal, so "no backtracking above noise" was confounded with "the checks
are blind". These 198 traces are the missing positive control, and the readout runs teacher-forced
over given text — forward passes only, no generation, which is the operation this machine can
afford (1.5 s per trace for hidden states). The same corpus also fixes the probe's training-data
shortage that sank item 2 (below-chance held-out accuracy on 17 traces).

## 2026-09-09 (23:49) — readout validation STOPPED at 4 of 21; preliminary signal is positive

`src/lattrack/validate_readout.py` over the 21 stated traces of the validation set. Readout for
maths traces: lean(t) = log P(later answer) - log P(earlier answer), each scored as its full token
sequence after a forcing suffix that opens the answer box, teacher-forced over the given text.
Null: two numbers appearing in the trace that were never committed as answers, read identically.
fp16 (measured cached-vs-full error 0.008-0.012, far below a full change of answer).

Completed 4 of 21 (`results/validation/validation_stated.jsonl`). Crossing analysis — does the
lean start below zero (favouring the earlier answer) and rise above it?

| from -> to | clean | real: first / last | crosses | null: first / last | crosses |
|---|---|---|---|---|---|
| 750001 -> -1. | no (malformed) | -2.68 / -13.28 | yes (transient) | 1.06 / -9.04 | no |
| 1 -> 13 | yes | -4.51 / -0.04 | no | -5.58 / 0.61 | yes |
| 144 -> 133 | yes | -6.39 / 3.25 | yes | 0.59 / 1.67 | no |
| 144 -> 133 | yes | -3.86 / 6.32 | yes | 1.12 / 0.41 | no |

Real pair crosses 3/4, null pair 1/4. On the 3 clean traces: real 2/3, null 1/3. All four start
below zero, so the read grid reaches early enough to see a pre-change state.

**Preliminary reading, n=4, not a result.** The readout is not blind: on two clean traces it moves
from favouring the earlier answer to favouring the later one, ending +3.25 and +6.32, while the
null pair does not move. That is the shape a working instrument must produce, and it is the first
positive evidence in this project that the method can see a real answer change. It is four
traces; nothing rests on it until the run completes.

Design note found in the data: the third trace's lean already favoured the corrected answer BEFORE
the written commitment (+4.11 both sides of it). The boxed value marks where the model WRITES the
new answer, not where it changes its mind, so the before/after contrast understates the effect and
the crossing analysis (adopted here, user agreed) is the right primary. Where the internal
crossing sits relative to the written commitment is itself worth reporting later.

**Why it stopped: the machine, for the third time today.** After 4 traces the run stalled — 25
minutes with no new row, process at 6.7% CPU and 10 MB resident (swapped out), physical memory
103 MB unused, swap 22.5 of 23.5 GB. The remaining 17 traces are 2-14x longer than the four that
completed (10k-56k characters). Mitigations applied and insufficient: torch.mps.empty_cache()
every 4 read positions (DynamicCache.crop does not release the allocator), token cap 7000 -> 5000,
stride 64 -> 96. The workload is forward passes over 3-5k-token contexts with a live KV cache, on
16 GB shared with a full desktop; it is not viable while the machine is loaded. Resumable by
question index: rerunning the same command continues from row 5.

### RESULT 6: the readout DOES follow a documented change of answer. RESULT 4's confound is reduced.

`src/lattrack/validate_readout.py` over the 21 stated traces of the downloaded validation set;
19 completed (2 skipped, no valid null candidates). Readout: lean = log P(later answer) −
log P(earlier answer), each scored as its full token sequence after a forcing suffix that opens
the answer box, teacher-forced. Null: two numbers appearing in the trace that were never
committed as answers. fp16, MPS. Files `results/validation/validation_stated.jsonl`.

Split by candidate quality, since the audit found the stated subset is not uniform: CLEAN = both
answers plain numbers differing by more than 5% (n=11); WEAK = rounding/formatting changes such
as 1.06 -> 1.061 (n=8).

| subset | lean change first->last, real | same, null | crosses (real / null) | final lean favours later answer |
|---|---|---|---|---|
| clean, n=11 | **+6.05** | **+0.29** | 4 / 3 | 9/11 vs 7/11 |
| weak, n=8 | −1.17 | −1.13 | 2 / 1 | 5/8 vs 2/8 |
| all, n=19 | +4.47 | −0.14 | 6 / 4 | 14/19 vs 9/19 |

**The headline is the movement, not the crossing count.** On clean cases the real pair moves
toward the later answer by a median of +6.05 log-units while the null pair moves +0.29 — a
twenty-fold difference on the same traces, same reads, same instrument. On weak cases (rounding
changes) real and null are indistinguishable (−1.17 vs −1.13), which is the right behaviour: there
is no real change of answer to track.

**Why the crossing counts are weaker than the movement, and it is my fault.** 8 of the 11 clean
traces were truncated to the last 3,000 tokens — the memory compromise made at 01:54 to stop the
run thrashing. For a truncated trace the first read sits very late in the original reasoning,
often after the model has already decided, so it starts ALREADY favouring the later answer and
cannot show a crossing. 5 of the 8 truncated traces start above zero; all 3 non-truncated traces
start below it. On those 3 (full early context): 2/3 cross vs 1/3 for the null, lean change
median +9.64 vs +1.08 for the null. The crossing test needs the early window that truncation
removed; the movement statistic survives truncation and is the one to quote.

One clear failure worth keeping: 1944 -> −1944 (a sign flip) moves the WRONG way, −9.27 to
−13.08. The readout does not follow a change of sign on an otherwise identical magnitude.

**What this does to the project.** RESULT 4 said the method was never shown to detect a real
answer reversal, so RESULT 1 and RESULT 2 ("decoded flips do not beat an arbitrary-pair null")
could not be separated from "the checks are blind". They now can, on this model and readout: given
a documented answer change, the leaning moves and the null does not. The two negative results
therefore carry their intended weight — with the scope stated honestly: validated on a 1.5B text
reasoning model with a forcing-suffix readout over numeric answers, not on the latent models
themselves, where no corpus of documented reversals exists to validate against.

### RESULT 7: the losing candidate IS held — but late, and never as a near-equal rival

`src/lattrack/superposition.py`, 418 test graphs, both accepted seeds, both readouts. Answers
the objection that an argmax cannot see two answers carried at once, so an absence of flips might
hide a sustained tie. Asked positively, with the matched control the task supplies: in ProsQA the
decoy is UNREACHABLE like every distractor node, and differs from them only by being named in the
question. So z(decoy) = standard deviations above the mean of the unreachable non-candidates, and
z(control) does the same for one unreachable non-candidate drawn per graph (it must sit near zero,
and it does: +0.02 to +0.12 everywhere).

Final thought position (`last`), by basis and seed:

| | z decoy | z control | z target | both candidates in top 2 | margin, in spreads |
|---|---|---|---|---|---|
| embedding, seed0 | +3.11 | +0.09 | +10.00 | 0.598 | +6.43 |
| embedding, seed1 | +3.40 | +0.02 | +9.26 | 0.725 | +5.88 |
| jacobian, seed0 | +2.34 | +0.12 | +6.57 | 0.512 | +4.24 |
| jacobian, seed1 | +1.50 | +0.07 | +5.75 | 0.244 | +4.22 |

Three findings, consistent across bases and seeds.

1. **The decoy is genuinely represented.** It sits 1.5 to 3.4 standard deviations above its own
reference class while a matched unreachable node sits at zero. Being named as a candidate, not
reachability, is what elevates it. So the model is not simply ignoring the answer it does not give.
2. **It is not a near-tie.** When both candidates are up, the winner leads by 4.2 to 6.4 spreads.
The two are co-present, not competing on equal terms.
3. **It happens late.** At the earliest positions z(decoy) is 0.0 to 1.2 and both candidates are
in the top two in 0-7% of graphs; at the final thought that reaches 24-73%. Across all positions
both are top-two only 9-24% of the time, and in NO graph (0.000 in all four runs) at every position.

**This closes the superposition objection with a positive account rather than an absence.** The
reason there are no flips is not that a tie is hidden from an argmax. It is that for most of the
computation there is no second candidate to flip to — the decoy is not elevated at all — and by
the time it appears, the winner is already several standard deviations clear. A flip needs two
candidates, present together, and close; on this substrate those three conditions never coincide.
It also matches RRR's own finding that identity is bound late, now measured on the losing side.

Scope: this is the 2-layer graph model. It says nothing about Huginn, where no equivalent matched
control exists (the four ARC options are all "named candidates"; there is no class of nodes that
differ only by not being named).

### RESULT 8: the detector is sensitive ON THE LATENT MODEL ITSELF (induced reversals)

RESULT 6 validated the readout on a text model, because no corpus of documented latent reversals
exists. This validates it on the substrate the negative results are actually about, by
MANUFACTURING the reversal. `src/lattrack/induced.py`, 120 test graphs, both seeds, both readouts.

Construction: a different-structure donor (training graph, same K, candidates disjoint from the
recipient's) has its thought blended into the recipient at the LAST intermediate pass, at strength
alpha. Early positions keep the recipient's own state, so there is a genuine "before"; the
injected pass and everything downstream carry the donor. The pair tracked is (donor's target,
recipient's target). Ground truth is the model's own output: did the answer actually move?

Dose-response, seed0 (n=120), with the contrast that matters — among graphs given the SAME
intervention, those whose answer moved vs those whose answer did not:

| alpha | answer moved | fires when it moved (emb / jac) | fires when it did not (emb / jac) |
|---|---|---|---|
| 0.00 | 0% | — | 0.300 / 0.225 |
| 0.25 | 5% | 0.500 / 0.667 | 0.289 / 0.219 |
| 0.50 | 23% | 0.407 / 0.519 | 0.301 / 0.258 |
| 0.75 | 72% | 0.517 / 0.678 | 0.273 / 0.273 |
| 1.00 | 90% | **0.630 / 0.731** | **0.083 / 0.167** |

Seed1 at full strength: answer moved on 86%; fires when moved 0.718 (emb) / 0.709 (jac), when not
0.412 / 0.412.

**The detector fires on a real mid-trajectory answer change in this model.** At full strength,
63-73% of graphs whose answer genuinely moved show a crossing, against 8-17% (seed0) of graphs
that received the identical intervention but whose answer did not move. The contrast is within
the same intervention, so it controls for the global perturbation an injection causes — which the
arbitrary-pair null cannot, because injecting a foreign state moves every node (the null pair
crosses on 44-68% regardless of alpha, and is uninformative here for exactly that reason).
Seed1's separation is weaker (0.71 vs 0.41) but the same direction.

Sensitivity floor: detection is reliable at alpha >= 0.75 and marginal at 0.5, where only 23% of
answers move at all. So the honest statement the negative results can now carry is: a natural flip
of the size we can induce would have been caught roughly two times in three, and we observed none
above the arbitrary-pair rate.

Design error worth keeping. The first version injected at ALL intermediate passes (RRR's
donor_intermediate). That replaces the whole trajectory, so the donor's answer leads from
position 0 and there is no "before" to cross from: at alpha=1 the answer moved on 95% of graphs
while the detector fired on 35%, BELOW the 70% null. Read naively that is "the detector is blind".
It was the construction that was wrong, not the detector. The sensitivity curve is what exposed
it — a single yes/no at full strength would have produced a confidently wrong negative.

### RESULT 9: on Huginn the crossing detector is NOT specific — and that quantifies the critique

`src/lattrack/huginn_induced.py`, 29 question pairs, ARC-Easy, K=32, bf16/MPS. Same logic as
RESULT 8: blend a donor question's recurrent state into the recipient at the LAST prompt position,
sustained from loop 19 to the end (a single blended loop was fully recovered within the remaining
loops — 0 of 6 answers moved, `rows_single_loop.jsonl`), sweep alpha, ground truth = did the
model's own answer letter change.

| alpha | answer changed | = donor's answer | fires when it moved | when it did not | null pair |
|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.000 | — | **0.345 [0.172, 0.517]** (n=29) | 0.379 |
| 0.5 | 0.621 | 0.483 | 0.722 (n=18) | 0.545 (n=11) | 0.414 |
| 1.0 | 0.793 | 0.621 | 0.783 [0.609, 0.913] (n=23) | 0.833 (n=6) | 0.379 |

The induction works: at full strength the answer changes on 79% of pairs and lands on the donor's
answer on 62%. **The detector does not discriminate.** When the answer moved it fires on 78%;
when the same intervention left the answer unchanged it fires on 83%. It responds to perturbation,
not to answer change. Contrast RESULT 8 on the graph model: 0.630 when moved vs 0.083 when not.

**The number that matters is the top row.** With NO intervention at all, a per-loop leader-change
detector fires on 34.5% of questions. Over 32 loops with four options, the argmax crosses by
chance in a third of questions. That is the floor any flip count on this model must clear, and it
is measured by intervention rather than argued.

Consequences, stated carefully.
- RESULT 2's Huginn negative CANNOT be backed by a calibrated instrument the way RESULT 1's can.
  The honest form is: flips on Huginn do not exceed the arbitrary-pair rate, AND the detector is
  dominated by noise on this model. Both point the same way — no evidence of real backtracking —
  but the second is a statement about the instrument, not the model.
- It sharpens the critique of the readout-only genre. Counting argmax changes across recurrent
  steps is not a measurement on this architecture: a third of questions produce one with nothing
  happening, and a deliberate answer change is indistinguishable from an inert perturbation.
- Cui & Ye report backtracking on 32% of instances. Our measured false-positive rate for a
  leader-change detector with no intervention is 34.5%. Their definition is stricter (>=3
  consecutive steps each side) so the two are not directly comparable, and we measured THEIR
  definition at 66-68% (RESULT 2) — but a headline rate sitting at the same order as a measured
  noise floor is the comparison a reader should be given.
- Why the two models differ: the graph model has 3-4 steps and two candidates; Huginn has 32
  loops and four options, so many more chances for a chance crossing, on a trajectory whose
  init-seed noise (q50 0.12) already equals its margins at flips (q50 0.12, RESULT 2).

Scope: 29 pairs, intervals are wide, one intervention site (last position, loops 19+). A stronger
intervention at every position would need prompt-length matching and was not run.
