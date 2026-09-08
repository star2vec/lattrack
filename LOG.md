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
