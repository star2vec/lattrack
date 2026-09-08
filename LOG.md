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
