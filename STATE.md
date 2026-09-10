# State snapshot — 2026-09-10 (updated after RESULT 12)

An index into `LOG.md`, not a second record. `LOG.md` is the chronological account and the only
place numbers live; this file exists so a new session can pick up without re-reading it. Anything
here that contradicts `LOG.md` is wrong. Regenerate freely.

## The question, and where it actually stands

`BRIEF.md`: when a latent reasoning model appears to backtrack, is it REVISITING an earlier
latent state, OVERRIDING it, or is the change only in the decoder?

Answer so far: **largely the decoder**. Revisit-vs-override was never tested and cannot be on
these models, because the phenomenon it presupposes is not there. The brief named
"only in the decoder" as the thing to rule out first; we ruled it in.

The positive claim the evidence supports: **latent steps are not deliberation between answers.**
These models compute a structure, bind an answer, and then sharpen it, rather than weighing
candidates and revising.

## Results (all in LOG.md with numbers and conditions)

| # | claim | strength |
|---|---|---|
| 1 | Graph model: decoded leader changes do not beat an arbitrary-pair null | solid, 419 graphs, 2 seeds |
| 2 | Huginn: same, on ARC-Challenge and ARC-Easy; Cui & Ye's event rate reproduces at 66-68% (they say 32%) but sits at the noise floor; their accuracy benefit does not replicate | 50 questions per dataset — THIN |
| 3 | Text model: "wait" is not where answer leaning changes; it is steadier there | 24 traces, within-instrument comparison |
| 4 | Forcing-suffix readout's phrasing noise exceeds any within-trace change | instrument limitation |
| 5 | Rerunning RESULT 1 in the causal-Jacobian basis gives the same verdict | closes the basis objection |
| 6 | The readout follows documented answer changes in downloaded R1 traces (+6.05 vs +0.29 null) | 19 of 198 traces, 8 of 11 truncated |
| 7 | The losing candidate IS held, but only late and never near-tied (z +1.5 to +3.4 vs control ~0; winner ahead by 4-6 spreads) | closes the superposition objection, positive |
| 8 | Induced reversals on the graph model: detector fires 63-73% when the answer moved vs 8-17% when it did not | calibrated on the substrate |
| 9 | Huginn: detector NOT specific — fires 78% when the answer moved, 83% when it did not, and 34.5% with no intervention at all | quantifies the critique; 29 pairs |
| 10 | Mean last-change point ~0.39 in all four cells — but the addendum shows the mean hides a spread. Graph: the answer step confirms the last latent's leader on 96-99% of graphs and the margin triples after. Huginn: answer forms over loops ~4-16, sharpens after; 10-16% of answers still move after loop 16 | descriptive, NOT a constant; do not lead with 0.39; bf16 caveat tested in RESULT 12 and withdrawn |
| 11 | K sweep (16/30/64, same 50 ARC-Easy questions, bit-identical prefixes): last change at loop 9-10 at every K, fraction 0.62/0.39/0.24 — RESULT 10's 0.39 was loop 10 / 29. Huginn converges by ~loop 24; loops after are inert (1-3% change per transition, margin flat at ~0.6). Accuracy peaks 0.66 at loop 13, converged 0.52 (n=50, overlapping) | settles fraction-vs-loop; convergence is Geiping et al.'s design, not new; the mid-formation accuracy peak is the one lead, untested at n |
| 12 | fp32 option readout on the identical trajectories: every Huginn number stands (changes per question 5.5->5.2, event rate 0.66->0.66 / 0.68->0.76, last change loop 10->10, final answers agree 49/50 and 47/50). The early near-ties are real, not bf16 artefacts — corrects the RESULT 10 addendum. ARC-Easy accuracy peak persists: 0.66 at loops 13-14 vs 0.54 converged | precision objection closed; the accuracy peak is the one live lead, n=50 |

Asymmetry to preserve in any write-up: RESULT 1 carries a calibrated instrument (RESULT 8);
RESULT 2 does not (RESULT 9). Both point the same way; only one is instrument-backed.

## Next

RESULT 10 addendum (2026-09-10, `src/lattrack/commitment_dist.py`, `results/commitment_dist.md`):
the 0.39 is not a constant. Graph model: last change spread over latent transitions, answer step
quiet (96-99% confirm), margin x3 after. Huginn: unformed (~loops 0-8) -> forming (8-16) ->
sharpening; half the answers differ from final at loop 8, 10-16% at loop 16; 56-70% of individual
transitions have a margin under 0.13 in either precision (RESULT 12: real near-ties). Early-exit accuracy on ARC-Easy peaks at loop 14 (0.64) and
ends at 0.50 (n=50, intervals overlap). Positive frame that survives: build -> bind -> sharpen -> stop (RESULT 11).

Candidate runs, all on this Mac, none started (user to choose):
- fp32 option readout rerun: DONE (RESULT 12). Nothing moved; the bf16 caveat is withdrawn.
- K sweep: DONE (RESULT 11). Loop count, not fraction; converged by ~loop 24; accuracy peak at
  loops 13-14 (0.66) vs converged 0.52-0.54, n=50, same in fp32 — the one lead. Next test: the
  same lens on 200 ARC-Easy questions at K=30 (~1.5 h at 21.6 s per trajectory), then decide.
- CODI (`zen-E/CODI-gpt2`, official, MIT, 406 MB, model class to vendor) for the third architecture
  family, if the paper goes ahead. Expect a phase description, not a 0.39.

Venue thinking unchanged: strong workshop plus arXiv, BlackboxNLP first, ICLR 2027 interpretability
workshop in parallel, short Alignment Forum post alongside. Open decision: post now vs CODI + paper.

## Where things are

- `src/lattrack/` — `lens.py` (per-step readout + nulls), `jlens_flips.py` (two bases),
  `superposition.py`, `induced.py`, `huginn_lens.py`, `huginn_induced.py`, `wait_lens.py`,
  `validate_readout.py`, `reversals.py` (corpus filter), `traces.py`, `mathopts.py`,
  `probe_lens.py` (probe failed, below chance on 17 traces), plus 8 modules copied from phoenix.
- `bases/` — causal-Jacobian bases copied from phoenix, fitted on these same checkpoints.
- `results/` — one directory per run; every number in LOG.md points here.
- `results/reversals_full/validation_set.jsonl` — 198 verified answer-reversal traces, self-contained.
- `ckpts/seed0, seed1` — gitignored; the paper's accepted seeds (sha256 in LOG.md).
- `vendor/` — gitignored, pinned to 72d22af7.
- `lit-sweep.md` — plus a 2026-09-09 appendix: the revisit/override gap is still open, nobody has
  contested Cui & Ye, and prior art exists for the logit-lens-blindness criticism.

## Machine constraints (this bit up front, it cost hours)

16 GB M1 iMac, MPS, no CUDA. Forward passes are fine; long generation from a 1.5B model is not
viable alongside a normal desktop. Three separate runs died in swap. Rules learned:
`torch.mps.empty_cache()` inside read loops, not just per item (`DynamicCache.crop` does not
release it); float32 only where precision is load-bearing (the readout), float16 elsewhere;
estimate cost from the MEDIAN item, not the first; check the compressor figure in Activity
Monitor, not "free memory". phoenix's README device advice (CPU over MPS) is for its 2-layer
model and does NOT transfer to a 1.5B one — measured, MPS is 2x faster there.
