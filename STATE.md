# State snapshot — 2026-09-10

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
These models compute a structure and bind an answer late, rather than weighing candidates and
revising.

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

Asymmetry to preserve in any write-up: RESULT 1 carries a calibrated instrument (RESULT 8);
RESULT 2 does not (RESULT 9). Both point the same way; only one is instrument-backed.

## Next, agreed

**Commitment-point analysis** (`src/lattrack/commitment.py`, to be written). For every model and
question, the last step after which the leader never changes again, as a fraction of the
trajectory; then whether it moves with task difficulty (ARC-Easy vs ARC-Challenge on Huginn,
K=3 vs K=4 on the graph model). All data is already on disk; no new runs. If the commitment point
is early and difficulty-invariant, that is a positive finding to lead with.

Then, if the paper is going ahead: add **CODI** (`zen-E/CODI-gpt2`, official, MIT, 406 MB, needs
its repo's model class vendored). That gives all three architecture families — feed-back (CODI),
recurrent-depth (Huginn), from-scratch two-layer (ours) — which is the difference between "two
models" and a paper. Optional after that: widen Huginn from 50 to 200 questions; finish the text
validation (179 traces remain); COCONUT ProsQA third-party checkpoints exist if wanted.

Venue thinking: strong workshop plus arXiv, BlackboxNLP first, ICLR 2027 interpretability
workshop in parallel, short Alignment Forum post alongside. Not a main conference; what would
change that is a model that demonstrably changes its mind.

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
