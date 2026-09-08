# lattrack — brief

Working name, not final.

## The question

When a latent reasoning model appears to backtrack (its decoded leaning changes mid-computation), is it *revisiting* an earlier latent state, or *overriding* it with further computation? A third possibility to rule out first: the change exists only in the decoder, not in the model.

Nothing in the literature distinguishes revisit from override. The closest existing claims are readout-only (decoded belief flips, Cui & Ye 2026) or test step necessity without asking whether a specific earlier state is re-read.

## Background to read

- `lit-sweep.md` — literature map, written for an earlier, wider version of this project. Its factual content stands. Its recommendation to make the looped vs feed-back architecture comparison the headline is superseded: that is at most a small subsection now.
- RRR ("Read, Replace, Never Rewrite", arXiv 2026-07-28) — the researcher's prior paper on continuous thoughts. Its finding: removing branch directions from intermediate thoughts, or the whole linearly decodable identity subspace (INLP), leaves the answer unchanged; only a final-step edit flips it. The repo at `~/Developer/RRR` is the record; it is not runnable on this Mac (vendor code and checkpoints absent).
- phoenix (the follow-on, `~/Developer/phoenix`; query-key cells run 2026-09-06/07, n=100, both seeds) — removing from the last intermediate thought the directions each layer-2 head uses to attend to the answer edge cuts that attention to a fraction and leaves the answer at the fallback rate. That is the override-flavored data point. Phoenix is the source of the pilot's infrastructure files (hooks, nulls, pinned serialization, bit-exact test); nothing else is shared.

## Current leanings (none locked)

- Keep the project short. Not months.
- Task dependence looks like the interesting axis. Which task is undecided (graph reachability with dead ends, as in RRR, is one candidate).
- A text CoT arm (same test at "wait" moments in a reasoning model) is being considered, not decided.
- Channel questions (keys vs values etc.) only matter if revisit shows up. Not a starting hypothesis.

## Immediate scope: the pilot

Before the main experiments, check whether decoded flips are real on the chosen model and task:

1. Decode the model's leaning at each latent step; mark where the leader changes.
2. For each flip: how far apart were the options? Estimate what gap noise alone can flip.
3. Can a late step be reproduced by sharpening an early step? (If yes, it's a gain ramp, not a reordering.)
4. For flips that look real: patch the flip-step state and check whether the output follows.

What the pilot can yield: almost nothing real → critique result, rethink model/task; plenty real → curated case list, main project starts from it; mixed → tells us where real flips cluster, which shapes task and depth choices.

Side products: which decoder to trust, whether patching works in the pipeline, readout noise level, cost per run.

## Still open

Model to start on, task, decoder, what counts as a flip, whether the text arm is in, repo name. Raise these; don't settle them.

Decisions taken so far, with dates, are in `LOG.md` (2026-09-08: pilot runs first on the 2-layer model with a logit lens; text arm parked).

See `WORKSTYLE.md` for how to work.
