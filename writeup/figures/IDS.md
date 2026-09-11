# Figures: what each one reads, and how its example was chosen

Every script reads `results/` directly and writes `out/<name>.png` and `.svg`. Regenerate all six with
`writeup/figures/make_all.sh`. Examples are chosen by a stated rule, applied in the script, so the
figure cannot drift from the log and the choice is not by eye. Result numbers (R1…R13) are in `LOG.md`.

| figure | reads | example and rule |
|---|---|---|
| `fig1_stories` | nothing (schematic) | — |
| `fig2_graph` | `results/seed0/lens_rows_all.jsonl` (base variant, per-node logits), `results/seed0/lens_summary_all.json` (`crossing_rate_forward_names`); node sets via `Prompt.from_sample` on the test split | **test graph gi=17** (seed0): first graph in index order with K=4, answered correctly, exactly one leader change, D→T, at a latent transition, decoy in the top three nodes at the last latent position, ≥10 unreachable non-candidates. Target 24, decoy 25, root 1; leaders D,D,T,T,T. Panel B: depth-4 transitions, n=217 of 419 |
| `fig3_calibration` | `results/seed0/induced_rows.jsonl` (R8; n=120 graphs, α ∈ {0,.25,.5,.75,1}, embedding readout); `results/huginn_induced/summary.json` (R9; n=29 pairs, α ∈ {0,.5,1}) | no example; all rows |
| `fig4_huginn` | A: `results/huginn_easy/lens_rows.jsonl` conditions base/init_1/init_2 (K=30); B: `results/huginn_easy_K64/lens_rows.jsonl` base (n=50, bf16; R11) | A: **LEAP__7_10346** (correct B): among K=30 questions where all three restarts end on the correct letter with 3–8 changes each, all before loop 20, the one with the LEAST overlap of change-loops (Jaccard 0.09). Restart statistics quoted: same final letter 96%, identical change-loop set 8% (all 50 questions). B: onset shading = median first loop at which a letter is the next token. Fractions 0.62/0.39/0.24 = mean last-change loop ÷ (K−1) from R11 |
| `fig5_text` | A: `results/wait/wait_rows.jsonl`, `wait_summary.json` (`decoder_noise_abs_gap_change.q95`; R3/R4); B: `results/validation/validation_stated.jsonl` (R6) | A: **Mercury_7007858** (two waits, ends on A, correct A): rule = every wait window has no 4-way leader change and a swing in the answer-vs-top-rival leaning below the phrasing-noise q95 (2.46). 9 of the 12 wait traces with a parsed answer satisfy it; this one has two waits and a literal step correction. **Not** Mercury_SC_LBS10272 (the literal "Wait, no, actually"): its third wait coincides with a 3.5-logit hardening of the answer it already had — no leader change, but a swing above q95. B: **idx 15** (2009 → 4019, "I made a mistake" at token 1576 of 3000, truncated at 3000): largest clean before/after swing (−3.2 → +0.2 mean, reaching +2.7) with a flat control pair; alternatives with a literal "Wait, no": idx 20 (1/10 → 3/10) and idx 6 (16 → 11), smaller swings |
| `fig6_checklist` | hand-coded from `LOG.md` results; text model id read from `src/lattrack/wait_lens.py` | — |

Readout definitions used in the figures: graph model — node score = embedding readout of the thought
(`wte @ thought`), z-scored per position against the unreachable non-candidates (R7's reference class);
Huginn — option probability = softmax over the four letter logits (`derived.probs`), leader = argmax;
text — panel A leaning = logit(final answer) − max other option under a forcing suffix, mean of three
suffixes; panel B leaning = log P(later answer) − log P(earlier answer) under the forcing suffix.
