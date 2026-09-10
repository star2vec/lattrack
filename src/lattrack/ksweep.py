"""K sweep: is the last-change point a loop count or a fraction of the trajectory?

Huginn's recurrent block receives no signal of how many loops it will run, so a
genuine "commitment at 0.39 of the trajectory" (RESULT 10) is impossible; only a loop
count can be invariant. Same 50 ARC-Easy questions at K=16, 30, 64 (base condition,
chat_prefill, bf16). Also: are the K=16 and K=30 trajectories literally prefixes of the
K=64 one (they should be, deterministic recurrence), and does accuracy keep moving
past loop 30 (early-exit curve).

    .venv/bin/python src/lattrack/ksweep.py
"""

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from commitment import commitment
from commitment_dist import robust_commit
from sets import ROOT, write_json
from stats import bootstrap

RUNS = {16: "huginn_easy_K16", 30: "huginn_easy", 64: "huginn_easy_K64"}
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def load(d):
    rows = [json.loads(l) for l in open(ROOT / "results" / d / "lens_rows.jsonl")]
    return {r["qid"]: r for r in rows if r["condition"] == "base"}


def main():
    R = {K: load(d) for K, d in RUNS.items()}
    qids = sorted(set.intersection(*(set(v) for v in R.values())))
    say(f"# K sweep, Huginn ARC-Easy, {len(qids)} shared questions (ksweep.py)\n")
    # prefix check
    say("## Are shorter runs prefixes of the longer one?\n")
    for a, b in ((16, 30), (30, 64), (16, 64)):
        diff = max(float(np.abs(np.array(R[a][q]["steps"]) - np.array(R[b][q]["steps"])[:a]).max()) for q in qids)
        say(f"- K={a} vs first {a} loops of K={b}: max |logit diff| {diff:.2e}")
    say("\n## Last change: loop count vs fraction\n")
    say("| K | last-change loop mean | median | as fraction of K-1 | last change > loop 16 | final answer differs from K=64's |\n|---|---|---|---|---|---|")
    summary = {}
    for K in RUNS:
        cs = [commitment(["ABCD"[i] for i in np.array(R[K][q]["steps"]).argmax(1)]) for q in qids]
        late = np.mean([c > 16 for c in cs])
        diff64 = np.mean([np.array(R[K][q]["steps"])[-1].argmax() != np.array(R[64][q]["steps"])[-1].argmax() for q in qids])
        b = bootstrap([float(c) for c in cs], np.mean)
        summary[K] = {"last_change_loop": b, "frac": b["point"] / (K - 1), "hist": dict(sorted(collections.Counter(cs).items()))}
        say(f"| {K} | {b['point']:.1f} [{b['lo']:.1f}, {b['hi']:.1f}] | {np.median(cs):.0f} | {b['point']/(K-1):.3f} | {late:.2f} | {diff64:.2f} |")
    say("\nLast-change loop histogram at K=64: " + " ".join(f"{k}:{v}" for k, v in summary[64]["hist"].items()))
    say("\nLast loop at which a DIFFERENT letter than the final one led by more than tol, K=64: " + "; ".join(
        f"tol {t}: {np.mean([robust_commit(np.array(R[64][q]['steps']), t) for q in qids]):.1f}" for t in (0.0, 0.13, 0.3, 0.5)))
    say("\n## Answers still moving late (K=64 trajectory)\n")
    S64 = {q: np.array(R[64][q]["steps"]) for q in qids}
    for m in (8, 16, 24, 32, 48):
        n_diff = sum(S64[q][m].argmax() != S64[q][-1].argmax() for q in qids)
        say(f"- answer at loop {m} differs from the answer at loop 63: {n_diff}/{len(qids)}")
    say("\n## Accuracy by loop (argmax at loop k == correct letter), K=64 trajectory\n")
    say("| loop | accuracy |\n|---|---|")
    acc = {}
    for k in (4, 8, 12, 14, 16, 20, 24, 29, 32, 40, 48, 56, 63):
        b = bootstrap([float(S64[q][k].argmax() == "ABCD".index(R[64][q]["correct_letter"])) for q in qids], np.mean)
        acc[k] = b
        say(f"| {k} | {b['point']:.2f} [{b['lo']:.2f}, {b['hi']:.2f}] |")
    peak = max(range(64), key=lambda k: np.mean([S64[q][k].argmax() == "ABCD".index(R[64][q]["correct_letter"]) for q in qids]))
    say(f"\nBest single loop: {peak} (accuracy {np.mean([S64[q][peak].argmax() == 'ABCD'.index(R[64][q]['correct_letter']) for q in qids]):.2f}).")
    # margin over loops
    marg = np.array([[float(np.sort(S64[q][k])[-1] - np.sort(S64[q][k])[-2]) for k in range(64)] for q in qids])
    say("\nMean top-1 minus top-2 margin by loop: " + ", ".join(f"{k}: {marg[:, k].mean():.2f}" for k in (4, 8, 16, 24, 32, 48, 63)))
    # per-loop change rate in the K=64 trajectory: does it go to zero?
    lead = np.array([S64[q].argmax(1) for q in qids])
    chg = (lead[:, 1:] != lead[:, :-1]).mean(0)
    say("\nFraction of questions whose leader changes at transition k->k+1, binned: " + ", ".join(
        f"loops {lo}-{hi}: {chg[lo:hi].mean():.3f}" for lo, hi in ((0, 8), (8, 16), (16, 24), (24, 32), (32, 48), (48, 63))))
    (ROOT / "results" / "ksweep.md").write_text("\n".join(OUT) + "\n")
    write_json(ROOT / "results" / "ksweep.json", {"schema": "ksweep-v1", "n": len(qids), "runs": RUNS,
                                                  "by_K": {str(k): v for k, v in summary.items()}, "accuracy_by_loop_K64": {str(k): v for k, v in acc.items()}})


if __name__ == "__main__":
    main()
