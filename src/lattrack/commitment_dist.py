"""The distribution behind RESULT 10, not the mean.

commitment.py reported a mean last-change point of ~0.39 of the trajectory in all four
model/task cells. Before reading that as a constant, look at what the mean is a mean of:
where the last change falls (histogram), whether the margin keeps moving after it, how
many of Huginn's changes are bfloat16 ties (the run was bf16; one ulp at |logit|~16 is
0.125), how the point moves if a change must clear a margin tolerance, how many answers
still differ from the final one at loops 8 and 16, and accuracy by loop (early exit).

No new runs. Reads results/{seed0,seed1}/lens_rows_all.jsonl, results/{huginn,huginn_easy}/
lens_rows.jsonl and results/huginn_induced/rows.jsonl; writes results/commitment_dist.md.

    .venv/bin/python src/lattrack/commitment_dist.py
"""

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from commitment import commitment
from sets import ROOT
from stats import bootstrap

ULP = 0.13          # one bfloat16 ulp at |logit| ~ 16, the range Huginn's option logits sit in
TOLS = (0.0, ULP, 0.3, 0.5)
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def graph(run):
    rows = [json.loads(l) for l in open(ROOT / "results" / run / "lens_rows_all.jsonl")]
    rows = [r for r in rows if r["variant"] == "base"]
    say(f"\n## graph {run}: n={len(rows)}")
    for K in (3, 4):
        sub = [r for r in rows if r["K"] == K]
        names = sub[0]["positions"]
        hist = collections.Counter(commitment(r["leaders"]) for r in sub)
        n = len(sub)
        say(f"\nK={K}, positions {names}. Where the LAST leader change falls (0 = never changed):\n")
        say("| " + " | ".join(["never"] + [f"{names[i-1]}->{names[i]}" for i in range(1, len(names))]) + " |")
        say("|" + "---|" * len(names))
        say("| " + " | ".join(f"{hist[i]} ({hist[i]/n:.3f})" for i in range(len(names))) + " |")
        gc, gA, grew = [], [], []
        for r in sub:
            c = commitment(r["leaders"])
            if c < len(r["gaps"]) - 1:
                gc.append(abs(r["gaps"][c])); gA.append(abs(r["gaps"][-1])); grew.append(abs(r["gaps"][-1]) > abs(r["gaps"][c]))
        say(f"\nGraphs committed before A (n={len(gc)}): mean |gap| {np.mean(gc):.2f} at commitment, "
            f"{np.mean(gA):.2f} at A; grew in {np.mean(grew):.2f}.")


def robust_commit(S, tol):
    """Last loop at which a letter OTHER than the final one led by more than tol; 0 if none.
    Unlike commitment(), a change that passes through near-ties still counts, so this
    undercounts nothing real; raising tol asks how confident the earlier leader was."""
    lead = S.argmax(1)
    srt = np.sort(S, 1)
    marg = srt[:, -1] - srt[:, -2]
    idx = [i for i in range(len(lead)) if lead[i] != lead[-1] and marg[i] > tol]
    return (max(idx) + 1) if idx else 0


def huginn(d, label):
    rows = [json.loads(l) for l in open(ROOT / "results" / d / "lens_rows.jsonl") if json.loads(l)["condition"] == "base"]
    n, K = len(rows), len(rows[0]["steps"])
    S = [np.array(r["steps"]) for r in rows]
    say(f"\n## {label}: n={n}, K={K}")
    cs = [commitment(["ABCD"[i] for i in s.argmax(1)]) for s in S]
    say(f"\nLast-change loop: mean {np.mean(cs):.1f} (frac {np.mean(cs)/(K-1):.3f}), median {np.median(cs):.0f}; "
        "histogram " + " ".join(f"{k}:{v}" for k, v in sorted(collections.Counter(cs).items())))
    on = [next(i for i, v in enumerate(r["argmax_is_letter"]) if v) for r in rows]
    say(f"Letter first becomes the argmax at loop {np.mean(on):.1f} mean, {np.median(on):.0f} median.")
    chg = [x for r in rows for x in r["derived"]["changes"]]
    tiny = sum(1 for x in chg if min(x["margin_before"], x["margin_after"]) < ULP)
    say(f"Leader-change transitions {len(chg)}; with one side's margin below one bf16 ulp ({ULP}): {tiny} ({tiny/len(chg):.2f}).")
    say("Last loop at which a DIFFERENT letter led by more than tol: " + "; ".join(
        f"tol {t}: loop {np.mean([robust_commit(s, t) for s in S]):.1f} (frac {np.mean([robust_commit(s, t) for s in S])/(K-1):.3f})" for t in TOLS))
    say("Some other letter ever led by > tol (event presence): " + ", ".join(
        f"tol {t}: {np.mean([robust_commit(s, t) > 0 for s in S]):.2f}" for t in TOLS)
        + f"; Cui-Ye event rate as logged {np.mean([r['derived']['cui_ye_event'] for r in rows]):.2f}.")
    for m in (8, 16):
        if m >= K - 1:
            continue
        late = [s for s in S if s[m].argmax() != s[-1].argmax()]
        margs = sorted(float(np.sort(s[m])[-1] - np.sort(s[m])[-2]) for s in late)
        say(f"Answer at loop {m} differs from the final answer: {len(late)}/{n}; margins at loop {m}: " + ", ".join(f"{x:.2f}" for x in margs))
    mc = [r["derived"]["margin"][c] for r, c in zip(rows, cs)]
    me = [r["derived"]["margin"][-1] for r in rows]
    say(f"Margin at the last change {np.mean(mc):.2f} -> at the last loop {np.mean(me):.2f}; grew in "
        f"{np.mean([b > a for a, b in zip(mc, me)]):.2f}.")
    say("\nAccuracy by loop (argmax at loop k == correct letter), bootstrap CI:\n")
    say("| loop | accuracy |\n|---|---|")
    for k in (0, 4, 8, 12, 14, 16, 20, 24, K - 1):
        b = bootstrap([float(s[k].argmax() == "ABCD".index(r["correct_letter"])) for s, r in zip(S, rows)], np.mean)
        say(f"| {k} | {b['point']:.2f} [{b['lo']:.2f}, {b['hi']:.2f}] |")


def result9_ties():
    rows = [json.loads(l) for l in open(ROOT / "results" / "huginn_induced" / "rows.jsonl")]
    say("\n## RESULT 9, alpha=0: are the crossings bf16 ties?")
    for key in ("null", "real"):
        R = [r["alphas"]["0.0"][key] for r in rows]
        crossed = [x for x in R if x["crossed"]]
        end_tiny = sum(1 for x in crossed if min(abs(x["gaps"][0]), abs(x["gaps"][-1])) < ULP)
        big = sum(1 for x in crossed if any((np.sign(np.array(x["gaps"])) == -np.sign(x["gaps"][-1])) & (np.abs(x["gaps"]) > ULP)))
        say(f"{key} pair: crossed {len(crossed)}/{len(R)}; end gap within one ulp {end_tiny}; opposite leader ahead by > {ULP} at some loop {big}.")


def main():
    say("# Distribution behind RESULT 10 (commitment_dist.py, no new runs)")
    graph("seed0"); graph("seed1")
    huginn("huginn", "Huginn ARC-Challenge"); huginn("huginn_easy", "Huginn ARC-Easy")
    result9_ties()
    (ROOT / "results" / "commitment_dist.md").write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
