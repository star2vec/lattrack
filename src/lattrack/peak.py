"""Does the mid-trajectory accuracy peak survive n=200?

RESULTS 11/12 (n=50, ARC-Easy): accuracy of the loop-k leader was 0.66 at loops 13-14 and
0.52-0.54 at convergence. Pre-registered reading (LOG, 2026-09-11): peak at loops 13-14 vs
the converged value at the last loop, bootstrap intervals, bar = intervals apart or not.
The paired statistic is the one that matters: per question, right at loop k minus right
at the last loop, bootstrapped over questions. Also split first 50 (the n=50 set) vs the
150 new questions, so the n=50 shape can be seen as prior or as noise, and the RESULT 2
quantities at n=200.

    .venv/bin/python src/lattrack/peak.py [--dir huginn_easy_fp32_n200]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from commitment import commitment
from huginn_lens import derive
from sets import ROOT, write_json
from stats import bootstrap

OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def fmt(b):
    return f"{b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"


def curve(rows, label):
    S = [np.array(r["steps"]) for r in rows]
    ci = [("ABCD".index(r["correct_letter"])) for r in rows]
    K = len(S[0])
    right = np.array([[int(s[k].argmax() == c) for k in range(K)] for s, c in zip(S, ci)])   # (n, K)
    acc = right.mean(0)
    say(f"\n## {label}: n={len(rows)}, K={K}\n")
    say("| loop | accuracy |\n|---|---|")
    for k in (4, 8, 10, 12, 13, 14, 15, 16, 18, 20, 24, K - 1):
        say(f"| {k} | {fmt(bootstrap(right[:, k].astype(float), np.mean))} |")
    pk = int(acc.argmax())
    say(f"\nBest loop {pk} ({acc[pk]:.3f}); loops 13-14 mean {acc[13:15].mean():.3f}; last loop {acc[-1]:.3f}.")
    out = {"n": len(rows), "K": K, "accuracy_by_loop": [float(a) for a in acc], "best_loop": pk}
    for k in (13, 14, pk):
        d = bootstrap((right[:, k] - right[:, -1]).astype(float), np.mean)
        n_gain = int(((right[:, k] == 1) & (right[:, -1] == 0)).sum()); n_loss = int(((right[:, k] == 0) & (right[:, -1] == 1)).sum())
        say(f"Paired: right at loop {k} minus right at last loop = {fmt(d)}  (gained {n_gain}, lost {n_loss} questions)")
        out[f"paired_{k}_minus_last"] = d | {"n_gain": n_gain, "n_loss": n_loss}
    return out, right


def result2(rows, label):
    D = [derive(r["steps"], "ABCD".index(r["correct_letter"])) for r in rows]
    S = [np.array(r["steps"]) for r in rows]
    cs = [commitment(["ABCD"[i] for i in s.argmax(1)]) for s in S]
    say(f"\n## RESULT 2 / 10 / 11 quantities at n={len(rows)} ({label})\n")
    b = lambda v: fmt(bootstrap([float(x) for x in v], np.mean))
    say(f"- Cui-Ye event rate {b([d['cui_ye_event'] for d in D])}; (correct, top distractor) ever crosses "
        f"{b([any(d['real_cross']) for d in D])} vs distractor pairs {b([np.mean([any(c) for c in d['distractor_cross']]) for d in D])}")
    say(f"- accuracy on questions with an event {b([d['correct'] for d in D if d['cui_ye_event']])} vs without "
        f"{b([d['correct'] for d in D if not d['cui_ye_event']])}")
    say(f"- last-change loop mean {b(cs)}, median {np.median(cs):.0f}; answer at loop 16 differs from final "
        f"{sum(int(s[16].argmax() != s[-1].argmax()) for s in S)}/{len(S)}")
    say(f"- final accuracy {b([d['correct'] for d in D])}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="huginn_easy_fp32_n200")
    args = p.parse_args()
    rows = [json.loads(l) for l in open(ROOT / "results" / args.dir / "lens_rows.jsonl")]
    rows = sorted((r for r in rows if r["condition"] == "base"), key=lambda r: r["qi"])
    say(f"# Accuracy by loop at n={len(rows)} (peak.py, {args.dir})")
    all_out, _ = curve(rows, "all questions")
    first_out, _ = curve(rows[:50], "first 50 (the n=50 set)")
    rest_out, _ = curve(rows[50:], "questions 51-200 (new)")
    result2(rows, "all")
    write_json(ROOT / "results" / "peak.json", {"schema": "peak-v1", "dir": args.dir, "all": all_out, "first50": first_out, "rest": rest_out})
    (ROOT / "results" / "peak.md").write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
