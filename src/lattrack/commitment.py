"""When does a latent model stop changing its answer, and does difficulty move it?

The negatives say what does not happen. This asks a positive question of the same
data, with no new runs: for each question, the LAST step after which the decoded
leader never changes again — the commitment point — as a fraction of the trajectory.

Two readings matter.
  Early and difficulty-invariant  => the answer is fixed at a set point in the
      computation regardless of how hard the problem is. Deliberation should scale
      with difficulty; this would not. That is a positive claim about what latent
      steps are doing, and the negatives become its evidence.
  Late, or moving with difficulty => the computation does adapt, and the absence of
      flips needs a different explanation.

Difficulty axes available without new runs: Huginn on ARC-Challenge vs ARC-Easy
(accuracy 0.38 vs 0.50), and the graph model at solution depth K=3 vs K=4. Also
reported: correct vs incorrect questions, since a model that deliberates might be
expected to take longer on the ones it gets wrong.

A caveat kept in view: a trajectory whose leader NEVER changes has a commitment point
of 0 by this definition, which is the honest reading (it was committed from the
start) but means the statistic mixes "decided instantly" with "decided early". Both
the mean and the never-changed fraction are reported.

    .venv/bin/python src/lattrack/commitment.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from sets import ROOT, write_json
from stats import bootstrap


def commitment(leaders):
    """Index of the last change; 0 if the leader never changes. Ties are carried
    forward so a single tied step does not count as two changes."""
    seq = [x for x in leaders]
    last = 0
    for i in range(1, len(seq)):
        a, b = seq[i - 1], seq[i]
        if a != "=" and b != "=" and a != b:
            last = i
    return last


def from_rows(rows, leader_key, n_key, label, extra=None):
    out = []
    for r in rows:
        lead = leader_key(r)
        if lead is None or len(lead) < 2:
            continue
        n = len(lead)
        c = commitment(lead)
        rec = {"frac": c / (n - 1), "step": c, "n_steps": n,
               "never_changed": c == 0, "n_changes": sum(
                   1 for i in range(1, n) if lead[i - 1] != lead[i] and "=" not in (lead[i - 1], lead[i]))}
        if extra:
            rec.update(extra(r))
        out.append(rec)
    return {"label": label, "recs": out}


def block(recs):
    if not recs:
        return {"n": 0}
    return {"n": len(recs),
            "commitment_frac": bootstrap([r["frac"] for r in recs], np.mean),
            "commitment_step": bootstrap([float(r["step"]) for r in recs], np.mean),
            "never_changed": bootstrap([float(r["never_changed"]) for r in recs], np.mean),
            "n_changes": bootstrap([float(r["n_changes"]) for r in recs], np.mean),
            "frac_committed_by_half": bootstrap([float(r["frac"] <= 0.5) for r in recs], np.mean)}


def graph_model(run):
    p = ROOT / "results" / run / "lens_rows_all.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p) if json.loads(l)["variant"] == "base"]
    d = from_rows(rows, lambda r: r["leaders"], None, f"graph {run}",
                  extra=lambda r: {"K": r["K"], "correct": r["correct"]})
    return d


def huginn(dirname, label):
    p = ROOT / "results" / dirname / "lens_rows.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p) if json.loads(l)["condition"] == "base"]

    def lead(r):
        L = np.array(r["steps"])
        return ["ABCD"[i] for i in L.argmax(1)]
    return from_rows(rows, lead, None, label,
                     extra=lambda r: {"correct": r["derived"]["correct"],
                                      "letter_onset": next((i for i, v in enumerate(r.get("argmax_is_letter", []))
                                                            if v), None)})


def main():
    sets = [s for s in (graph_model("seed0"), graph_model("seed1"),
                        huginn("huginn", "Huginn ARC-Challenge (K=32)"),
                        huginn("huginn_easy", "Huginn ARC-Easy (K=30)")) if s]
    out = {"schema": "commitment-v1", "sets": {}}
    print("| model | n | commitment point (fraction of trajectory) | mean step | never changed | committed by halfway |")
    print("|---|---|---|---|---|---|")
    f = lambda b: "—" if b is None else f"{b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"
    for s in sets:
        b = block(s["recs"])
        out["sets"][s["label"]] = {"all": b}
        print(f"| {s['label']} | {b['n']} | {f(b['commitment_frac'])} | {b['commitment_step']['point']:.1f} | "
              f"{f(b['never_changed'])} | {f(b['frac_committed_by_half'])} |")

    print("\n**does difficulty move it?**")
    print("| split | n | commitment point | never changed |")
    print("|---|---|---|---|")
    for s in sets:
        if "graph" in s["label"]:
            for K in (3, 4):
                sub = [r for r in s["recs"] if r.get("K") == K]
                b = block(sub)
                if b["n"]:
                    out["sets"][s["label"]][f"K={K}"] = b
                    print(f"| {s['label']}, depth {K} | {b['n']} | {f(b['commitment_frac'])} | {f(b['never_changed'])} |")
        for name, sub in (("correct", [r for r in s["recs"] if r.get("correct")]),
                          ("incorrect", [r for r in s["recs"] if r.get("correct") is False])):
            b = block(sub)
            if b["n"]:
                out["sets"][s["label"]][name] = b
                print(f"| {s['label']}, {name} | {b['n']} | {f(b['commitment_frac'])} | {f(b['never_changed'])} |")
    write_json(ROOT / "results" / "commitment.json", out)


if __name__ == "__main__":
    main()
