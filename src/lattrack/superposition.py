"""Is the losing candidate HELD alongside the winner, or simply not there?

RESULT 1/2/5 are negative: the decoded winner does not change more often than for
an arbitrary pair. That is an absence. It leaves open the account you raised — the
model may carry both answers at once, so nothing ever "flips" because nothing was
ever a race. Both readouts take an argmax, which cannot see that.

This asks the positive question with the data already saved (no forwards): at each
step, is the decoy ELEVATED relative to nodes that share its situation?

The task gives an exact matched control. In ProsQA the decoy is always UNREACHABLE
from the root, exactly like the other distractor nodes; the only thing that
distinguishes it is that the question names it as a candidate. So:

    z(v, k) = ( s_k(v) - mean of s_k over unreachable non-candidates )
              / sd of s_k over unreachable non-candidates

z(decoy) >> 0 means the model is holding the decoy above its own reference class —
it is a live candidate, not just an unreached node. z(control), for an unreachable
non-candidate drawn per graph, is the null: it must sit near zero by construction,
and it says how much of any elevation is an artifact of the z-score itself.

Reported per position: z of the decoy against z of the control; how often the decoy
is rank 1 or 2 among all nodes in the graph; how often BOTH candidates occupy the
top two ranks at once (the direct "held together" measure); and the target-decoy
margin against the spread of the reference class, which says whether the two sit
near-tied on the scale the model itself uses.

    .venv/bin/python src/lattrack/superposition.py --run-name seed0
    .venv/bin/python src/lattrack/superposition.py --summarize seed0 seed1
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from lens import NULL_SEED_TAG, from_end
from prompts import Prompt, pin_seed
from sets import ROOT, TEST_OFFSET, load_test, results_dir, write_json
from stats import bootstrap


def classify(pr):
    """Node sets for one graph: candidates, and the decoy's reference class."""
    d = pr.depths()
    nodes = pr.nodes()
    unreachable_noncand = sorted(v for v in nodes if v not in d and v not in (pr.target, pr.decoy))
    reachable_noncand = sorted(v for v in nodes if v in d and v not in (pr.target, pr.decoy))
    return d, unreachable_noncand, reachable_noncand


def analyse_graph(row, pr, rng):
    d, unreach, reach = classify(pr)
    if len(unreach) < 3:
        return None
    control = rng.choice(unreach)
    others = [v for v in unreach if v != control]
    out = []
    for i, per in enumerate(row["per_position"]):
        s = np.array(per["node_logits"])
        ref = s[unreach]
        mu, sd = ref.mean(), ref.std() + 1e-6
        # control is z-scored against its own class WITHOUT itself, same as the decoy
        cref = s[others]
        present = sorted(pr.nodes())
        ranks = {v: 1 + int((s[present] > s[v]).sum()) for v in (pr.target, pr.decoy, control)}
        out.append({
            "z_decoy": float((s[pr.decoy] - mu) / sd),
            "z_target": float((s[pr.target] - mu) / sd),
            "z_control": float((s[control] - cref.mean()) / (cref.std() + 1e-6)),
            "rank_decoy": ranks[pr.decoy], "rank_target": ranks[pr.target],
            "rank_control": ranks[control],
            "both_top2": bool(ranks[pr.target] <= 2 and ranks[pr.decoy] <= 2),
            "margin": float(s[pr.target] - s[pr.decoy]),
            "ref_spread": float(sd),
            "margin_in_spreads": float((s[pr.target] - s[pr.decoy]) / sd),
            "n_unreachable": len(unreach), "n_present": len(present),
        })
    return out


def jacobian_scores(run_name, gis, device="cpu"):
    """Per-node causal-Jacobian scores at every trajectory position, recomputed because
    jlens_flips.py stored only the chosen pairs' gaps. One forward per graph."""
    import torch
    from harness import Runner
    from jlens_flips import load_basis
    from measure import capture
    from sets import require_checkpoint
    test = load_test()
    runner = Runner(str(require_checkpoint(run_name)), device=device, seed=0)
    basis = load_basis(run_name).to(runner.device)
    out = {}
    for gi in gis:
        pr = Prompt.from_sample(test[gi], pin_seed(gi + TEST_OFFSET, 0))
        cap = capture(runner, pr.ids(runner.tok))
        out[gi] = [ (basis[k] @ cap[k].float())[:31].tolist() for k in range(pr.K) ]
    return out


def run(args):
    test = load_test()
    rows = [json.loads(l) for l in open(ROOT / "results" / args.run_name / "lens_rows_all.jsonl")]
    base = [r for r in rows if r["variant"] == "base"]
    jac = jacobian_scores(args.run_name, [r["gi"] for r in base]) if args.basis == "jacobian" else None
    if jac is not None:
        for r in base:
            # the Jacobian trajectory covers the K thoughts (root, l0..l_{K-2}); the lens rows
            # add [A], which has no thought, so it is dropped for this basis
            r["per_position"] = [{"node_logits": v} for v in jac[r["gi"]]]
    per_pos = {}
    per_graph = []
    for r in base:
        pr = Prompt.from_sample(test[r["gi"]], pin_seed(r["gi"] + TEST_OFFSET, 0))
        rng = random.Random(pin_seed(r["gi"] + TEST_OFFSET, NULL_SEED_TAG + 1))
        a = analyse_graph(r, pr, rng)
        if a is None:
            continue
        n = len(a)
        for i, rec in enumerate(a):
            per_pos.setdefault(from_end(i, n), []).append(rec)
        per_graph.append({"gi": r["gi"], "K": r["K"],
                          "frac_both_top2": float(np.mean([x["both_top2"] for x in a])),
                          "min_margin_in_spreads": float(min(x["margin_in_spreads"] for x in a)),
                          "z_decoy_mean": float(np.mean([x["z_decoy"] for x in a]))})

    order = sorted(per_pos, key=lambda k: 0 if k == "last" else int(k.split("-")[1]))
    summary = {"run_name": args.run_name, "n_graphs": len(per_graph), "by_position": {}}
    for k in order:
        recs = per_pos[k]
        summary["by_position"][k] = {
            "n": len(recs),
            "z_decoy": bootstrap([x["z_decoy"] for x in recs], np.median),
            "z_target": bootstrap([x["z_target"] for x in recs], np.median),
            "z_control": bootstrap([x["z_control"] for x in recs], np.median),
            "decoy_rank1or2": bootstrap([float(x["rank_decoy"] <= 2) for x in recs], np.mean),
            "control_rank1or2": bootstrap([float(x["rank_control"] <= 2) for x in recs], np.mean),
            "both_candidates_top2": bootstrap([float(x["both_top2"]) for x in recs], np.mean),
            "margin_in_spreads": bootstrap([x["margin_in_spreads"] for x in recs], np.median),
        }
    summary["per_graph"] = {
        "frac_positions_both_top2": bootstrap([g["frac_both_top2"] for g in per_graph], np.mean),
        "graphs_with_both_top2_everywhere": bootstrap([float(g["frac_both_top2"] == 1.0) for g in per_graph], np.mean),
        "graphs_with_both_top2_never": bootstrap([float(g["frac_both_top2"] == 0.0) for g in per_graph], np.mean),
    }
    summary["basis"] = args.basis
    write_json(results_dir(args.run_name) / f"superposition_{args.basis}.json", summary)
    print(table(summary))


def fmt(b):
    return "—" if b is None else f"{b['point']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}]"


def frac(b):
    return "—" if b is None else f"{b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"


def table(s):
    lines = [f"**{s['run_name']} ({s.get('basis','embedding')} basis)**: n={s['n_graphs']} graphs. z = standard deviations above the "
             f"decoy's own reference class (unreachable non-candidates).",
             "", "| position | z decoy | z control | z target | decoy in top 2 | control in top 2 | both candidates top 2 | margin (in spreads) |",
             "|---|---|---|---|---|---|---|---|"]
    for k, v in s["by_position"].items():
        lines.append(f"| {k} | {fmt(v['z_decoy'])} | {fmt(v['z_control'])} | {fmt(v['z_target'])} | "
                     f"{frac(v['decoy_rank1or2'])} | {frac(v['control_rank1or2'])} | "
                     f"{frac(v['both_candidates_top2'])} | {fmt(v['margin_in_spreads'])} |")
    g = s["per_graph"]
    lines.append(f"\nper graph: positions with both candidates in the top two "
                 f"{frac(g['frac_positions_both_top2'])}; graphs where that holds at EVERY position "
                 f"{frac(g['graphs_with_both_top2_everywhere'])}; at NO position "
                 f"{frac(g['graphs_with_both_top2_never'])}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name")
    p.add_argument("--basis", default="embedding", choices=("embedding", "jacobian"))
    p.add_argument("--summarize", nargs="+", metavar="RUN")
    args = p.parse_args()
    if args.summarize:
        for n in args.summarize:
            for b in ("embedding", "jacobian"):
                f = ROOT / "results" / n / f"superposition_{b}.json"
                if f.exists():
                    print(table(json.load(open(f))), "\n")
    elif args.run_name:
        run(args)
    else:
        p.error("give --run-name or --summarize")


if __name__ == "__main__":
    main()
