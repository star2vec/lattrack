"""Mind-changing graphs without training: a dead-end branch that shares the target's
path prefix and points, in label space, at the decoy.

ProsQA facts measured on the 14,785 training graphs (2026-09-08): every
concept->concept edge goes from a lower to a higher label (a topological numbering,
so a label is a depth cue: label 2 is always at depth 1 and the mean depth rises to
3.0 by label 23); the two name tokens 0 and 1 are the root and the source of the
unreachable component (92% of unreachable nodes descend from the other name); the
decoy is unreachable with 1-6 in-edges from unreachable nodes; root out-degree 1-6.

Construction, on a real test graph under its base serialization: let p be the node
at depth K-2 on a shortest root->target path (lowest label if several). Pick an
unreachable concept node u with label(u) > label(p) and no path from u to the decoy.
Add the single edge (p, u): u becomes reachable at depth K-1 on a branch that shares
the prefix root..p with the target's path and dead-ends short of the decoy. Variants:
  near: u is the eligible node with the largest label BELOW the decoy's label, so
        under the label rule the decoy looks like a child of u;
  far:  u is the eligible node with the smallest label ABOVE the decoy's label, so
        the decoy cannot be u's child under the rule (the matched control: same edit,
        same depth, same slot, label cue points away from the decoy).
Invariants checked on every variant: target depth still K, decoy still unreachable,
u at depth K-1, the new edge respects the label rule. The new edge is inserted at a
per-graph seeded slot of the base edge order (same slot for near and far).

Readout: the lens trajectory of lens.py (root, l0..l_{K-2}, A). Under the label-cue
account the decoy's logit rises at the depth-K readout (position l_{K-2}) in `near`
more than in `far`, and the answer may follow at [A]. No thresholds here.

    .venv/bin/python src/lattrack/mindchange.py --run-name seed0
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from harness import Runner
from lens import flips_of, from_end, position_names, readout, run_ids_all, sha256, transition_names
from prompts import Prompt, pin_seed, reverse_distances, on_path_nodes
from sets import PILOT, TEST_OFFSET, load_test, require_checkpoint, results_dir, write_json
from stats import bootstrap

SCHEMA = "mindchange-v1"
SLOT_SEED_TAG = 4242


def build(pr, gi):
    """(variants, meta). variants: {"near": (Prompt, info), "far": (Prompt, info)} with
    missing keys when no eligible u exists."""
    d = pr.depths()
    K = pr.K
    on = on_path_nodes(pr)
    ps = sorted(v for v, dep in on.items() if dep == K - 2)
    if not ps:
        return {}, {"reason": "no_depth_K-2_path_node"}
    p = ps[0]
    to_decoy = reverse_distances(pr.edges, pr.decoy)  # every node with a path to the decoy
    elig = sorted(v for v in pr.nodes()
                  if v >= 2 and v not in d and v not in to_decoy and v > p)
    near = [v for v in elig if v < pr.decoy]
    far = [v for v in elig if v > pr.decoy]
    rng = random.Random(pin_seed(gi + TEST_OFFSET, SLOT_SEED_TAG))
    slot = rng.randrange(len(pr.edges) + 1)
    variants = {}
    for name, pool, pick in (("near", near, max), ("far", far, min)):
        if not pool:
            continue
        u = pick(pool)
        edges = [list(e) for e in pr.edges]
        edges.insert(slot, [p, u])
        new = pr.with_edges(edges)
        dn = new.depths()
        assert dn.get(pr.target) == K, "target depth changed"
        assert pr.decoy not in dn, "decoy became reachable"
        assert dn.get(u) == K - 1, "u not at depth K-1"
        assert p < u, "label rule broken"
        variants[name] = (new, {"u": u, "slot": slot, "u_minus_decoy": u - pr.decoy,
                                "u_out_degree": sum(1 for a, _ in pr.edges if a == u)})
    return variants, {"p": p, "n_eligible": len(elig), "n_near": len(near), "n_far": len(far),
                      "decoy": pr.decoy, "target": pr.target}


def measure(runner, pr):
    lg = run_ids_all(runner, pr.ids(runner.tok))
    L = pr.layout()
    positions = [L["root"]] + L["latents"][:-1] + [L["a"]]
    per = readout(lg, pr, positions)
    gaps = [r["gap"] for r in per]
    leaders, flips = flips_of(gaps)
    return {"gaps": gaps, "leaders": leaders, "flips": flips,
            "logit_t": [r["logit_t"] for r in per], "logit_d": [r["logit_d"] for r in per],
            "p_t": [r["p_t"] for r in per], "p_d": [r["p_d"] for r in per],
            "rank_d": [r["rank_d"] for r in per],
            "correct": per[-1]["argmax_vocab"] == pr.target, "argmax_A": per[-1]["argmax_vocab"]}


def run(args):
    test = load_test()
    idx = PILOT if args.graphs == "pilot" else list(range(len(test)))
    ckpt = require_checkpoint(args.run_name)
    runner = Runner(str(ckpt), device=args.device, seed=0)
    out_dir = results_dir(args.run_name)
    rows_path = out_dir / f"mindchange_rows_{args.graphs}.jsonl"
    if rows_path.exists():
        rows_path.unlink()
    rows, avail = [], {"near": 0, "far": 0, "both": 0, "none": 0, "no_p": 0}
    t0 = time.time()
    with open(rows_path, "w") as f:
        for gi in idx:
            pr = Prompt.from_sample(test[gi], pin_seed(gi + TEST_OFFSET, 0))
            variants, meta = build(pr, gi)
            if "reason" in meta:
                avail["no_p"] += 1
                continue
            avail["near"] += "near" in variants
            avail["far"] += "far" in variants
            avail["both"] += "near" in variants and "far" in variants
            avail["none"] += not variants
            row = {"schema": SCHEMA, "gi": gi, "K": pr.K, "positions": position_names(pr.K),
                   "transitions": transition_names(pr.K), "meta": meta,
                   "base": measure(runner, pr)}
            for name, (new, info) in variants.items():
                row[name] = {**measure(runner, new), "info": info}
            rows.append(row)
            f.write(json.dumps(row) + "\n")
    elapsed = time.time() - t0

    summary = summarize(rows)
    summary["availability"] = avail
    summary["conditions"] = {"run_name": args.run_name, "graphs": args.graphs, "n_graphs": len(idx),
                             "checkpoint_sha256": sha256(ckpt), "device": args.device,
                             "serialization": "base seed 0; new edge at a per-graph seeded slot",
                             "seconds": round(elapsed, 1), "date": time.strftime("%Y-%m-%d")}
    write_json(out_dir / f"mindchange_{args.graphs}.json", summary)
    print(table(summary, args.run_name))


def summarize(rows):
    out = {"schema": SCHEMA, "n_rows": len(rows)}
    both = [r for r in rows if "near" in r and "far" in r]
    out["n_both"] = len(both)
    # accuracy per condition on the graphs where both variants exist
    for cond in ("base", "near", "far"):
        out[f"accuracy_{cond}"] = bootstrap([float(r[cond]["correct"]) for r in both], np.mean)
    # decoy named at [A]
    for cond in ("near", "far"):
        out[f"answer_is_decoy_{cond}"] = bootstrap(
            [float(r[cond]["argmax_A"] == r["meta"]["decoy"]) for r in both], np.mean)
    # per position (from end): mean gap, mean decoy logit, decoy rank among present nodes
    pos = {}
    for r in both:
        n = len(r["positions"])
        for i in range(n):
            k = from_end(i, n)
            for cond in ("base", "near", "far"):
                pos.setdefault(k, {}).setdefault(cond, {"gap": [], "logit_d": [], "rank_d": [], "t_leads": []})
                pos[k][cond]["gap"].append(r[cond]["gaps"][i])
                pos[k][cond]["logit_d"].append(r[cond]["logit_d"][i])
                pos[k][cond]["rank_d"].append(r[cond]["rank_d"][i])
                pos[k][cond]["t_leads"].append(float(r[cond]["gaps"][i] > 0))
    out["by_position"] = {k: {cond: {m: bootstrap(v, np.mean) for m, v in d.items()} for cond, d in conds.items()}
                          for k, conds in pos.items()}
    # paired differences at the depth-K readout (position l_{K-2} = "last-1") and at [A]
    paired = {}
    for k in ("last-1", "last"):
        i = lambda r: len(r["positions"]) - (2 if k == "last-1" else 1)
        paired[k] = {
            "gap_near_minus_base": bootstrap([r["near"]["gaps"][i(r)] - r["base"]["gaps"][i(r)] for r in both], np.mean),
            "gap_far_minus_base": bootstrap([r["far"]["gaps"][i(r)] - r["base"]["gaps"][i(r)] for r in both], np.mean),
            "gap_near_minus_far": bootstrap([r["near"]["gaps"][i(r)] - r["far"]["gaps"][i(r)] for r in both], np.mean),
            "logit_d_near_minus_far": bootstrap([r["near"]["logit_d"][i(r)] - r["far"]["logit_d"][i(r)] for r in both], np.mean),
        }
    out["paired"] = paired
    # flips: crossing at the last transition and anywhere, per condition
    for cond in ("base", "near", "far"):
        out[f"graphs_with_flip_{cond}"] = bootstrap([float(bool(r[cond]["flips"])) for r in both], np.mean)
        out[f"crossing_last_{cond}"] = bootstrap(
            [float(any(f["t"] == len(r["positions"]) - 2 for f in r[cond]["flips"])) for r in both], np.mean)
    out["u_minus_decoy_near_median"] = float(np.median([r["near"]["info"]["u_minus_decoy"] for r in both]))
    out["u_minus_decoy_far_median"] = float(np.median([r["far"]["info"]["u_minus_decoy"] for r in both]))
    return out


def fmt(b):
    return "—" if b is None else f"{b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"


def table(s, label):
    a = s["availability"]
    lines = [f"**mind-changing graphs, {label}**: availability near {a['near']}, far {a['far']}, both {a['both']}, "
             f"none {a['none']}, no depth-(K-2) path node {a['no_p']} (of {s['n_rows'] + a['no_p']}); "
             f"paired analysis on the {s['n_both']} graphs with both variants",
             f"accuracy at [A]: base {fmt(s['accuracy_base'])}, near {fmt(s['accuracy_near'])}, far {fmt(s['accuracy_far'])}; "
             f"answer is the decoy: near {fmt(s['answer_is_decoy_near'])}, far {fmt(s['answer_is_decoy_far'])}",
             f"label distance u−decoy, median: near {s['u_minus_decoy_near_median']:.0f}, far {s['u_minus_decoy_far_median']:.0f}",
             "", "| position | mean gap base / near / far | target leads base / near / far | decoy logit base / near / far |",
             "|---|---|---|---|"]
    keys = sorted(s["by_position"], key=lambda k: 0 if k == "last" else int(k.split("-")[1]))
    for k in keys:
        c = s["by_position"][k]
        lines.append(f"| {k} | " + " / ".join(f"{c[x]['gap']['point']:.2f}" for x in ("base", "near", "far")) + " | "
                     + " / ".join(f"{c[x]['t_leads']['point']:.2f}" for x in ("base", "near", "far")) + " | "
                     + " / ".join(f"{c[x]['logit_d']['point']:.2f}" for x in ("base", "near", "far")) + " |")
    for k in ("last-1", "last"):
        p = s["paired"][k]
        lines.append(f"  paired at {k}: gap near−base {fmt(p['gap_near_minus_base'])}; far−base {fmt(p['gap_far_minus_base'])}; "
                     f"near−far {fmt(p['gap_near_minus_far'])}; decoy logit near−far {fmt(p['logit_d_near_minus_far'])}")
    lines.append("  graphs with a flip: " + ", ".join(f"{c} {fmt(s[f'graphs_with_flip_{c}'])}" for c in ("base", "near", "far")))
    lines.append("  crossing at the last transition: " + ", ".join(f"{c} {fmt(s[f'crossing_last_{c}'])}" for c in ("base", "near", "far")))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--graphs", default="all", choices=("all", "pilot"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
