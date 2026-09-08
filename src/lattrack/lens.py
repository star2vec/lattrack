"""Per-step logit lens over the held-out test split: the deciding run of the pilot.

For each graph the prompt is rendered under a pinned serialization and run once; the
logits at every position come out of that one forward. At a latent position the
logits are the tied-embedding lens of the thought recycled from it
(logits[latent_j] == wte @ t_{j+1}, checked on the first graph); at the root they are
the lens of t_0; at [A] they are the answer readout.

Readout positions, in order:  root, l0 .. l{K-1}, A       (K+2 positions)
Transitions are named by their endpoints ("root>l0", "l2>A") and also by their
offset from the end ("last" = l{K-1}>A, "last-1", ...), because the answer arrives
at the last transition whatever K is.

One row per (graph, prompt variant) holds the target-decoy logit gap at every
position, the leader (T = target, D = decoy, = tie), the transitions where the
leader changes ("flips"), the same for two null pairs of non-candidate nodes, and
the full node-token logits so other readouts can be tried without a rerun.

Prompt variants per graph: base (vendor draw under serialization seed 0);
reserial_1..n (edge order redrawn under seed s, candidate order held at base: the
serialization noise floor); cand_swap (base edges, candidates displayed the other
way round: a prompt change, reported separately, never pooled with the noise).

Null pairs (neither candidate, never the name tokens 0 and 1): "any" = two nodes
drawn by a per-graph seeded RNG; "matched" = a reachable depth-K non-target vs an
unreachable non-candidate, the shape of the real pair. The headline of the summary
is the target-decoy crossing rate against these rates, per transition.

No thresholds are applied here. The summary reports counts, rates with bootstrap
intervals, and quantiles; cutoffs come later, after the noise has been seen.

    .venv/bin/python src/lattrack/lens.py --run-name seed1 --graphs pilot   # smoke
    .venv/bin/python src/lattrack/lens.py --run-name seed0                  # all 419
    .venv/bin/python src/lattrack/lens.py --run-name random                 # untrained init
    .venv/bin/python src/lattrack/lens.py --summarize seed0 seed1 random
"""

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from harness import ROOT, VENDOR, Runner
from measure import capture
from prompts import N_NODE_TOKENS, Prompt, covariates, pin_seed, vendor_draws
from sets import PILOT, TEST_OFFSET, load_test, require_checkpoint, results_dir, write_json
from stats import bootstrap

SCHEMA = "lens-v1"
NULL_SEED_TAG = 777        # seed namespace for the null-pair draw, apart from serialization seeds
RANDOM_INIT_SEED = 0       # untrained-init control: vendor set_seed before building the model
ROOT_TOL, LATENT_TOL = 1e-4, 1e-6   # self-consistency tripwire (observed 5.7e-6 and 0.0)
QUANTILES = (50, 90, 95, 99)


# --- forward and readout -----------------------------------------------------

@torch.no_grad()
def run_ids_all(runner, ids):
    """One forward; the full (n, vocab) logits. measure.run_ids keeps only the last row."""
    input_ids = torch.tensor([ids], device=runner.device)
    n = len(ids)
    out = runner.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids.clone(),
        position_ids=torch.arange(n, device=runner.device).reshape(1, -1),
    )
    return out.logits[0].float()


def position_names(K):
    return ["root"] + [f"l{i}" for i in range(K)] + ["A"]


def transition_names(K):
    p = position_names(K)
    return [f"{a}>{b}" for a, b in zip(p[:-1], p[1:])]


def from_end(i, n):
    """Name of transition (or position) i of n counted from the end."""
    k = n - 1 - i
    return "last" if k == 0 else f"last-{k}"


def leader_of(gap):
    return "T" if gap > 0 else ("D" if gap < 0 else "=")


def flips_of(gaps):
    """Transitions where the leader changes, with the gap on each side. A transition
    into or out of a tie is not a flip; ties are counted separately."""
    leaders = [leader_of(g) for g in gaps]
    out = []
    for i in range(len(gaps) - 1):
        a, b = leaders[i], leaders[i + 1]
        if a != "=" and b != "=" and a != b:
            out.append({"t": i, "dir": f"{a}>{b}", "gap_before": gaps[i], "gap_after": gaps[i + 1]})
    return leaders, out


def readout(lg, pr, positions):
    present = sorted(pr.nodes())
    rows = []
    for p in positions:
        row = lg[p]
        node = row[:N_NODE_TOKENS]
        prob = torch.softmax(row, dim=-1)
        pt, pd = float(prob[pr.target]), float(prob[pr.decoy])
        pres = row[present]
        rows.append({
            "logit_t": float(row[pr.target]),
            "logit_d": float(row[pr.decoy]),
            "gap": float(row[pr.target] - row[pr.decoy]),
            "p_t": pt,
            "p_d": pd,
            "T": 100.0 * pt / (pt + pd) if pt + pd > 0 else 50.0,
            "e": 1.0 - (pt + pd),
            "argmax_node": int(node.argmax()),
            "argmax_vocab": int(row.argmax()),
            "rank_t": 1 + int((pres > row[pr.target]).sum()),
            "rank_d": 1 + int((pres > row[pr.decoy]).sum()),
            "node_logits": [round(float(x), 4) for x in node],
        })
    return rows


# --- variants and null pairs -------------------------------------------------

def variants(sample, gi, n_reserial):
    base = Prompt.from_sample(sample, pin_seed(gi + TEST_OFFSET, 0))
    out = [("base", base)]
    for s in range(1, n_reserial + 1):
        edges, _ = vendor_draws(sample, pin_seed(gi + TEST_OFFSET, s))
        out.append((f"reserial_{s}", base.with_edges(edges)))
    out.append(("cand_swap", replace(base, cands=(base.cands[1], base.cands[0]))))
    return out


def null_pairs(pr, gi):
    d = pr.depths()
    eligible = sorted(v for v in pr.nodes() if v not in (pr.target, pr.decoy) and v >= 2)
    rng = random.Random(pin_seed(gi + TEST_OFFSET, NULL_SEED_TAG))
    any_pair = tuple(rng.sample(eligible, 2)) if len(eligible) >= 2 else None
    deep = [v for v in eligible if d.get(v) == pr.K]
    unreach = [v for v in eligible if v not in d]
    matched = (rng.choice(deep), rng.choice(unreach)) if deep and unreach else None
    return any_pair, matched


def pair_block(lg, positions, pr, pair):
    if pair is None:
        return None
    a, b = pair
    d = pr.depths()
    gaps = [float(lg[p, a] - lg[p, b]) for p in positions]
    leaders, flips = flips_of(gaps)
    return {"pair": [a, b], "depth": [d.get(a), d.get(b)], "gaps": gaps,
            "leaders": leaders, "flips": flips}


def make_row(runner, gi, variant, pr):
    lg = run_ids_all(runner, pr.ids(runner.tok))
    L = pr.layout()
    positions = [L["root"]] + L["latents"] + [L["a"]]
    per = readout(lg, pr, positions)
    gaps = [r["gap"] for r in per]
    leaders, flips = flips_of(gaps)
    any_pair, matched = null_pairs(pr, gi)
    return {
        "schema": SCHEMA, "gi": gi, "variant": variant, "K": pr.K,
        "positions": position_names(pr.K), "transitions": transition_names(pr.K),
        "target": pr.target, "decoy": pr.decoy,
        "target_first": pr.cands[0] == pr.target,
        "correct": per[-1]["argmax_vocab"] == pr.target,
        "gaps": gaps, "leaders": leaders, "flips": flips, "n_ties": leaders.count("="),
        "per_position": per,
        "null_any": pair_block(lg, positions, pr, any_pair),
        "null_matched": pair_block(lg, positions, pr, matched),
        "cov": covariates(pr),
    }


# --- tripwires ----------------------------------------------------------------

def tripwires(runner, pr):
    ids = pr.ids(runner.tok)
    a, b = run_ids_all(runner, ids), run_ids_all(runner, ids)
    if not torch.equal(a, b):
        sys.exit("[stop] determinism tripwire: two forwards on the same ids differ")
    cap = capture(runner, ids)
    L = pr.layout()
    wte = runner.wte.detach().float()
    root_diff = float((a[L["root"]] - wte @ cap[0].float()).abs().max())
    lat_diff = max(float((a[L["latents"][j]] - wte @ cap[j + 1].float()).abs().max())
                   for j in range(pr.K - 1))
    if root_diff > ROOT_TOL or lat_diff > LATENT_TOL:
        sys.exit(f"[stop] self-consistency tripwire: root {root_diff:.2e} (limit {ROOT_TOL}), "
                 f"latents {lat_diff:.2e} (limit {LATENT_TOL})")
    return {"root_max_abs_diff": root_diff, "latent_max_abs_diff": lat_diff,
            "limits": {"root": ROOT_TOL, "latents": LATENT_TOL}}


# --- summary ------------------------------------------------------------------

def q(values, quantiles=QUANTILES):
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if len(x) == 0:
        return None
    out = {f"q{p}": float(np.percentile(x, p)) for p in quantiles}
    out["n"] = int(len(x))
    return out


def crossing_rates(base_rows, get):
    """Per transition (forward name and from-end name): mean indicator that the pair
    given by get(row) crosses there, with bootstrap CI. get returns a flips list or None."""
    fwd, end = {}, {}
    for r in base_rows:
        fl = get(r)
        if fl is None:
            continue
        tset = {f["t"] for f in fl}
        n = len(r["transitions"])
        for i, nm in enumerate(r["transitions"]):
            fwd.setdefault(nm, []).append(float(i in tset))
            end.setdefault(from_end(i, n), []).append(float(i in tset))
    return ({k: bootstrap(v, np.mean) for k, v in fwd.items()},
            {k: bootstrap(v, np.mean) for k, v in end.items()})


def recycled_view(base, by, n_reserial):
    """The trajectory without the never-recycled last latent position l_{K-1}: root,
    l0..l_{K-2}, A. l_{K-1} holds a hidden state that is not a thought (nothing reads
    it back; the paper's readout stops at t_{K-1} = position l_{K-2}), and it behaves
    unlike the recycled positions (2026-09-08 run: target leads there in 71-78% of
    graphs against 94-96% at l_{K-2} and at A), producing a zigzag that inflates the
    last two transitions. Reported alongside the full view, not instead of it.

    Also carries a CANDIDATE gate, not adopted: a flip is "above serialization
    noise" when both margins exceed the 95th percentile of |gap change| under
    edge-order redraws at their positions. Reported for the real pair and both nulls
    so the gate can be judged before anyone uses it."""
    def strip(g):
        return g[:-2] + g[-1:]

    # per-position serialization q95 in the full view, keyed by from-end name
    noise = {}
    for r in base:
        n = len(r["positions"])
        for s in range(1, n_reserial + 1):
            x = by.get((r["gi"], f"reserial_{s}"))
            if x is None:
                continue
            for i in range(n):
                noise.setdefault(from_end(i, n), []).append(abs(r["gaps"][i] - x["gaps"][i]))
    q95 = {k: float(np.percentile(v, 95)) for k, v in noise.items()}

    def full_name(i, n_full):
        # stripped index i -> from-end name of the same position in the full view
        j = i if i < n_full - 2 else n_full - 1
        return from_end(j, n_full)

    out = {"positions": "root, l0..l_{K-2}, A", "gate_q95_by_position_full_view": q95,
           "gate_sentence": "candidate, not adopted: both margins of a flip exceed the 95th "
                            "percentile of |gap change| under edge-order redraws at their positions"}
    for name, get in (("target_decoy", lambda r: r["gaps"]),
                      ("null_any", lambda r: r["null_any"]["gaps"] if r["null_any"] else None),
                      ("null_matched", lambda r: r["null_matched"]["gaps"] if r["null_matched"] else None)):
        cross, dirs, two, gate_last, gate_non = {}, {}, [], [], []
        for r in base:
            g = get(r)
            if g is None:
                continue
            n_full = len(r["positions"])
            g = strip(g)
            n = len(g) - 1
            _, fl = flips_of(g)
            ts = {f["t"] for f in fl}
            for i in range(n):
                cross.setdefault(from_end(i, n), []).append(float(i in ts))
            for f in fl:
                dirs.setdefault(from_end(f["t"], n), {"T>D": 0, "D>T": 0})[f["dir"]] += 1
                ok = (abs(f["gap_before"]) > q95[full_name(f["t"], n_full)]
                      and abs(f["gap_after"]) > q95[full_name(f["t"] + 1, n_full)])
                (gate_last if f["t"] == n - 1 else gate_non).append(float(ok))
            two.append(float(len(fl) >= 2))
        out[name] = {
            "crossing_from_end": {k: bootstrap(v, np.mean) for k, v in cross.items()},
            "direction_from_end": dirs,
            "graphs_with_2plus_flips": bootstrap(two, np.mean),
            "gate_survivors_last": bootstrap(gate_last, np.mean),
            "gate_survivors_nonlast": bootstrap(gate_non, np.mean),
        }
    return out


def summarize(rows, n_reserial):
    by = {(r["gi"], r["variant"]): r for r in rows}
    base = sorted((r for r in rows if r["variant"] == "base"), key=lambda r: r["gi"])
    gis = [r["gi"] for r in base]

    real_fwd, real_end = crossing_rates(base, lambda r: r["flips"])
    any_fwd, any_end = crossing_rates(base, lambda r: r["null_any"]["flips"] if r["null_any"] else None)
    mat_fwd, mat_end = crossing_rates(base, lambda r: r["null_matched"]["flips"] if r["null_matched"] else None)

    # direction of the real flips per transition (from end)
    dirs = {}
    for r in base:
        n = len(r["transitions"])
        for f in r["flips"]:
            dirs.setdefault(from_end(f["t"], n), {"T>D": 0, "D>T": 0})[f["dir"]] += 1

    # graphs with flips, by K
    def has_nonlast(r):
        return any(f["t"] < len(r["transitions"]) - 1 for f in r["flips"])
    by_k = {}
    for K in sorted({r["K"] for r in base}):
        rs = [r for r in base if r["K"] == K]
        by_k[K] = {
            "n": len(rs),
            "graphs_with_flip": bootstrap([float(bool(r["flips"])) for r in rs], np.mean),
            "graphs_with_nonlast_flip": bootstrap([float(has_nonlast(r)) for r in rs], np.mean),
            "graphs_with_2plus_flips": bootstrap([float(len(r["flips"]) >= 2) for r in rs], np.mean),
        }

    # margins at flips, last vs non-last
    gap_last, gap_nonlast = {"before": [], "after": []}, {"before": [], "after": []}
    for r in base:
        n = len(r["transitions"])
        for f in r["flips"]:
            dst = gap_last if f["t"] == n - 1 else gap_nonlast
            dst["before"].append(abs(f["gap_before"]))
            dst["after"].append(abs(f["gap_after"]))
    margins = {
        "last": {k: q(v, (10, 50, 90)) for k, v in gap_last.items()},
        "nonlast": {k: q(v, (10, 50, 90)) for k, v in gap_nonlast.items()},
    }

    # serialization noise floor: base vs each edge-only reserial
    noise_end, agree_pos, flipset_same, flip_stable = {}, [], [], []
    for r in base:
        n_pos = len(r["positions"])
        res = [by.get((r["gi"], f"reserial_{s}")) for s in range(1, n_reserial + 1)]
        res = [x for x in res if x is not None]
        if not res:
            continue
        for x in res:
            for i in range(n_pos):
                noise_end.setdefault(from_end(i, n_pos), []).append(abs(r["gaps"][i] - x["gaps"][i]))
        for i in range(n_pos):
            agree_pos.append(float(all(x["leaders"][i] == r["leaders"][i] for x in res)))
        fs = {(f["t"], f["dir"]) for f in r["flips"]}
        flipset_same.append(float(all({(f["t"], f["dir"]) for f in x["flips"]} == fs for x in res)))
        for f in r["flips"]:
            flip_stable.append(float(all((f["t"], f["dir"]) in {(g["t"], g["dir"]) for g in x["flips"]} for x in res)))
    noise = {
        "abs_gap_change_by_position": {k: q(v) for k, v in noise_end.items()},
        "leader_agrees_all_reserials_by_position": bootstrap(agree_pos, np.mean),
        "flip_set_identical_all_reserials": bootstrap(flipset_same, np.mean),
        "base_flip_present_in_all_reserials": bootstrap(flip_stable, np.mean),
        "n_reserial": n_reserial,
    }

    # candidate-order cell (a prompt change, reported apart from the noise)
    cs_end, cs_same = {}, []
    for r in base:
        x = by.get((r["gi"], "cand_swap"))
        if x is None:
            continue
        n_pos = len(r["positions"])
        for i in range(n_pos):
            cs_end.setdefault(from_end(i, n_pos), []).append(abs(r["gaps"][i] - x["gaps"][i]))
        cs_same.append(float({(f["t"], f["dir"]) for f in x["flips"]} == {(f["t"], f["dir"]) for f in r["flips"]}))
    cand_swap = {"abs_gap_change_by_position": {k: q(v) for k, v in cs_end.items()},
                 "flip_set_identical": bootstrap(cs_same, np.mean)}

    return {
        "schema": SCHEMA,
        "n_graphs": len(base),
        "graph_ids": [gis[0], gis[-1]] if gis else [],
        "accuracy_at_A_base": bootstrap([float(r["correct"]) for r in base], np.mean),
        "n_ties_total": int(sum(r["n_ties"] for r in base)),
        "headline_crossing_rate_from_end": {
            "target_decoy": real_end, "null_any": any_end, "null_matched": mat_end,
            "n_null_any": int(sum(r["null_any"] is not None for r in base)),
            "n_null_matched": int(sum(r["null_matched"] is not None for r in base)),
        },
        "crossing_rate_forward_names": {"target_decoy": real_fwd, "null_any": any_fwd, "null_matched": mat_fwd},
        "flip_direction_from_end": dirs,
        "by_K": by_k,
        "margins_at_flips_abs": margins,
        "serialization_noise": noise,
        "cand_swap_cell": cand_swap,
        "recycled_only_view": recycled_view(base, by, n_reserial),
    }


def fmt(b):
    return "—" if b is None else f"{b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}] (n={b['n']})"


def headline_table(summary, label):
    h = summary["headline_crossing_rate_from_end"]
    keys = sorted(set(h["target_decoy"]) | set(h["null_any"]) | set(h["null_matched"]),
                  key=lambda k: 0 if k == "last" else int(k.split("-")[1]))
    lines = [f"**{label}**: n={summary['n_graphs']} graphs, accuracy at [A] {fmt(summary['accuracy_at_A_base'])}, "
             f"ties {summary['n_ties_total']}",
             "", "| transition | target-decoy crossing | null: any pair | null: matched pair | T>D / D>T |",
             "|---|---|---|---|---|"]
    for k in keys:
        d = summary["flip_direction_from_end"].get(k, {"T>D": 0, "D>T": 0})
        lines.append(f"| {k} | {fmt(h['target_decoy'].get(k))} | {fmt(h['null_any'].get(k))} | "
                     f"{fmt(h['null_matched'].get(k))} | {d['T>D']} / {d['D>T']} |")
    n = summary["serialization_noise"]
    lines += ["", f"serialization noise (edge order only, {n['n_reserial']} redraws): "
              f"leader agrees at a position {fmt(n['leader_agrees_all_reserials_by_position'])}; "
              f"flip set identical {fmt(n['flip_set_identical_all_reserials'])}; "
              f"a base flip present in every redraw {fmt(n['base_flip_present_in_all_reserials'])}"]
    for k, v in n["abs_gap_change_by_position"].items():
        lines.append(f"  |Δgap| at {k}: q50 {v['q50']:.2f} q90 {v['q90']:.2f} q95 {v['q95']:.2f} q99 {v['q99']:.2f}")
    m = summary["margins_at_flips_abs"]
    for grp in ("last", "nonlast"):
        b, a = m[grp]["before"], m[grp]["after"]
        if b:
            lines.append(f"  |gap| at {grp} flips: before q10/50/90 {b['q10']:.2f}/{b['q50']:.2f}/{b['q90']:.2f}, "
                         f"after {a['q10']:.2f}/{a['q50']:.2f}/{a['q90']:.2f} (n={b['n']})")
    for K, v in summary["by_K"].items():
        lines.append(f"  K={K}: graphs with a flip {fmt(v['graphs_with_flip'])}; with a non-last flip "
                     f"{fmt(v['graphs_with_nonlast_flip'])}; with 2+ flips {fmt(v['graphs_with_2plus_flips'])}")
    c = summary["cand_swap_cell"]
    lines.append(f"  candidate-order swap (prompt change): flip set identical {fmt(c['flip_set_identical'])}; "
                 + "; ".join(f"|Δgap| {k} q50 {v['q50']:.2f}" for k, v in c["abs_gap_change_by_position"].items()))
    v = summary.get("recycled_only_view")
    if v:
        keys = sorted(v["target_decoy"]["crossing_from_end"],
                      key=lambda k: 0 if k == "last" else int(k.split("-")[1]))
        lines += ["", f"recycled-only view ({v['positions']}); 'last' = l_{{K-2}}>A",
                  "| transition | target-decoy crossing | null: any pair | null: matched pair | T>D / D>T |",
                  "|---|---|---|---|---|"]
        for k in keys:
            d = v["target_decoy"]["direction_from_end"].get(k, {"T>D": 0, "D>T": 0})
            lines.append(f"| {k} | {fmt(v['target_decoy']['crossing_from_end'].get(k))} | "
                         f"{fmt(v['null_any']['crossing_from_end'].get(k))} | "
                         f"{fmt(v['null_matched']['crossing_from_end'].get(k))} | {d['T>D']} / {d['D>T']} |")
        for name in ("target_decoy", "null_any", "null_matched"):
            x = v[name]
            lines.append(f"  [{name}] graphs with 2+ flips (reversal shape) {fmt(x['graphs_with_2plus_flips'])}; "
                         f"candidate gate survivors: last {fmt(x['gate_survivors_last'])}, "
                         f"non-last {fmt(x['gate_survivors_nonlast'])}")
        lines.append(f"  gate ({v['gate_sentence']}); q95 by position: "
                     + ", ".join(f"{k} {q:.2f}" for k, q in v["gate_q95_by_position_full_view"].items()))
    return "\n".join(lines)


# --- drivers ------------------------------------------------------------------

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vendor_commit():
    try:
        return subprocess.check_output(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def load_rows(path):
    rows = []
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def run(args):
    test = load_test()
    idx = PILOT if args.graphs == "pilot" else list(range(len(test)))
    if args.run_name == "random":
        from utils import set_seed  # vendor (harness put VENDOR on sys.path)
        set_seed(RANDOM_INIT_SEED)
        runner = Runner(None, device=args.device, seed=0)
        ckpt, ckpt_sha = None, None
    else:
        ckpt = require_checkpoint(args.run_name)
        runner = Runner(str(ckpt), device=args.device, seed=0)
        ckpt_sha = sha256(ckpt)

    out_dir = results_dir(args.run_name)
    rows_path = out_dir / f"lens_rows_{args.graphs}.jsonl"
    if args.fresh and rows_path.exists():
        rows_path.unlink()
    existing = {}
    for r in load_rows(rows_path):
        if r.get("schema") != SCHEMA:
            sys.exit(f"[stop] {rows_path} holds schema {r.get('schema')!r}; current is {SCHEMA!r}. Rerun with --fresh.")
        existing[(r["gi"], r["variant"])] = r

    trip = tripwires(runner, Prompt.from_sample(test[idx[0]], pin_seed(idx[0] + TEST_OFFSET, 0)))
    t0, n_new = time.time(), 0
    with open(rows_path, "a") as f:
        for gi in idx:
            for name, pr in variants(test[gi], gi, args.n_reserial):
                if (gi, name) in existing:
                    continue
                r = make_row(runner, gi, name, pr)
                f.write(json.dumps(r) + "\n")
                f.flush()
                existing[(gi, name)] = r
                n_new += 1
    elapsed = time.time() - t0

    summary = summarize(list(existing.values()), args.n_reserial)
    summary["conditions"] = {
        "run_name": args.run_name, "graphs": args.graphs, "n_graphs": len(idx),
        "checkpoint": None if ckpt is None else str(ckpt), "checkpoint_sha256": ckpt_sha,
        "random_init_seed": RANDOM_INIT_SEED if args.run_name == "random" else None,
        "test_split": "vendor prosqa_test_graph_4_coconut.json", "vendor_commit": vendor_commit(),
        "serialization": {"base_seed": 0, "reserial_seeds": list(range(1, args.n_reserial + 1)),
                          "namespace": "pin_seed(gi + TEST_OFFSET, seed)"},
        "device": args.device, "torch": torch.__version__,
        "tripwires": trip, "rows_new": n_new, "rows_total": len(existing),
        "seconds": round(elapsed, 1), "date": time.strftime("%Y-%m-%d"),
    }
    write_json(out_dir / f"lens_summary_{args.graphs}.json", summary)
    print()
    print(headline_table(summary, f"{args.run_name} ({args.graphs})"))
    print(f"\nrows: {rows_path} ({n_new} new, {len(existing)} total, {elapsed:.0f}s)")


def compare(args):
    runs = {}
    for name in args.summarize:
        rows = load_rows(ROOT / "results" / name / f"lens_rows_{args.graphs}.jsonl")
        if not rows:
            sys.exit(f"[stop] no rows for {name} ({args.graphs}); run it first")
        runs[name] = {(r["gi"], r["variant"]): r for r in rows}
    out = {"schema": SCHEMA, "graphs": args.graphs, "runs": {}}
    tables = []
    for name, by in runs.items():
        s = summarize(list(by.values()), args.n_reserial)
        # refresh the per-run summary from the rows, keeping its run conditions
        path = ROOT / "results" / name / f"lens_summary_{args.graphs}.json"
        if path.exists():
            s["conditions"] = json.load(open(path)).get("conditions")
            write_json(path, s)
        out["runs"][name] = {"full_view": s["headline_crossing_rate_from_end"],
                             "recycled_only_view": s["recycled_only_view"]}
        tables.append(headline_table(s, name))
    # cross-seed agreement between the first two trained runs given
    trained = [n for n in args.summarize if n != "random"]
    if len(trained) >= 2:
        a, b = runs[trained[0]], runs[trained[1]]
        diff, same = {}, []
        for key, ra in a.items():
            if key[1] != "base" or key not in b:
                continue
            rb = b[key]
            n_pos = len(ra["positions"])
            for i in range(n_pos):
                diff.setdefault(from_end(i, n_pos), []).append(abs(ra["gaps"][i] - rb["gaps"][i]))
            same.append(float({(f["t"], f["dir"]) for f in ra["flips"]} == {(f["t"], f["dir"]) for f in rb["flips"]}))
        out["cross_seed_agreement"] = {
            "runs": trained[:2],
            "abs_gap_difference_by_position": {k: q(v) for k, v in diff.items()},
            "flip_set_identical": bootstrap(same, np.mean),
        }
        tables.append(f"**cross-seed agreement {trained[0]} vs {trained[1]}** (two trained models, a replication, "
                      f"not readout noise): flip set identical {fmt(out['cross_seed_agreement']['flip_set_identical'])}; "
                      + "; ".join(f"|Δgap| {k} q50 {v['q50']:.2f} q90 {v['q90']:.2f}"
                                  for k, v in out["cross_seed_agreement"]["abs_gap_difference_by_position"].items()))
    write_json(ROOT / "results" / f"lens_compare_{args.graphs}.json", out)
    print("\n\n".join(tables))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", help="seed0 | seed1 | random (untrained init)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--graphs", default="all", choices=("all", "pilot"))
    p.add_argument("--n-reserial", type=int, default=3, help="edge-only serialization redraws per graph")
    p.add_argument("--fresh", action="store_true", help="discard existing rows for this run")
    p.add_argument("--summarize", nargs="+", metavar="RUN", help="compare finished runs; no forwards")
    args = p.parse_args()
    if args.summarize:
        compare(args)
    elif args.run_name:
        run(args)
    else:
        p.error("give --run-name or --summarize")


if __name__ == "__main__":
    main()
