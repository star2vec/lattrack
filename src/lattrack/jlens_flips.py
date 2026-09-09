"""The flip analysis in two readouts at once: token embeddings vs causal Jacobian.

RESULT 1 and RESULT 2 say decoded leader changes do not beat an arbitrary-pair null.
Both were read in the TOKEN-EMBEDDING basis (a logit lens). That basis may be the
wrong place to look: information can sit in directions orthogonal to the unembedding
(2604.09885), superposition of candidate answers survives a logit lens mainly in
from-scratch models like this one (2604.06374), and a Jacobian readout is the
standard remedy where the logit lens fails away from the output (2608.25347). RRR's
own result — the frontier is readable at AUC 0.998 and removing it changes nothing —
is the same warning from inside this project.

So: capture each thought ONCE and read it two ways.
  embedding   s_k(v) = <t_k, wte[v]>            (identical to the lens.py readout)
  jacobian    s_k(v) = <t_k, j_k[v]>            j_k[v] = normalise(mean d logit_v(answer) / d t_k)
Everything downstream is identical between the two: leader, flip, margin, the
arbitrary-pair and matched-pair nulls, the serialisation noise floor. Any difference
in the conclusion is then attributable to the basis and nothing else.

Trajectory: the K recycled thoughts, at positions root, l_0 .. l_{K-2} (the same
positions lens.py reads; the never-recycled last latent is excluded, LOG 2026-09-08).
The J-lens has no [A] entry because [A] carries no thought, so the answer readout is
reported separately rather than as a trajectory point.

    .venv/bin/python src/lattrack/jlens_flips.py --run-name seed0
    .venv/bin/python src/lattrack/jlens_flips.py --summarize seed0 seed1
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

from harness import ROOT, Runner
from lens import NULL_SEED_TAG, flips_of, from_end, sha256, variants, vendor_commit
from measure import capture, run_ids
from prompts import N_NODE_TOKENS, Prompt, pin_seed
from sets import PILOT, TEST_OFFSET, load_test, require_checkpoint, results_dir, write_json
from stats import bootstrap

SCHEMA = "jlens-flips-v1"
BASES = ROOT / "bases"


def load_basis(run_name):
    p = BASES / f"{run_name}_jlens_basis.pt"
    if not p.exists():
        sys.exit(f"[stop] no Jacobian basis at {p}; see bases/README.txt")
    return torch.load(p, map_location="cpu", weights_only=True).float()   # (K, vocab, 768)


def scores(thoughts, basis, wte, K):
    """(K, vocab) scores per position for each readout, from the same thoughts.
    Position i (0=root, 1..K-1 = l_0..l_{K-2}) reads thought t_i with j_i."""
    emb, jac = [], []
    for k in range(K):
        t = thoughts[k].float()
        emb.append((wte @ t).tolist())
        jac.append((basis[k] @ t).tolist())
    return np.array(emb), np.array(jac)


def pair_trajectory(S, a, b):
    g = (S[:, a] - S[:, b]).tolist()
    leaders, flips = flips_of(g)
    return {"gaps": g, "leaders": leaders, "flips": flips}


def null_pairs(pr, gi, basis_ok):
    """Same rule as lens.py (neither candidate, never the name tokens 0/1), further
    restricted to nodes the Jacobian basis has a direction for, so both readouts see
    the same pairs."""
    d = pr.depths()
    elig = sorted(v for v in pr.nodes() if v not in (pr.target, pr.decoy) and v >= 2 and basis_ok[v])
    rng = random.Random(pin_seed(gi + TEST_OFFSET, NULL_SEED_TAG))
    any_pair = tuple(rng.sample(elig, 2)) if len(elig) >= 2 else None
    deep = [v for v in elig if d.get(v) == pr.K]
    unreach = [v for v in elig if v not in d]
    matched = (rng.choice(deep), rng.choice(unreach)) if deep and unreach else None
    return any_pair, matched


def make_row(runner, basis, gi, variant, pr, basis_ok):
    ids = pr.ids(runner.tok)
    cap = capture(runner, ids)
    K = pr.K
    emb, jac = scores(cap, basis, runner.wte.detach().float(), K)
    any_pair, matched = null_pairs(pr, gi, basis_ok)
    row = {"schema": SCHEMA, "gi": gi, "variant": variant, "K": K,
           "positions": ["root"] + [f"l{i}" for i in range(K - 1)],
           "target": pr.target, "decoy": pr.decoy,
           "answer_correct": None, "null_any_pair": list(any_pair) if any_pair else None,
           "null_matched_pair": list(matched) if matched else None}
    for name, S in (("emb", emb), ("jac", jac)):
        row[name] = {"real": pair_trajectory(S, pr.target, pr.decoy)}
        row[name]["null_any"] = pair_trajectory(S, *any_pair) if any_pair else None
        row[name]["null_matched"] = pair_trajectory(S, *matched) if matched else None
    # the model's actual answer, for the record (not a trajectory point)
    lg = run_ids(runner, ids)
    row["answer_correct"] = int(lg[:N_NODE_TOKENS].argmax()) == pr.target
    return row


def crossing(rows, readout, key):
    fwd = {}
    for r in rows:
        blk = r[readout][key]
        if blk is None:
            continue
        n = len(r["positions"]) - 1
        ts = {f["t"] for f in blk["flips"]}
        for i in range(n):
            fwd.setdefault(from_end(i, n), []).append(float(i in ts))
    return {k: bootstrap(v, np.mean) for k, v in fwd.items()}


def margins(rows, readout, key):
    out = []
    for r in rows:
        blk = r[readout][key]
        if blk is None:
            continue
        for f in blk["flips"]:
            out += [abs(f["gap_before"]), abs(f["gap_after"])]
    return {f"q{p}": float(np.percentile(out, p)) for p in (10, 50, 90)} if out else None


def summarize(rows, n_reserial):
    by = {(r["gi"], r["variant"]): r for r in rows}
    base = sorted((r for r in rows if r["variant"] == "base"), key=lambda r: r["gi"])
    out = {"schema": SCHEMA, "n_graphs": len(base),
           "accuracy_at_A": bootstrap([float(r["answer_correct"]) for r in base], np.mean)}
    for readout in ("emb", "jac"):
        noise, agree = {}, []
        for r in base:
            n = len(r["positions"])
            for s in range(1, n_reserial + 1):
                x = by.get((r["gi"], f"reserial_{s}"))
                if x is None:
                    continue
                for i in range(n):
                    noise.setdefault(from_end(i, n), []).append(
                        abs(r[readout]["real"]["gaps"][i] - x[readout]["real"]["gaps"][i]))
                agree.append(float(all(a == b for a, b in zip(r[readout]["real"]["leaders"],
                                                              x[readout]["real"]["leaders"]))))
        q95 = {k: float(np.percentile(v, 95)) for k, v in noise.items()}
        gated = []
        for r in base:
            n = len(r["positions"])
            for f in r[readout]["real"]["flips"]:
                gated.append(float(abs(f["gap_before"]) > q95[from_end(f["t"], n)]
                                   and abs(f["gap_after"]) > q95[from_end(f["t"] + 1, n)]))
        out[readout] = {
            "crossing_real": crossing(base, readout, "real"),
            "crossing_null_any": crossing(base, readout, "null_any"),
            "crossing_null_matched": crossing(base, readout, "null_matched"),
            "graphs_with_flip": bootstrap([float(bool(r[readout]["real"]["flips"])) for r in base], np.mean),
            "graphs_with_2plus": bootstrap([float(len(r[readout]["real"]["flips"]) >= 2) for r in base], np.mean),
            "margins_at_flips": margins(base, readout, "real"),
            "serialisation_noise_q95": q95,
            "leader_sequence_identical_across_reserials": bootstrap(agree, np.mean),
            "flips_clearing_noise_q95": bootstrap(gated, np.mean),
            "n_flips": len(gated),
        }
    return out


def fmt(b):
    return "—" if b is None else f"{b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"


def table(s, label):
    keys = sorted(s["emb"]["crossing_real"], key=lambda k: 0 if k == "last" else int(k.split("-")[1]))
    lines = [f"**{label}**: n={s['n_graphs']} graphs, accuracy {fmt(s['accuracy_at_A'])}",
             "", "| transition | embedding: real / any / matched | jacobian: real / any / matched |",
             "|---|---|---|"]
    for k in keys:
        e, j = s["emb"], s["jac"]
        lines.append(f"| {k} | " + " / ".join(f"{e[c].get(k)['point']:.3f}" if e[c].get(k) else "—"
                                              for c in ("crossing_real", "crossing_null_any", "crossing_null_matched"))
                     + " | " + " / ".join(f"{j[c].get(k)['point']:.3f}" if j[c].get(k) else "—"
                                          for c in ("crossing_real", "crossing_null_any", "crossing_null_matched")) + " |")
    for r, name in (("emb", "embedding"), ("jac", "jacobian")):
        b = s[r]
        m = b["margins_at_flips"]
        lines.append(f"  [{name}] graphs with a flip {fmt(b['graphs_with_flip'])}; with 2+ {fmt(b['graphs_with_2plus'])}; "
                     f"flips clearing the serialisation q95 {fmt(b['flips_clearing_noise_q95'])} of {b['n_flips']}; "
                     f"leader sequence identical across redraws {fmt(b['leader_sequence_identical_across_reserials'])}")
        if m:
            lines.append(f"     |gap| at flips q10/50/90 {m['q10']:.3f}/{m['q50']:.3f}/{m['q90']:.3f}; "
                         f"serialisation q95 by position " + ", ".join(f"{k} {v:.3f}" for k, v in b["serialisation_noise_q95"].items()))
    return "\n".join(lines)


def run(args):
    test = load_test()
    idx = PILOT if args.graphs == "pilot" else list(range(len(test)))
    ckpt = require_checkpoint(args.run_name)
    runner = Runner(str(ckpt), device=args.device, seed=0)
    basis = load_basis(args.run_name).to(runner.device)
    basis_ok = (basis.norm(dim=-1) > 0).all(dim=0).tolist()   # a direction at every pass
    out_dir = results_dir(args.run_name)
    rows_path = out_dir / f"jlens_rows_{args.graphs}.jsonl"
    if args.fresh and rows_path.exists():
        rows_path.unlink()
    existing = {}
    if rows_path.exists():
        for line in open(rows_path):
            r = json.loads(line)
            existing[(r["gi"], r["variant"])] = r
    t0, n_new = time.time(), 0
    with open(rows_path, "a") as f:
        for gi in idx:
            for name, pr in variants(test[gi], gi, args.n_reserial):
                if (gi, name) in existing:
                    continue
                r = make_row(runner, basis, gi, name, pr, basis_ok)
                f.write(json.dumps(r) + "\n")
                existing[(gi, name)] = r
                n_new += 1
    elapsed = time.time() - t0
    s = summarize(list(existing.values()), args.n_reserial)
    s["conditions"] = {"run_name": args.run_name, "graphs": args.graphs,
                       "checkpoint_sha256": sha256(ckpt), "basis": f"bases/{args.run_name}_jlens_basis.pt",
                       "basis_sha256": sha256(BASES / f"{args.run_name}_jlens_basis.pt"),
                       "nodes_with_a_jacobian_direction": int(sum(basis_ok)),
                       "vendor_commit": vendor_commit(), "device": args.device,
                       "n_reserial": args.n_reserial, "seconds": round(elapsed, 1),
                       "rows_new": n_new, "date": time.strftime("%Y-%m-%d")}
    write_json(out_dir / f"jlens_summary_{args.graphs}.json", s)
    print()
    print(table(s, f"{args.run_name} ({args.graphs})"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name")
    p.add_argument("--device", default="cpu")
    p.add_argument("--graphs", default="all", choices=("all", "pilot"))
    p.add_argument("--n-reserial", type=int, default=3)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--summarize", nargs="+", metavar="RUN")
    args = p.parse_args()
    if args.summarize:
        for name in args.summarize:
            rows = [json.loads(l) for l in open(ROOT / "results" / name / f"jlens_rows_{args.graphs}.jsonl")]
            print(table(summarize(rows, args.n_reserial), name), "\n")
    elif args.run_name:
        run(args)
    else:
        p.error("give --run-name or --summarize")


if __name__ == "__main__":
    main()
