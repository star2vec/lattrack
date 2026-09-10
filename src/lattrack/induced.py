"""Induced reversals: how large a mid-trajectory change of answer can this detector catch?

RESULT 6 validated the readout on a TEXT model, because no corpus of documented
reversals exists for a latent one. A referee will ask why that transfers. It need
not: we can MANUFACTURE a reversal inside the latent model and check the detector
on the substrate the claims are about.

Construction (RRR's transplant, #010): inject a different-structure donor's thoughts
into a recipient's intermediate passes. RRR measured this at -91.7 on the answer, so
the recipient's output moves to the donor's target. That is a mid-trajectory change of
answer with ground truth by construction: before the injected passes the state is the
recipient's, after it is the donor's.

Graded, not binary. `thoughts.mix_edit` blends live and donor thoughts at strength
alpha with the norm preserved, so we sweep alpha and get a SENSITIVITY CURVE rather
than one yes/no: at each strength, how far did the answer actually move, and did the
detector fire? That answers the question the negative results really need — how large
would a natural flip have had to be for us to have caught it — instead of just
"the detector works".

Everything downstream is the pipeline used on the natural data: the same leader/flip
rule, the same arbitrary-pair null, both readouts. The pair tracked here is
(recipient's target, donor's target), since that is the answer change being induced.

    .venv/bin/python src/lattrack/induced.py --run-name seed0 --n 100
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
from jlens_flips import load_basis
from lens import NULL_SEED_TAG, flips_of, from_end, sha256
from measure import capture, intermediates, run_ids
from prompts import N_NODE_TOKENS, Prompt, pin_seed
from sets import TEST_OFFSET, load_test, load_train, require_checkpoint, results_dir, write_json
from stats import bootstrap
from thoughts import mix_edit

SCHEMA = "induced-v1"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def find_donor(pool, pr, k, used):
    """A different-structure donor of the same depth whose candidates are disjoint from
    the recipient's, so the induced answer is unambiguous (RRR find_random_donor)."""
    for i, d in enumerate(pool):
        if i in used or len(d["steps"]) != k:
            continue
        if d["target"] in (pr.target, pr.decoy) or d["neg_target"] in (pr.target, pr.decoy):
            continue
        used.add(i)
        return d
    return None


def scores_at_positions(runner, basis, ids, K, edit=None):
    """(K, vocab) per-node scores at each trajectory position, in both readouts, from
    one forward with the given thought edit."""
    cap = capture(runner, ids) if edit is None else capture_edited(runner, ids, edit)
    wte = runner.wte.detach().float()
    emb = np.array([(wte @ cap[k].float()).tolist() for k in range(K)])
    jac = np.array([(basis[k] @ cap[k].float()).tolist() for k in range(K)])
    return emb, jac


@torch.no_grad()
def capture_edited(runner, ids, edit):
    """Thoughts actually used, under an edit (the edit fires, we record what went in)."""
    cap = {}

    def grab(k, v):
        v2 = edit(k, v)
        cap[k] = v2.detach().clone()
        return v2
    run_ids(runner, ids, grab)
    return cap


def answer_of(runner, ids, edit=None):
    lg = run_ids(runner, ids, edit)
    return int(lg[:N_NODE_TOKENS].argmax()), lg


def pair_flip(S, a, b):
    g = (S[:, a] - S[:, b]).tolist()
    leaders, flips = flips_of(g)
    return {"gaps": g, "flips": flips, "n_flips": len(flips),
            "leader_first": leaders[0], "leader_last": leaders[-1],
            "crossed": leaders[0] != leaders[-1] and "=" not in (leaders[0], leaders[-1])}


def run(args):
    test, train = load_test(), load_train()
    ckpt = require_checkpoint(args.run_name)
    runner = Runner(str(ckpt), device=args.device, seed=0)
    basis = load_basis(args.run_name).to(runner.device)
    out_dir = results_dir(args.run_name)
    rows, used = [], set()
    t0 = time.time()
    for gi in range(min(args.n, len(test))):
        pr = Prompt.from_sample(test[gi], pin_seed(gi + TEST_OFFSET, 0))
        K = pr.K
        donor = find_donor(train, pr, K, used)
        if donor is None:
            continue
        dpr = Prompt.from_sample(donor, pin_seed(gi + TEST_OFFSET, 12345))
        ids, dids = pr.ids(runner.tok), dpr.ids(runner.tok)
        dcap = capture(runner, dids)
        base_ans, _ = answer_of(runner, ids)
        # null pair: two nodes in the recipient, neither candidate, never the name tokens
        rng = random.Random(pin_seed(gi + TEST_OFFSET, NULL_SEED_TAG + 2))
        elig = sorted(v for v in pr.nodes() if v >= 2 and v not in (pr.target, pr.decoy, dpr.target))
        if len(elig) < 2:
            continue
        null_pair = tuple(rng.sample(elig, 2))
        # Inject at the LAST intermediate pass only. Injecting at ALL of them (RRR's
        # donor_intermediate) replaces the whole trajectory, so the donor's answer leads
        # from position 0 and there is no "before" to cross FROM: the detector then
        # correctly reports no crossing and we would wrongly read that as insensitivity
        # (measured 2026-09-10: at alpha=1 the answer changed on 95% of graphs while the
        # detector fired on 35%, below the 70% null). A mid-trajectory change needs the
        # early positions left on the recipient's own state.
        passes = intermediates(K) if args.inject_at == "all_intermediate" else [max(1, K - 2)]
        row = {"schema": SCHEMA, "gi": gi, "K": K, "target": pr.target, "decoy": pr.decoy,
               "donor_target": dpr.target, "null_pair": list(null_pair),
               "baseline_answer": base_ans, "baseline_correct": base_ans == pr.target,
               "injected_passes": list(passes),
               "alphas": {}}
        for a in ALPHAS:
            edit = None if a == 0.0 else mix_edit(dcap, set(passes), a)
            ans, _ = answer_of(runner, ids, edit)
            emb, jac = scores_at_positions(runner, basis, ids, K, edit)
            row["alphas"][str(a)] = {
                "answer": ans,
                "answer_is_donor_target": ans == dpr.target,
                "answer_changed": ans != base_ans,
                "emb": {"real": pair_flip(emb, dpr.target, pr.target),
                        "null": pair_flip(emb, *null_pair)},
                "jac": {"real": pair_flip(jac, dpr.target, pr.target),
                        "null": pair_flip(jac, *null_pair)},
            }
        rows.append(row)
    elapsed = time.time() - t0
    path = out_dir / "induced_rows.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    s = summarize(rows)
    s["conditions"] = {"run_name": args.run_name, "n_graphs": len(rows),
                       "checkpoint_sha256": sha256(ckpt), "alphas": list(ALPHAS),
                       "injected_passes": args.inject_at,
                       "donor": "different-structure training graph, same K, disjoint candidates",
                       "seconds": round(elapsed, 1), "date": time.strftime("%Y-%m-%d")}
    write_json(out_dir / "induced_summary.json", s)
    print(table(s, args.run_name))


def summarize(rows):
    out = {"schema": SCHEMA, "n_graphs": len(rows), "by_alpha": {}}
    for a in ALPHAS:
        k = str(a)
        R = [r["alphas"][k] for r in rows]
        cell = {"answer_changed": bootstrap([float(x["answer_changed"]) for x in R], np.mean),
                "answer_is_donor_target": bootstrap([float(x["answer_is_donor_target"]) for x in R], np.mean)}
        for basis in ("emb", "jac"):
            moved = [x for x in R if x["answer_changed"]]
            cell[basis] = {
                "detector_fires_real": bootstrap([float(x[basis]["real"]["crossed"]) for x in R], np.mean),
                "detector_fires_null": bootstrap([float(x[basis]["null"]["crossed"]) for x in R], np.mean),
                "detector_fires_real_given_answer_moved":
                    bootstrap([float(x[basis]["real"]["crossed"]) for x in moved], np.mean) if moved else None,
                "n_answer_moved": len(moved),
            }
        out["by_alpha"][k] = cell
    return out


def fmt(b):
    return "—" if b is None else f"{b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"


def table(s, label):
    lines = [f"**induced reversals, {label}**: n={s['n_graphs']} graphs; donor thoughts mixed into "
             f"passes 0..K-2 at strength alpha",
             "", "| alpha | answer changed | answer = donor's target | detector fires (emb) real / null | (jac) real / null | fires given the answer moved (emb) |",
             "|---|---|---|---|---|---|"]
    for a in ALPHAS:
        c = s["by_alpha"][str(a)]
        lines.append(f"| {a} | {fmt(c['answer_changed'])} | {fmt(c['answer_is_donor_target'])} | "
                     f"{fmt(c['emb']['detector_fires_real'])} / {fmt(c['emb']['detector_fires_null'])} | "
                     f"{fmt(c['jac']['detector_fires_real'])} / {fmt(c['jac']['detector_fires_null'])} | "
                     f"{fmt(c['emb']['detector_fires_real_given_answer_moved'])} (n={c['emb']['n_answer_moved']}) |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--device", default="cpu")
    p.add_argument("--inject-at", default="last_intermediate",
                   choices=("last_intermediate", "all_intermediate"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
