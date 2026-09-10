"""Induced reversals in Huginn: is the per-loop detector sensitive on THIS model too?

RESULT 8 calibrated the detector on the 2-layer graph model by manufacturing an
answer change and checking that the detector fires. Huginn is a different
architecture (depth-recurrent, not feed-back), a different readout (its own coda),
and a different task, so RESULT 2's negative still rests on an assumption that the
calibration transfers. This removes the assumption.

Construction. Run the recipient question normally to loop j, then blend the DONOR
question's recurrent state into the recipient's, at the last prompt position only,
at strength alpha, norm preserved:

    x[0, -1] <- ||x[0,-1]|| * normalise( (1-a) x[0,-1] + a * donor_state )

Only the answer position is touched, so no prompt-length matching is needed and the
model can still attend to its own unchanged earlier positions. Loops before j keep
the recipient's state, giving a real "before" — the lesson from RESULT 8, where
injecting at every step left nothing to cross from and produced a confidently wrong
negative.

Ground truth is the model's own output: did the answer letter change? Detector: the
same per-loop leader rule used on the natural data, over the pair (donor's answer,
recipient's answer). The contrast that matters is within one intervention strength —
graphs whose answer moved against those whose answer did not — because an injection
perturbs everything and an arbitrary-pair null cannot control for that (RESULT 8).

    .venv/bin/python src/lattrack/huginn_induced.py --n 24 --device mps
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from huginn_lens import LETTERS, MODEL_ID, load_model, load_questions, prompt_ids, render
from lens import flips_of
from sets import ROOT, write_json
from stats import bootstrap

SCHEMA = "huginn-induced-v1"
ALPHAS = (0.0, 0.5, 1.0)
INJECT_FRAC = 0.62      # loop 20 of 32: after the answer settles (RESULT 2: median last change
                        # at loop 9.5-10), so the "before" leader is stable and real


@torch.no_grad()
def trajectory(model, ids, letter_ids, K, init_seed, inject=None, keep_states=False):
    """Per-loop option logits. `inject` = (first_loop, {loop: donor_vector}, alpha): from
    first_loop to the end, the running state at the LAST position is blended toward the
    donor's state at that same loop, norm preserved. Sustained rather than single-shot,
    because one blended loop was fully recovered within the remaining loops (measured
    2026-09-10: 0 of 6 answers moved). Loops before first_loop keep the recipient's own
    state, so there is still a real 'before' to cross from."""
    torch.manual_seed(init_seed)
    emb, block_idx = model.embed_inputs(ids)
    x = model.initialize_state(emb)
    out, states = [], {}
    for k in range(K):
        x, block_idx, _ = model.iterate_one_step(emb, x, block_idx=block_idx, current_step=k)
        if inject is not None and inject[2] > 0 and k >= inject[0] and k in inject[1]:
            cur = x[0, -1]
            mixed = (1 - inject[2]) * cur + inject[2] * inject[1][k].to(cur.dtype)
            x = x.clone()
            x[0, -1] = mixed / mixed.norm().clamp_min(1e-6) * cur.norm()
        if keep_states:
            states[k] = x[0, -1].detach().clone()
        row = model.predict_from_latents(x).logits[0, -1].float()
        out.append(torch.logsumexp(row[letter_ids], 0).tolist())
    return np.array(out), states


def leader_pair(S, a, b):
    g = (S[:, a] - S[:, b]).tolist()
    leaders, flips = flips_of(g)
    return {"gaps": [round(x, 4) for x in g], "n_flips": len(flips),
            "crossed": leaders[0] != leaders[-1] and "=" not in (leaders[0], leaders[-1]),
            "leader_first": leaders[0], "leader_last": leaders[-1]}


def run(args):
    tok, model = load_model(args.device, getattr(torch, args.dtype))
    nospace = [tok.encode(L, add_special_tokens=False) for L in LETTERS]
    space = [tok.encode(" " + L, add_special_tokens=False) for L in LETTERS]
    letter_ids = torch.tensor([[x[0] for x in nospace], [x[0] for x in space]], device=args.device)
    qs = load_questions(args.n * 2, dataset=args.dataset)
    K, j = args.K, int(args.K * INJECT_FRAC)
    out_dir = ROOT / "results" / "huginn_induced"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "rows.jsonl"
    done = {json.loads(l)["qid"] for l in open(path)} if path.exists() and not args.fresh else set()
    if args.fresh and path.exists():
        path.unlink()

    t0, n = time.time(), 0
    with open(path, "a") as f:
        # pair each question with the next one as its donor
        for i in range(0, min(len(qs) - 1, args.n * 2), 2):
            rq, dq = qs[i], qs[i + 1]
            if rq["id"] in done:
                continue
            rid = prompt_ids(tok, render(rq, rq["choices"]["text"]), args.device)
            did = prompt_ids(tok, render(dq, dq["choices"]["text"]), args.device)
            # donor's running state at loop j, and its own answer
            dS, dstates = trajectory(model, did, letter_ids, K, 0, keep_states=True)
            d_ans = int(dS[-1].argmax())
            base, _ = trajectory(model, rid, letter_ids, K, 0)
            inj_states = {k: v for k, v in dstates.items() if k >= j}
            r_ans = int(base[-1].argmax())
            if d_ans == r_ans:
                continue                      # need a donor that answers differently
            row = {"schema": SCHEMA, "qid": rq["id"], "donor_qid": dq["id"], "K": K, "inject_from_loop": j, "inject_mode": "sustained_last_position",
                   "recipient_answer": LETTERS[r_ans], "donor_answer": LETTERS[d_ans],
                   "recipient_correct": LETTERS[r_ans] == rq["answerKey"], "alphas": {}}
            for a in ALPHAS:
                S, _ = trajectory(model, rid, letter_ids, K, 0, inject=(j, inj_states, a))
                ans = int(S[-1].argmax())
                # null: the two options that are neither answer
                others = [c for c in range(4) if c not in (d_ans, r_ans)]
                row["alphas"][str(a)] = {
                    "answer": LETTERS[ans], "answer_changed": ans != r_ans,
                    "answer_is_donor": ans == d_ans,
                    "real": leader_pair(S, d_ans, r_ans),
                    "null": leader_pair(S, others[0], others[1]),
                }
            f.write(json.dumps(row) + "\n")
            f.flush()
            n += 1
            if args.device == "mps":
                torch.mps.empty_cache()
            print(f"  {n} pairs, {(time.time() - t0)/n:.0f} s each "
                  f"(recipient {row['recipient_answer']}, donor {row['donor_answer']}, "
                  f"alpha=1 -> {row['alphas']['1.0']['answer']})", flush=True)
            if n >= args.n:
                break
    summarize(path, out_dir, {"model": MODEL_ID, "K": K, "inject_from_loop": j, "inject_mode": "sustained_last_position", "alphas": list(ALPHAS),
                              "dataset": args.dataset, "device": args.device, "dtype": args.dtype,
                              "seconds": round(time.time() - t0, 1), "date": time.strftime("%Y-%m-%d")})


def summarize(path, out_dir, conditions=None):
    rows = [json.loads(l) for l in open(path)]
    out = {"schema": SCHEMA, "n_pairs": len(rows), "by_alpha": {}}
    for a in ALPHAS:
        k = str(a)
        R = [r["alphas"][k] for r in rows]
        moved = [x for x in R if x["answer_changed"]]
        still = [x for x in R if not x["answer_changed"]]
        out["by_alpha"][k] = {
            "answer_changed": bootstrap([float(x["answer_changed"]) for x in R], np.mean),
            "answer_is_donor": bootstrap([float(x["answer_is_donor"]) for x in R], np.mean),
            "fires_when_answer_moved": bootstrap([float(x["real"]["crossed"]) for x in moved], np.mean) if moved else None,
            "fires_when_answer_did_not": bootstrap([float(x["real"]["crossed"]) for x in still], np.mean) if still else None,
            "null_fires": bootstrap([float(x["null"]["crossed"]) for x in R], np.mean),
            "n_moved": len(moved), "n_still": len(still),
        }
    if conditions:
        out["conditions"] = conditions
    write_json(out_dir / "summary.json", out)
    f = lambda b: "—" if b is None else f"{b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"
    print(f"\n**Huginn induced reversals**: n={out['n_pairs']} question pairs, inject at loop "
          f"{(conditions or {}).get('inject_from_loop','?')} onward, of {(conditions or {}).get('K','?')}")
    print("| alpha | answer changed | = donor's answer | fires when it moved | when it did not | null |")
    print("|---|---|---|---|---|---|")
    for a in ALPHAS:
        c = out["by_alpha"][str(a)]
        print(f"| {a} | {f(c['answer_changed'])} | {f(c['answer_is_donor'])} | "
              f"{f(c['fires_when_answer_moved'])} (n={c['n_moved']}) | "
              f"{f(c['fires_when_answer_did_not'])} (n={c['n_still']}) | {f(c['null_fires'])} |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--dataset", default="arc_easy", choices=("arc_challenge", "arc_easy", "gsm8k"))
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    if args.summarize_only:
        d = ROOT / "results" / "huginn_induced"
        prev = json.load(open(d / "summary.json")).get("conditions") if (d / "summary.json").exists() else None
        summarize(d / "rows.jsonl", d, prev)
    else:
        run(args)


if __name__ == "__main__":
    main()
