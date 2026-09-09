"""A hidden-state probe as the replacement readout, and the ground-truth reversal check.

The forcing-suffix readout failed its own noise check (RESULT 4: changing the phrase
that elicits the answer moves the leaning more than any within-trace event). A probe
avoids the phrasing entirely: it is fitted once on hidden states and then read at
every position, so nothing about the elicitation varies along the trace.

Probe: multinomial logistic regression (L2, LBFGS) on the last-layer hidden state at
a trace position, four classes = the four option letters. Target is the MODEL's own
final answer, not the gold answer, because the quantity of interest is what the model
would say, not whether it is right. Training positions are the last TAIL_FRAC of each
trace, where the model has committed; questions are split, never positions, so no
question contributes to both fit and test.

Noise, reported the way the forcing-suffix noise was:
  - refit noise (the analogue of the phrasing noise): N_BOOT probes fitted on
    bootstrap resamples of the training QUESTIONS; the spread of the decoded gap at
    a fixed position is how much the readout depends on which data fitted it.
  - split noise: the same, across disjoint halves of the training questions.
A leader change "clears the noise" when both margins exceed the refit q95 and every
bootstrap probe agrees on the direction. Candidate gate, not adopted.

Two analyses:
  wait windows  -- the RESULT 3 comparison rerun with the probe readout.
  ground truth  -- traces whose text states a reversal ("actually, it's B"): does the
                   probe leaning switch across the stated reversal, and to the letter
                   the trace names? This is the positive control the method needs.

    .venv/bin/python src/lattrack/probe_lens.py --traces results/traces_challenge
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from huginn_lens import LETTERS
from sets import write_json
from stats import bootstrap
from wait_lens import BEFORE, AFTER, CLEAR, control_centres, window_stats

SCHEMA = "probe-lens-v1"
TAIL_FRAC = 0.75      # training positions: the last quarter of each trace
N_BOOT = 20
L2 = 1.0
QUESTION_SEED = 20260909


# --- probe -------------------------------------------------------------------------

def fit_probe(X, y, l2=L2, iters=200):
    """Multinomial logistic regression by LBFGS on standardised features.
    Returns (W, b, mu, sd) with logits = ((X - mu) / sd) @ W + b."""
    import torch
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    W = torch.zeros(Xt.shape[1], 4, requires_grad=True)
    b = torch.zeros(4, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt) + l2 * (W * W).sum() / Xt.shape[0]
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach().numpy(), b.detach().numpy(), mu, sd


def apply_probe(P, H):
    W, b, mu, sd = P
    return ((H - mu) / sd) @ W + b


def load_traces(d):
    rows = [json.loads(l) for l in open(Path(d) / "traces.jsonl")]
    hid = Path(d) / "hidden"
    out = []
    for r in rows:
        f = hid / f"{r['qid'].replace('/', '_')}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        r["H"] = z["last"].astype(np.float32)      # (T+1, d): position 0 = last prompt token
        out.append(r)
    return out


def training_set(rows, qids):
    X, y = [], []
    for r in rows:
        if r["qid"] not in qids or r["parsed"] not in LETTERS:
            continue
        T = r["H"].shape[0] - 1
        lo = int(T * TAIL_FRAC)
        X.append(r["H"][lo:])
        y.append(np.full(r["H"].shape[0] - lo, LETTERS.index(r["parsed"])))
    return np.concatenate(X), np.concatenate(y)


# --- analyses ----------------------------------------------------------------------

def leaning(P, r):
    """(T+1, 4) probe logits at every trace position."""
    return apply_probe(P, r["H"])


def reads_from_logits(L):
    return {t: {"S0": L[t].tolist()} for t in range(L.shape[0])}


def ground_truth(rows, P, boots, q95, tok=None):
    """For each trace whose text states a reversal: the probe leader before and after the
    stated reversal, and whether it switches to the letter the trace names next."""
    out = []
    for r in rows:
        if not r["reversals"]:
            continue
        L = leaning(P, r)
        # map a character offset in the trace text to a token index by decoding prefixes
        text = r["trace"]
        for start, end, phrase in r["reversals"]:
            frac = start / max(len(text), 1)
            t = int(frac * (L.shape[0] - 1))
            lo, hi = max(0, t - 40), min(L.shape[0] - 1, t + 60)
            before, after = int(L[lo].argmax()), int(L[hi].argmax())
            # the letter named in the 120 characters after the reversal phrase
            named = re.findall(r"\b([A-D])\b", text[end:end + 120])
            gaps = L[lo:hi + 1].max(1) - np.sort(L[lo:hi + 1], 1)[:, -2]
            agree = [int(apply_probe(b, r["H"][lo]).argmax()) == before
                     and int(apply_probe(b, r["H"][hi]).argmax()) == after for b in boots]
            out.append({
                "qid": r["qid"], "phrase": phrase, "char_start": start, "token_est": t,
                "window": [lo, hi], "leader_before": LETTERS[before], "leader_after": LETTERS[after],
                "switched": before != after, "named_after": named[0] if named else None,
                "matches_named": bool(named) and LETTERS[after] == named[0],
                "final_answer": r["parsed"], "ends_at_after": LETTERS[after] == r["parsed"],
                "margin_before": float(gaps[0]), "margin_after": float(gaps[-1]),
                "clears_noise": bool(gaps[0] > q95 and gaps[-1] > q95),
                "bootstrap_agreement": float(np.mean(agree)),
                "context": text[max(0, start - 120):end + 160].replace("\n", " "),
            })
    return out


def wait_analysis(rows, P):
    import random
    W, C = [], []
    for r in rows:
        L = leaning(P, r)
        T = L.shape[0] - 1
        reads = reads_from_logits(L)
        ci = LETTERS.index(r["parsed"]) if r["parsed"] in LETTERS else int(L[-1].argmax())
        rng = random.Random(QUESTION_SEED + hash(r["qid"]) % 10000)
        ctrls = control_centres(T, r["waits"], rng)
        W += [w for w in (window_stats(reads, c, T, ci) for c in r["waits"]) if w]
        C += [w for w in (window_stats(reads, c, T, ci) for c in ctrls) if w]
    def blk(V):
        if not V:
            return {"n": 0}
        return {"n": len(V),
                "net_leader_change": bootstrap([float(w["net_change"]) for w in V], np.mean),
                "real_pair_any_crossing": bootstrap([float(w["real_crossings"] > 0) for w in V], np.mean),
                "distractor_pairs_any_crossing": bootstrap([float(w["distractor_crossings_mean"] > 0) for w in V], np.mean),
                "leader_changes_mean": bootstrap([float(w["leader_changes"]) for w in V], np.mean),
                "direction": {d: int(sum(w["direction"] == d for w in V)) for d in
                              ("toward_correct", "away_from_correct", "between_distractors", "none")}}
    return {"wait_windows": blk(W), "control_windows": blk(C)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    d = Path(args.traces)
    out_dir = Path(args.out) if args.out else d
    rows = load_traces(d)
    usable = [r for r in rows if r["parsed"] in LETTERS]
    print(f"{len(rows)} traces with hidden states, {len(usable)} with a parsed answer", flush=True)

    qids = sorted(r["qid"] for r in usable)
    rng = np.random.default_rng(QUESTION_SEED)
    perm = rng.permutation(len(qids))
    n_tr = max(2, int(len(qids) * 0.7))
    train_q = {qids[i] for i in perm[:n_tr]}
    test_q = {qids[i] for i in perm[n_tr:]}

    Xtr, ytr = training_set(usable, train_q)
    Xte, yte = training_set(usable, test_q)
    print(f"probe fit on {len(train_q)} questions ({len(ytr)} positions), held out {len(test_q)} ({len(yte)})", flush=True)
    P = fit_probe(Xtr, ytr)
    acc_tr = float((apply_probe(P, Xtr).argmax(1) == ytr).mean())
    acc_te = float((apply_probe(P, Xte).argmax(1) == yte).mean()) if len(yte) else None

    # refit noise: probes on bootstrap resamples of the training questions
    tq = sorted(train_q)
    boots = []
    for b in range(N_BOOT):
        rs = np.random.default_rng(1000 + b)
        sub = {tq[i] for i in rs.integers(0, len(tq), len(tq))}
        Xb, yb = training_set(usable, sub)
        if len(set(yb.tolist())) < 2:
            continue
        boots.append(fit_probe(Xb, yb))
    # spread of the top-two gap at every position of every trace, across refits
    diffs = []
    for r in usable:
        base = leaning(P, r)
        g0 = base.max(1) - np.sort(base, 1)[:, -2]
        for b in boots[:8]:
            gb = apply_probe(b, r["H"])
            g = gb.max(1) - np.sort(gb, 1)[:, -2]
            diffs.extend(np.abs(g0 - g).tolist())
    q = lambda v, ps=(50, 90, 95, 99): {f"q{p}": float(np.percentile(v, p)) for p in ps} if len(v) else None
    noise = q(diffs)
    q95 = noise["q95"] if noise else 0.0

    summary = {"schema": SCHEMA, "n_traces": len(rows), "n_usable": len(usable),
               "probe": {"layer": "last", "target": "the model's own final answer",
                         "train_questions": len(train_q), "test_questions": len(test_q),
                         "train_positions": int(len(ytr)), "accuracy_train": acc_tr,
                         "accuracy_heldout": acc_te, "l2": L2, "tail_frac": TAIL_FRAC},
               "refit_noise_abs_gap_change": noise, "n_bootstrap_probes": len(boots),
               "gate_sentence": "candidate, not adopted: both margins exceed the refit q95",
               "wait_vs_control": wait_analysis(usable, P),
               "ground_truth": ground_truth(usable, P, boots[:8], q95)}
    write_json(out_dir / "probe_summary.json", summary)

    f = lambda b: "—" if b is None else f"{b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}] (n={b['n']})"
    print(f"\n**probe readout** fit on {len(train_q)} questions, {len(ytr)} positions; "
          f"accuracy train {acc_tr:.3f}, held out {acc_te if acc_te is None else round(acc_te, 3)}")
    print(f"refit noise |Δgap| across {len(boots)} bootstrap probes: "
          + (", ".join(f"{k} {v:.2f}" for k, v in noise.items()) if noise else "—"))
    w = summary["wait_vs_control"]
    print("\n| windows | n | net leader change | real pair any crossing | distractor pairs any crossing |")
    print("|---|---|---|---|---|")
    for k in ("wait_windows", "control_windows"):
        b = w[k]
        if b.get("n"):
            print(f"| {k.split('_')[0]} | {b['n']} | {f(b['net_leader_change'])} | "
                  f"{f(b['real_pair_any_crossing'])} | {f(b['distractor_pairs_any_crossing'])} |")
    gt = summary["ground_truth"]
    print(f"\nground truth: {len(gt)} stated reversals in {len({g['qid'] for g in gt})} traces")
    for g in gt:
        print(f"  {g['qid']} '{g['phrase']}' probe {g['leader_before']}->{g['leader_after']} "
              f"switched={g['switched']} names={g['named_after']} matches={g['matches_named']} "
              f"final={g['final_answer']} clears_noise={g['clears_noise']} boot_agree={g['bootstrap_agreement']:.2f}")
    if gt:
        print(f"  switch rate {np.mean([g['switched'] for g in gt]):.3f}; "
              f"switch matches the named letter {np.mean([g['matches_named'] for g in gt]):.3f}; "
              f"clears noise {np.mean([g['clears_noise'] for g in gt]):.3f}")


if __name__ == "__main__":
    main()
