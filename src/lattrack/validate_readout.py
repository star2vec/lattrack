"""Does the readout follow a documented change of answer? The positive control.

RESULT 4: the method was never shown to DETECT a real answer reversal, so "no flips
above noise" (RESULT 1, 2) could not be separated from "the checks are blind". These
198 traces (results/reversals_full/validation_set.jsonl) each commit to one numeric
answer and finish on a different one, with the switch point known from the text.

The traces are maths, not four-option questions, so the letter readout does not apply.
Leaning here is the model's preference between the two committed answers:

    lean(t) = log P(to | prompt, trace[:t], suffix) - log P(from | ...)

scored as full token sequences after a forcing suffix that opens the answer
("\\n</think>\\n\\nThe final answer is \\boxed{"). Positive means the later answer is
preferred. Everything is teacher-forced over supplied text: forward passes only, no
generation, which is the operation this machine can afford.

Read positions are dense near the end because the median switch sits at 90% of the
trace: every STRIDE tokens over the last WINDOW_FRAC of the trace, plus a few early
baselines. Nulls, as everywhere in this project: two numbers that appear in the trace
but were never committed as answers, read identically. If the real pair moves across
the switch and the null pair does not, the instrument can see a reversal.

Precision note: float16. The earlier bfloat16 problem was a cached-vs-full mismatch of
0.03-0.06 on option logits, the size of the effect then being measured. Here the effect
is a full change of committed answer, and fp16 was measured at 0.008-0.012 on the same
comparison, so it is comfortably below the signal. fp32 would need a 4 GB cache for a
6,000-token trace on a machine that has been in swap all evening.

    .venv/bin/python src/lattrack/validate_readout.py --subset stated
    .venv/bin/python src/lattrack/validate_readout.py --subset all --limit 60
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from sets import ROOT, write_json
from stats import bootstrap
from wait_lens import MODEL_ID

SCHEMA = "validate-readout-v1"
SUFFIX = "\n</think>\n\nThe final answer is \\boxed{"
STRIDE = 96            # tokens between reads inside the dense window
WINDOW_FRAC = 0.35     # dense reads over the last third of the trace
N_EARLY = 4            # early baseline reads, evenly spaced over the rest
MAX_TOKENS = 5000      # traces longer than this are truncated from the LEFT (keeps the switch;
# the median switch is at 90%, so the last 5k tokens contain it and a pre-switch stretch)
EMPTY_EVERY = 4        # release the MPS allocator every N read positions: crop() does not free it,
# and forward+crop cycles grew it until the machine swapped (2026-09-09, twice)


def load_set(subset, limit):
    rows = [json.loads(l) for l in open(ROOT / "results" / "reversals_full" / "validation_set.jsonl")]
    if subset == "stated":
        rows = [r for r in rows if r.get("stated")]
    rows.sort(key=lambda r: r["len_chars"])          # cheapest first
    return rows[:limit] if limit else rows


def candidate_numbers(text, exclude):
    """Two numbers that appear in the trace but were never committed as answers: the
    arbitrary-pair null, in the same spirit as the graph and Huginn runs."""
    seen, out = set(), []
    for m in re.finditer(r"(?<![\w.])\d{1,6}(?![\w.])", text):
        v = m.group(0)
        if v in exclude or v in seen or len(v) < 2:
            continue
        seen.add(v)
        out.append(v)
        if len(out) == 2:
            break
    return out if len(out) == 2 else None


@torch.no_grad()
def score_candidates(model, tok, cache, keep, suffix_ids, cand_ids, device):
    """log P of each candidate's token sequence, continuing from the cropped cache."""
    out = []
    for cid in cand_ids:
        ids = torch.cat([suffix_ids, cid], 1)
        res = model(input_ids=ids, past_key_values=cache, use_cache=True)
        logits = res.logits[0].float()
        # token j of the candidate is predicted by the row before it
        start = suffix_ids.shape[1] - 1
        lp = torch.log_softmax(logits[start:start + cid.shape[1]], -1)
        out.append(float(lp.gather(1, cid[0].unsqueeze(1)).sum()))
        cache.crop(keep)
    return out


def read_positions(T, switch_tok):
    lo = max(0, min(int(T * (1 - WINDOW_FRAC)), switch_tok - 6 * STRIDE))
    dense = list(range(lo, T + 1, STRIDE))
    early = [int(lo * i / (N_EARLY + 1)) for i in range(1, N_EARLY + 1)]
    return sorted(set(early + dense + [T]))


def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(args.device).eval()
    rows = load_set(args.subset, args.limit)
    out_dir = ROOT / "results" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"validation_{args.subset}.jsonl"
    done = {json.loads(l)["idx"] for l in open(path)} if path.exists() and not args.fresh else set()
    if args.fresh and path.exists():
        path.unlink()
    suffix_ids = torch.tensor([tok(SUFFIX, add_special_tokens=False)["input_ids"]], device=args.device)

    t0, n = time.time(), 0
    with open(path, "a") as f:
        for idx, r in enumerate(rows):
            if idx in done:
                continue
            text = r["text"]
            nulls = candidate_numbers(text, {r["from"], r["to"], r["from_norm"], r["to_norm"]})
            if not nulls:
                continue
            ids_all = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids_all) > MAX_TOKENS:
                ids_all = ids_all[-MAX_TOKENS:]
            T = len(ids_all)
            # character offset of the switch -> token index
            switch_tok = len(tok(text[:r["switch_char"]], add_special_tokens=False)["input_ids"])
            switch_tok = max(0, min(T, switch_tok - (len(tok(text, add_special_tokens=False)["input_ids"]) - T)))
            cands = [r["from"], r["to"]] + nulls
            cand_ids = [torch.tensor([tok(c, add_special_tokens=False)["input_ids"]], device=args.device) for c in cands]
            if any(c.shape[1] == 0 for c in cand_ids):
                continue

            ids_t = torch.tensor([ids_all], device=args.device)
            cache = DynamicCache()
            model(input_ids=ids_t[:, :1], past_key_values=cache, use_cache=True)
            pos_list = read_positions(T, switch_tok)
            reads, cur = {}, 1
            for t in pos_list:
                t = max(1, t)
                if t > cur:
                    model(input_ids=ids_t[:, cur:t], past_key_values=cache, use_cache=True)
                    cur = t
                reads[t] = score_candidates(model, tok, cache, cur, suffix_ids, cand_ids, args.device)
                if args.device == "mps" and len(reads) % EMPTY_EVERY == 0:
                    torch.mps.empty_cache()
            row = {"schema": SCHEMA, "idx": idx, "from": r["from"], "to": r["to"],
                   "nulls": nulls, "stated": r.get("stated"), "stated_phrase": r.get("stated_phrase"),
                   "switch_char": r["switch_char"], "switch_frac": r["switch_frac"],
                   "switch_tok": switch_tok, "T": T, "truncated": len(tok(text, add_special_tokens=False)["input_ids"]) > MAX_TOKENS,
                   "positions": sorted(reads), "scores": [reads[t] for t in sorted(reads)],
                   "final_correct": r.get("correct")}
            f.write(json.dumps(row) + "\n")
            f.flush()
            n += 1
            del cache
            if args.device == "mps":
                torch.mps.empty_cache()
            print(f"  {n} traces, {(time.time() - t0) / n:.0f} s each (T={T}, switch at {r['switch_frac']:.0%})", flush=True)
    summarize(path, out_dir, args.subset)


def summarize(path, out_dir, subset):
    rows = [json.loads(l) for l in open(path)]
    if not rows:
        print("no rows")
        return
    real_before, real_after, null_before, null_after, followed, null_followed = [], [], [], [], [], []
    for r in rows:
        S = np.array(r["scores"])                      # (n_pos, 4): from, to, null0, null1
        pos = np.array(r["positions"])
        pre = pos <= r["switch_tok"]
        post = pos > r["switch_tok"]
        if not pre.any() or not post.any():
            continue
        real = S[:, 1] - S[:, 0]                       # log P(to) - log P(from)
        null = S[:, 3] - S[:, 2]
        b, a = float(real[pre].mean()), float(real[post].mean())
        nb, na = float(null[pre].mean()), float(null[post].mean())
        real_before.append(b); real_after.append(a)
        null_before.append(nb); null_after.append(na)
        followed.append(float(a > b))                  # leaning moves toward the later answer
        null_followed.append(float(na > nb))
    out = {"schema": SCHEMA, "subset": subset, "n_traces": len(real_before),
           "real_pair": {"lean_before_switch": bootstrap(real_before, np.mean),
                         "lean_after_switch": bootstrap(real_after, np.mean),
                         "change": bootstrap([a - b for a, b in zip(real_after, real_before)], np.mean),
                         "fraction_moving_toward_the_later_answer": bootstrap(followed, np.mean)},
           "null_pair": {"lean_before_switch": bootstrap(null_before, np.mean),
                         "lean_after_switch": bootstrap(null_after, np.mean),
                         "change": bootstrap([a - b for a, b in zip(null_after, null_before)], np.mean),
                         "fraction_moving_toward_the_later_answer": bootstrap(null_followed, np.mean)},
           "reading": ("lean = log P(later answer) - log P(earlier answer) under the forcing suffix; "
                       "positive means the later answer is preferred")}
    write_json(out_dir / f"validation_{subset}_summary.json", out)
    f = lambda b: "—" if b is None else f"{b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"
    print(f"\n**readout validation, {subset}: n={out['n_traces']} traces with a documented answer change**")
    print("| pair | lean before switch | lean after switch | change | fraction moving to the later answer |")
    print("|---|---|---|---|---|")
    for k, name in (("real_pair", "real (from, to)"), ("null_pair", "null (two uncommitted numbers)")):
        d = out[k]
        print(f"| {name} | {f(d['lean_before_switch'])} | {f(d['lean_after_switch'])} | {f(d['change'])} | "
              f"{f(d['fraction_moving_toward_the_later_answer'])} |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", default="stated", choices=("stated", "all"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="mps")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    if args.summarize_only:
        summarize(ROOT / "results" / "validation" / f"validation_{args.subset}.jsonl",
                  ROOT / "results" / "validation", args.subset)
    else:
        run(args)


if __name__ == "__main__":
    main()
