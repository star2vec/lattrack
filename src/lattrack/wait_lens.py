"""Leaning across "wait" in a small text reasoning model: the positive control.

DeepSeek-R1-Distill-Qwen-1.5B answers four-option questions with a greedy chain of
thought. At chosen positions t of the trace we read the model's leaning "if it had
to answer now": prompt + trace[:t] + a forcing suffix that closes the think block
and starts the answer ("\\n</think>\\n\\nThe correct answer is"), and the logits of
the four option letters at the next position (logsumexp over the "A" and " A"
spellings). Two alternative suffixes ("So the answer is", "Answer:") give the
decoder-noise floor: how much the leaning depends on the phrasing that elicits it.

Positions read: every 4th token of the trace, plus every token in a window
[w-6, w+12] around each "wait" token w. Controls: for each wait window a matched
window of the same size centred at a seeded random position of the same trace at
least 8 tokens from any wait ("elsewhere"). Nulls as before: the (correct, top
distractor) pair against the three distractor-distractor pairs.

What a positive control must show: at wait windows the real pair changes leader
or crosses more than in matched windows elsewhere, and those changes clear the
decoder noise and appear under every suffix. No thresholds; the gate is a candidate.

Readouts reuse one KV cache per question, cropped to prompt + t tokens; a tripwire
checks two cached readouts against full forwards on the first question.

    .venv/bin/python src/lattrack/wait_lens.py --n 40 --device mps
"""

import argparse
import copy
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from huginn_lens import LETTERS, load_questions, render
from sets import ROOT, write_json
from stats import bootstrap

SCHEMA = "wait-lens-v1"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
QUESTION_SEED = 20260909
SUFFIXES = {"S0": "\n</think>\n\nThe correct answer is",
            "S1": "\n</think>\n\nSo the answer is",
            "S2": "\n</think>\n\nAnswer:"}
BEFORE, AFTER, CLEAR, STRIDE = 6, 12, 8, 4


# --- model ------------------------------------------------------------------------

def load_model(device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device).eval()
    return tok, model


def prompt_ids(tok, text, device):
    msgs = [{"role": "user", "content": text}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return torch.tensor([tok(s, add_special_tokens=False)["input_ids"]], device=device), s


@torch.no_grad()
def generate(model, ids, max_new):
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=model.config.eos_token_id)
    return out[0, ids.shape[1]:]


@torch.no_grad()
def build_cache(model, full_ids):
    from transformers import DynamicCache
    cache = DynamicCache()
    model(input_ids=full_ids, past_key_values=cache, use_cache=True)
    return cache


@torch.no_grad()
def read_cached(model, cache, keep, suffix_ids, letter_ids, letter_set):
    c = copy.deepcopy(cache)
    c.crop(keep)
    out = model(input_ids=suffix_ids, past_key_values=c, use_cache=True)
    row = out.logits[0, -1].float()
    both = row[letter_ids]
    return (torch.logsumexp(both, 0).tolist(), float(torch.softmax(row, -1)[letter_ids].sum()),
            int(row.argmax()) in letter_set)


@torch.no_grad()
def read_full(model, prefix_ids, suffix_ids, letter_ids):
    out = model(input_ids=torch.cat([prefix_ids, suffix_ids], 1))
    row = out.logits[0, -1].float()
    return torch.logsumexp(row[letter_ids], 0).tolist()


# --- trace analysis -----------------------------------------------------------------

def wait_positions(tok, trace_ids):
    return [i for i, t in enumerate(trace_ids.tolist()) if tok.decode([t]).strip().lower().startswith("wait")]


def control_centres(T, waits, rng):
    """One matched centre per wait window, at least CLEAR tokens from any wait."""
    pool = [c for c in range(BEFORE, T - AFTER) if all(abs(c - w) >= CLEAR for w in waits)]
    if not pool:
        return []
    return [rng.choice(pool) for _ in waits]


def read_positions(T, waits, controls):
    pos = set(range(0, T + 1, STRIDE)) | {T}
    for c in list(waits) + list(controls):
        pos |= set(range(max(0, c - BEFORE), min(T, c + AFTER) + 1))
    return sorted(pos)


def parse_answer(tok, trace_ids):
    text = tok.decode(trace_ids, skip_special_tokens=True)
    tail = text.split("</think>")[-1] if "</think>" in text else ""
    m = re.search(r"\b([A-D])\b", tail)
    return (m.group(1) if m else None), text, "</think>" in text


def window_stats(reads, centre, T, correct_idx, key="S0"):
    """Leader at the window start and end, net change and direction, crossings of the real
    pair and of the distractor pairs inside the window (suffix `key`)."""
    lo, hi = max(0, centre - BEFORE), min(T, centre + AFTER)
    ts = [t for t in range(lo, hi + 1) if t in reads]
    if len(ts) < 2:
        return None
    L = np.array([reads[t][key] for t in ts])
    lead = L.argmax(1)
    final_order = np.argsort(-L[-1])
    dis = [int(i) for i in final_order if i != correct_idx]
    a, b = correct_idx, dis[0]
    g = L[:, a] - L[:, b]
    dis_pairs = [(dis[0], dis[1]), (dis[0], dis[2]), (dis[1], dis[2])]
    cross = lambda x, y: int(sum(1 for k in range(len(L) - 1) if (L[k, x] - L[k, y]) * (L[k + 1, x] - L[k + 1, y]) < 0))
    start, end = int(lead[0]), int(lead[-1])
    if start == end:
        direction = "none"
    elif end == correct_idx:
        direction = "toward_correct"
    elif start == correct_idx:
        direction = "away_from_correct"
    else:
        direction = "between_distractors"
    return {"lo": lo, "hi": hi, "n_reads": len(ts), "leader_start": start, "leader_end": end,
            "net_change": start != end, "direction": direction,
            "gap_start": float(g[0]), "gap_end": float(g[-1]),
            "real_crossings": cross(a, b), "distractor_crossings_mean": float(np.mean([cross(x, y) for x, y in dis_pairs])),
            "leader_changes": int(sum(lead[k] != lead[k + 1] for k in range(len(lead) - 1)))}


# --- summary ----------------------------------------------------------------------------

def summarize(rows):
    out = {"schema": SCHEMA, "n_questions": len(rows)}
    with_wait = [r for r in rows if r["waits"]]
    out["n_with_wait"] = len(with_wait)
    out["waits_per_trace"] = bootstrap([float(len(r["waits"])) for r in rows], np.mean)
    out["trace_tokens_mean"] = float(np.mean([r["T"] for r in rows]))
    out["truncated"] = int(sum(not r["closed_think"] for r in rows))
    out["accuracy_parsed"] = bootstrap([float(r["parsed"] == r["correct"]) for r in rows if r["parsed"]], np.mean)
    out["unparsed"] = int(sum(r["parsed"] is None for r in rows))
    # end-of-trace leaning vs the parsed answer
    agree = []
    for r in rows:
        if r["parsed"] is None:
            continue
        L = r["reads"][str(r["T"])]["S0"]
        agree.append(float(LETTERS[int(np.argmax(L))] == r["parsed"]))
    out["end_leaning_matches_parsed_answer"] = bootstrap(agree, np.mean)
    # decoder noise: real-pair gap under S0 vs S1/S2 at every read position
    diffs = []
    for r in rows:
        ci = LETTERS.index(r["correct"])
        for t, rd in r["reads"].items():
            L0 = np.array(rd["S0"])
            order = np.argsort(-L0)
            top_d = int([i for i in order if i != ci][0])
            g0 = L0[ci] - L0[top_d]
            for k in ("S1", "S2"):
                Lk = np.array(rd[k])
                diffs.append(abs(g0 - (Lk[ci] - Lk[top_d])))
    qq = lambda v, ps=(50, 90, 95, 99): {f"q{p}": float(np.percentile(v, p)) for p in ps} if v else None
    noise = qq(diffs)
    out["decoder_noise_abs_gap_change"] = noise
    q95 = noise["q95"] if noise else None
    # wait windows vs control windows
    for kind in ("wait", "control"):
        W = [w for r in rows for w in r[f"{kind}_windows"] if w]
        if not W:
            out[f"{kind}_windows"] = {"n": 0}
            continue
        gated = [float(w["net_change"] and abs(w["gap_start"]) > q95 and abs(w["gap_end"]) > q95) for w in W]
        out[f"{kind}_windows"] = {
            "n": len(W),
            "net_leader_change": bootstrap([float(w["net_change"]) for w in W], np.mean),
            "direction": {d: int(sum(w["direction"] == d for w in W)) for d in ("toward_correct", "away_from_correct", "between_distractors", "none")},
            "real_pair_crossings_mean": bootstrap([float(w["real_crossings"]) for w in W], np.mean),
            "distractor_pairs_crossings_mean": bootstrap([w["distractor_crossings_mean"] for w in W], np.mean),
            "real_pair_any_crossing": bootstrap([float(w["real_crossings"] > 0) for w in W], np.mean),
            "distractor_pairs_any_crossing": bootstrap([float(w["distractor_crossings_mean"] > 0) for w in W], np.mean),
            "leader_changes_mean": bootstrap([float(w["leader_changes"]) for w in W], np.mean),
            "net_change_gated_q95": bootstrap(gated, np.mean),
            "abs_gap_start_q50": float(np.median([abs(w["gap_start"]) for w in W])),
            "abs_gap_end_q50": float(np.median([abs(w["gap_end"]) for w in W])),
        }
    # decoder consistency of wait-window net changes: same start and end leaders under all suffixes
    consistent = []
    for r in rows:
        for w, w1, w2 in zip(r["wait_windows"], r["wait_windows_S1"], r["wait_windows_S2"]):
            if w and w["net_change"]:
                consistent.append(float(all(x and x["leader_start"] == w["leader_start"] and x["leader_end"] == w["leader_end"] for x in (w1, w2))))
    out["wait_net_changes_present_under_all_suffixes"] = bootstrap(consistent, np.mean) if consistent else None
    out["gate_sentence"] = "candidate, not adopted: |gap| at the window start and end both exceed the decoder-noise q95"
    return out


def fmt(b):
    return "—" if b is None else f"{b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}] (n={b['n']})"


def table(s, cond):
    lines = [f"**DeepSeek-R1-Distill-Qwen-1.5B, leaning across 'wait', {s['n_questions']} ARC-Easy questions** "
             f"({cond.get('device')}, {cond.get('dtype')}, {cond.get('seconds_per_question', float('nan')):.0f} s per question)",
             f"traces: mean {s['trace_tokens_mean']:.0f} tokens, {s['truncated']} truncated before </think>; questions with a wait "
             f"{s['n_with_wait']}/{s['n_questions']}; waits per trace {fmt(s['waits_per_trace'])}",
             f"accuracy (parsed answer) {fmt(s['accuracy_parsed'])}, unparsed {s['unparsed']}; end-of-trace leaning matches the "
             f"parsed answer {fmt(s['end_leaning_matches_parsed_answer'])}",
             f"decoder noise (|Δgap| across forcing suffixes): q50 {s['decoder_noise_abs_gap_change']['q50']:.2f} "
             f"q90 {s['decoder_noise_abs_gap_change']['q90']:.2f} q95 {s['decoder_noise_abs_gap_change']['q95']:.2f}",
             "", "| windows | n | net leader change | gated (both |gap| > q95) | real pair any crossing | distractor pairs any crossing | direction toward / away / between / none |",
             "|---|---|---|---|---|---|---|"]
    for kind in ("wait", "control"):
        w = s[f"{kind}_windows"]
        if w.get("n", 0) == 0:
            lines.append(f"| {kind} | 0 | | | | | |")
            continue
        d = w["direction"]
        lines.append(f"| {kind} | {w['n']} | {fmt(w['net_leader_change'])} | {fmt(w['net_change_gated_q95'])} | "
                     f"{fmt(w['real_pair_any_crossing'])} | {fmt(w['distractor_pairs_any_crossing'])} | "
                     f"{d['toward_correct']} / {d['away_from_correct']} / {d['between_distractors']} / {d['none']} |")
    lines.append(f"wait-window net changes present under all three suffixes: {fmt(s['wait_net_changes_present_under_all_suffixes'])}")
    lines.append(f"gate: {s['gate_sentence']}")
    return "\n".join(lines)


# --- driver ------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--max-new", type=int, default=600)
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--dataset", default="arc_easy")
    p.add_argument("--out", default=str(ROOT / "results" / "wait"))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "wait_rows.jsonl"

    if args.summarize_only:
        rows = [json.loads(l) for l in open(rows_path) if l.strip()]
        s = summarize(rows)
        prev = out_dir / "wait_summary.json"
        s["conditions"] = json.load(open(prev)).get("conditions") if prev.exists() else {}
        write_json(prev, s)
        print(table(s, s["conditions"]))
        return

    if args.fresh and rows_path.exists():
        rows_path.unlink()
    existing = {}
    if rows_path.exists():
        for line in open(rows_path):
            r = json.loads(line)
            existing[r["qid"]] = r

    dtype = getattr(torch, args.dtype)
    t0 = time.time()
    tok, model = load_model(args.device, dtype)
    load_s = time.time() - t0
    nospace = [tok.encode(L, add_special_tokens=False) for L in LETTERS]
    space = [tok.encode(" " + L, add_special_tokens=False) for L in LETTERS]
    assert all(len(x) == 1 for x in nospace + space), (nospace, space)
    letter_ids = torch.tensor([[x[0] for x in nospace], [x[0] for x in space]], device=args.device)
    letter_set = set(letter_ids.flatten().tolist())
    suffix_ids = {k: torch.tensor([tok.encode(v, add_special_tokens=False)], device=args.device) for k, v in SUFFIXES.items()}
    qs = load_questions(args.n, seed=QUESTION_SEED, dataset=args.dataset)

    trip = None
    t1 = time.time()
    n_new = 0
    with open(rows_path, "a") as f:
        for qi, q in enumerate(qs):
            if q["id"] in existing:
                continue
            ids, prompt_text = prompt_ids(tok, render(q, q["choices"]["text"]).replace("Answer with the letter only.", "Answer with the letter."), args.device)
            trace = generate(model, ids, args.max_new)
            T = int(trace.shape[0])
            parsed, text, closed = parse_answer(tok, trace)
            waits = wait_positions(tok, trace)
            rng = random.Random(QUESTION_SEED + qi)
            controls = control_centres(T, waits, rng)
            positions = read_positions(T, waits, controls)
            full = torch.cat([ids, trace.unsqueeze(0)], 1)
            cache = build_cache(model, full)
            P = ids.shape[1]
            reads = {}
            for t in positions:
                rd = {}
                for k, sid in suffix_ids.items():
                    logits, mass, is_letter = read_cached(model, cache, P + t, sid, letter_ids, letter_set)
                    rd[k] = logits
                    if k == "S0":
                        rd["mass0"], rd["argmax_letter0"] = mass, is_letter
                reads[t] = rd
            if trip is None:
                # tripwire: cached readout equals a full forward at two positions (bf16 rounding allowed)
                diffs = []
                for t in (positions[0], positions[len(positions) // 2]):
                    a = reads[t]["S0"]
                    b = read_full(model, full[:, :P + t], suffix_ids["S0"], letter_ids)
                    diffs.append(max(abs(x - y) for x, y in zip(a, b)))
                trip = {"cached_vs_full_max_abs_diff": max(diffs), "positions": [positions[0], positions[len(positions) // 2]]}
                print("tripwire:", trip, flush=True)
                if trip["cached_vs_full_max_abs_diff"] > 0.1:
                    sys.exit("[stop] cached readout disagrees with the full forward")
            ci = LETTERS.index(q["answerKey"])
            row = {"schema": SCHEMA, "qid": q["id"], "qi": qi, "correct": q["answerKey"], "parsed": parsed,
                   "closed_think": closed, "T": T, "n_prompt_tokens": P, "waits": waits, "controls": controls,
                   "trace": text, "reads": {str(t): v for t, v in reads.items()},
                   "wait_windows": [window_stats(reads, w, T, ci) for w in waits],
                   "wait_windows_S1": [window_stats(reads, w, T, ci, "S1") for w in waits],
                   "wait_windows_S2": [window_stats(reads, w, T, ci, "S2") for w in waits],
                   "control_windows": [window_stats(reads, c, T, ci) for c in controls]}
            f.write(json.dumps(row) + "\n")
            f.flush()
            existing[q["id"]] = row
            n_new += 1
            print(f"  q{qi}: T={T} waits={len(waits)} parsed={parsed} correct={q['answerKey']} "
                  f"reads={len(positions)} ({(time.time() - t1) / n_new:.0f} s/question)", flush=True)
    per_q = (time.time() - t1) / max(n_new, 1)
    rows = list(existing.values())
    s = summarize(rows)
    s["conditions"] = {"model": MODEL_ID, "device": args.device, "dtype": args.dtype, "dataset": args.dataset,
                       "question_seed": QUESTION_SEED, "max_new_tokens": args.max_new, "greedy": True,
                       "suffixes": SUFFIXES, "window": {"before": BEFORE, "after": AFTER, "clear": CLEAR, "stride": STRIDE},
                       "tripwire": trip, "model_load_seconds": round(load_s, 1), "seconds_per_question": round(per_q, 1),
                       "torch": torch.__version__, "date": time.strftime("%Y-%m-%d")}
    write_json(out_dir / "wait_summary.json", s)
    print()
    print(table(s, s["conditions"]))


if __name__ == "__main__":
    main()
