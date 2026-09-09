"""Greedy traces and their hidden states: the shared input for the ground-truth
reversal check and the probe readout.

Generation is batched with left padding (measured 2026-09-09: 192 ms per token per
sequence at batch 1, 104 ms at batch 4 on this Mac in float32 on MPS). Hidden states
come from ONE forward per trace with output_hidden_states, which gives every position
at once, so they cost about a second per trace rather than one forward per position.

Two phases, both resumable by question id:
    --phase generate   traces (text, parsed answer, wait positions) -> traces.jsonl
    --phase hidden     last-layer hidden states at every position   -> hidden/<qid>.npy

Device note (phoenix README): on this Mac the small 2-layer model is faster on CPU
(25 ms vs 178 ms per forward) and the MPS allocator grows until a run is killed. For
this 1.5B model MPS is 2x faster than CPU (123 s vs 247 s per question, measured
2026-09-09), so MPS is used, with torch.mps.empty_cache() after every batch.

    .venv/bin/python src/lattrack/traces.py --phase generate --dataset arc_challenge --n 24
    .venv/bin/python src/lattrack/traces.py --phase hidden --out results/traces_challenge
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

from huginn_lens import LETTERS, load_questions, render
from sets import ROOT
from wait_lens import MODEL_ID, QUESTION_SEED, parse_answer, wait_positions

# an explicit written reversal: the trace states that a previous answer was wrong or
# names a new one. Ground truth for "the model changed its mind", read off the text.
REVERSAL = re.compile(
    r"(actually,?\s+(it'?s|the answer is|i think)|"
    r"no,?\s+(wait|it'?s|the answer is)|"
    r"wait,?\s+(no|actually)|"
    r"i was wrong|i made a mistake|"
    r"that'?s (wrong|incorrect|not right)|"
    r"change my answer|scratch that|correction|"
    r"on second thought|hold on,?\s+(no|actually))", re.I)


def user_text(q, options=None):
    return render(q, options or q["choices"]["text"]).replace(
        "Answer with the letter only.", "Answer with the letter.")


def find_reversals(text):
    """[(char_start, char_end, matched phrase)] of explicit reversal statements inside
    the think block (a reversal after </think> is a restatement, not a change of mind)."""
    think = text.split("</think>")[0]
    return [(m.start(), m.end(), m.group(0)) for m in REVERSAL.finditer(think)]


def generate_phase(args, out_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=getattr(torch, args.dtype)).to(args.device).eval()

    path = out_dir / "traces.jsonl"
    done = {json.loads(l)["qid"] for l in open(path)} if path.exists() else set()
    qs = [q for q in load_questions(args.n, seed=QUESTION_SEED, dataset=args.dataset) if q["id"] not in done]
    print(f"{len(done)} traces on file; {len(qs)} to generate", flush=True)

    t0, n = time.time(), 0
    with open(path, "a") as f:
        for i in range(0, len(qs), args.batch):
            chunk = qs[i:i + args.batch]
            texts = [tok.apply_chat_template([{"role": "user", "content": user_text(q)}],
                                             tokenize=False, add_generation_prompt=True) for q in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(args.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                     temperature=None, top_p=None, pad_token_id=tok.eos_token_id)
            for j, q in enumerate(chunk):
                # left padding: the prompt occupies the last enc positions of the input block
                gen = out[j, enc["input_ids"].shape[1]:]
                gen = gen[gen != tok.eos_token_id] if args.strip_eos else gen
                parsed, text, closed = parse_answer(tok, gen)
                revs = find_reversals(text)
                row = {"qid": q["id"], "dataset": args.dataset, "correct": q["answerKey"],
                       "options": q["choices"]["text"], "question": q["question"],
                       "prompt": texts[j], "trace_ids": gen.tolist(), "trace": text,
                       "T": int(gen.shape[0]), "parsed": parsed, "closed_think": closed,
                       "waits": wait_positions(tok, gen), "reversals": revs,
                       "n_prompt_tokens": int((enc["attention_mask"][j] == 1).sum())}
                f.write(json.dumps(row) + "\n")
                f.flush()
                n += 1
            if args.device == "mps":
                torch.mps.empty_cache()
            print(f"  {n}/{len(qs)} traces, {(time.time() - t0) / max(n, 1):.0f} s each, "
                  f"{sum(1 for _ in open(path))} on file", flush=True)


def hidden_phase(args, out_dir):
    """One forward per trace with output_hidden_states; saves the last layer and a
    middle layer at every position, float16."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=getattr(torch, args.dtype)).to(args.device).eval()
    hid_dir = out_dir / "hidden"
    hid_dir.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in open(out_dir / "traces.jsonl")]
    n_layers = model.config.num_hidden_layers
    mid = n_layers // 2
    t0 = time.time()
    for i, r in enumerate(rows):
        f = hid_dir / f"{r['qid'].replace('/', '_')}.npz"
        if f.exists():
            continue
        pid = tok(r["prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(args.device)
        ids = torch.cat([pid, torch.tensor([r["trace_ids"]], device=args.device)], 1)
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
        P = pid.shape[1]
        np.savez_compressed(f,
                            last=out.hidden_states[-1][0, P - 1:].to(torch.float16).cpu().numpy(),
                            mid=out.hidden_states[mid][0, P - 1:].to(torch.float16).cpu().numpy(),
                            layer_last=n_layers, layer_mid=mid, n_prompt=P)
        del out
        if args.device == "mps":
            torch.mps.empty_cache()
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(rows)} hidden states, {(time.time() - t0) / (i + 1):.1f} s each", flush=True)
    print(f"hidden states in {hid_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", required=True, choices=("generate", "hidden"))
    p.add_argument("--dataset", default="arc_challenge", choices=("arc_challenge", "arc_easy", "gsm8k"))
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="mps")
    # float16, not float32: generation is greedy (argmax is robust to fp16) and the probe is
    # fitted and applied on the same fp16 hidden states, so no cross-precision comparison is
    # made. fp32 (6 GB) plus batch-4 KV caches filled 27 GB of swap on this machine (2026-09-09).
    p.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    p.add_argument("--strip-eos", action="store_true", default=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out_dir = Path(args.out) if args.out else ROOT / "results" / f"traces_{args.dataset.replace(chr(39), '').replace('arc_', '')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (generate_phase if args.phase == "generate" else hidden_phase)(args, out_dir)


if __name__ == "__main__":
    main()
