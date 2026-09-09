"""GSM8K as four-option questions: the task where a small reasoning model actually
reverses itself, so that a stated reversal can serve as ground truth.

On ARC the 1.5B distill deliberates but does not self-correct (0 stated reversals in
28 traces, 2026-09-09), so there was nothing to validate the readout against. Grade-
school arithmetic is where R1-style traces backtrack. The answer is a number, so the
four-option readout needs constructed distractors.

Distractors, per question, all integers, all distinct from the answer and from each
other, drawn deterministically from a per-question seeded shuffle of these families:
  off-by-one and off-by-ten     a +/- 1, a +/- 10
  scale                         2a, a/2 (when integral), 10a, a/10 (when integral)
  digit                         the answer with its last two digits swapped
  numbers in the question       an integer appearing in the problem text, not equal to a
The first three that are valid and distinct are used, so a distractor is always a
plausible slip rather than a random integer, which is what makes a mid-trace change of
leader meaningful.

Writes data/gsm8k_options.json in the ARC schema ({question, choices:{text,label},
answerKey}) so load_questions() and render() take it unchanged.

    .venv/bin/python src/lattrack/mathopts.py --n 200
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq

from sets import ROOT

LETTERS = ["A", "B", "C", "D"]
SEED = 20260909


def answer_of(a):
    m = re.search(r"####\s*(-?[\d,]+)", a)
    return int(m.group(1).replace(",", "")) if m else None


def candidates(ans, question):
    out = [ans + 1, ans - 1, ans + 10, ans - 10, 2 * ans, 10 * ans]
    if ans % 2 == 0:
        out.append(ans // 2)
    if ans % 10 == 0:
        out.append(ans // 10)
    s = str(abs(ans))
    if len(s) >= 2:
        swapped = int(s[:-2] + s[-1] + s[-2]) * (1 if ans >= 0 else -1)
        out.append(swapped)
    out += [int(x.replace(",", "")) for x in re.findall(r"\b\d[\d,]*\b", question)]
    return out


def build(rows, n, seed=SEED):
    out = []
    for i, r in enumerate(rows):
        ans = answer_of(r["answer"])
        if ans is None:
            continue
        rng = random.Random((seed << 8) ^ i)
        pool = [c for c in candidates(ans, r["question"]) if c != ans and c > 0]
        seen, dis = set(), []
        rng.shuffle(pool)
        for c in pool:
            if c not in seen:
                seen.add(c)
                dis.append(c)
            if len(dis) == 3:
                break
        if len(dis) < 3:
            continue
        opts = dis + [ans]
        rng.shuffle(opts)
        out.append({"id": f"gsm8k_{i}", "question": r["question"].strip(),
                    "choices": {"text": [str(o) for o in opts], "label": LETTERS},
                    "answerKey": LETTERS[opts.index(ans)],
                    "gold_number": ans, "distractors": dis})
        if len(out) >= n:
            break
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", default=str(ROOT / "data" / "gsm8k_options.json"))
    args = p.parse_args()
    rows = pq.read_table(ROOT / "data" / "gsm8k_test.parquet").to_pylist()
    qs = build(rows, args.n)
    json.dump(qs, open(args.out, "w"))
    from collections import Counter
    print(f"{len(qs)} four-option GSM8K questions -> {args.out}")
    print("answer position:", dict(Counter(q["answerKey"] for q in qs)))
    q = qs[0]
    print("example:", q["question"][:90].replace("\n", " "), "|", q["choices"]["text"], "|", q["answerKey"])


if __name__ == "__main__":
    main()
