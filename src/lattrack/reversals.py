"""Find traces that genuinely change their final answer, from a downloaded corpus.

Source: uzaymacar/openr1_math_backtracking_dataset (193,767 DeepSeek-R1 traces over
OpenR1-Math-220k, MIT, ungated), which ships a `has_backtracking` flag. That flag is
NOT what we need: it marks any doubling back, and the published breakdown of
backtracking scope (ReasonOps 2605.29192, Table 7: GPQA-Diamond) is 85.4% local,
12.8% sub-problem, 1.6% global. A local re-derivation is not a change of mind about
the answer, and validating an answer-leaning readout against one would be a mistake
(the readout would correctly show no change and we would wrongly call it blind).

So we recompute the label from the text. A trace COMMITS to an answer whenever it
writes \\boxed{...}; the sequence of boxed values in order is the sequence of
commitments. A trace is a GLOBAL REVERSAL when its last commitment differs from some
earlier commitment — the model said one answer and finished on another. Values are
normalised before comparison (LaTeX spacing, \\dfrac -> \\frac, trailing zeros,
$ and \\! stripped, numeric forms compared as numbers) so that "0.5" and "\\frac{1}{2}"
do not count as a change, which would manufacture reversals out of formatting.

Reported for each reversal: the character offset of the switch, the two answers, the
switch as a fraction of the trace, whether the final commitment equals the dataset's
gold `solution` answer, and whether the trace states the switch in words. The last is
the strongest ground truth: the text both names a different answer AND says so.

    .venv/bin/python src/lattrack/reversals.py --input data/backtracking/slice.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sets import ROOT, write_json

# an explicit statement that a previous answer was wrong (same family as traces.py,
# plus the arithmetic-specific phrasings that appear in maths traces)
TAIL_CHARS = 400        # window at the end of a trace treated as its answer block
MIN_SEPARATION = 400    # a commitment must precede the final one by this many characters

STATED = re.compile(
    r"(actually,?\s+(it'?s|the answer is|i think)|no,?\s+(wait|it'?s|the answer is)|"
    r"wait,?\s+(no|actually)|i was wrong|i made a mistake|that'?s (wrong|incorrect|not right)|"
    r"change my answer|scratch that|correction|on second thought|"
    r"let me recompute|recalculating|i miscalculated|that can'?t be right)", re.I)


def iter_records(path, limit=None):
    """Stream complete objects out of a pretty-printed JSON array, tolerating a file
    truncated mid-record (we download a byte range, not the whole 3.73 GB)."""
    dec = json.JSONDecoder()
    buf = open(path, encoding="utf-8", errors="replace").read()
    i = buf.find("{")
    n = 0
    while i != -1:
        try:
            obj, end = dec.raw_decode(buf, i)
        except ValueError:
            break                       # truncated tail
        yield obj
        n += 1
        if limit and n >= limit:
            return
        i = buf.find("{", end)


BOXED = re.compile(r"\\boxed\s*\{")


def boxed_values(text):
    """Every \\boxed{...} payload in order, brace-matched (payloads nest braces)."""
    out = []
    for m in BOXED.finditer(text):
        i = m.end() - 1
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j < len(text):
            out.append((m.start(), text[i + 1:j]))
    return out


def normalise(v):
    """Collapse formatting so that only a real change of value counts as a change.
    Whitespace and newlines are collapsed first: the same multi-line LaTeX printed
    twice with different line breaks was a false positive in v2 (2026-09-09)."""
    s = re.sub(r"\\s+", " ", v).strip()
    for a, b in (("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\left", ""), ("\\right", ""),
                 ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""), ("$", ""), ("~", ""),
                 ("{,}", "")):
        s = s.replace(a, b)
    # thousands separators only. Stripping every comma turned the LIST "0, 1, 2" into
    # the number 12, which manufactured reversals out of traces where the model was
    # merely deciding how to format a multi-answer list (2026-09-09).
    s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)
    if "," in s:
        return "list:" + re.sub(r"\s+", "", s)     # a list of answers, never a number
    s = s.replace(" ", "")
    s = s.rstrip(".").strip()
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b:
            s = f"num:{a / b:.6g}"
            return s
    try:
        s = f"num:{float(s):.6g}"
    except ValueError:
        pass
    return s


def analyse(rec):   # rec: one dataset record
    text = rec.get("text") or ""
    vals = [(pos, v) for pos, v in boxed_values(text) if v.strip()]   # drop empty boxes
    norm = [(pos, normalise(v), v) for pos, v in vals]
    uniq = []
    for pos, nv, raw in norm:
        if not uniq or uniq[-1][1] != nv:
            uniq.append((pos, nv, raw))
    out = {"n_boxed": len(vals), "n_distinct_runs": len(uniq),
           "has_backtracking": rec.get("has_backtracking"),
           "correct": rec.get("correctness_math_verify"),
           "complete": rec.get("is_reasoning_complete"),
           "len_chars": len(text)}
    if len(uniq) < 2:
        out["reversal"] = False
        return out
    final = uniq[-1][1]
    # A multi-part problem ends with several DIFFERENT boxed values close together
    # ("the ship is \\boxed{10} km/h and the river is \\boxed{4}"). That is a list of
    # answers, not a change of mind, and it was the dominant false positive in the
    # first version of this filter (2026-09-09). Detect it and exclude it.
    tail_start = min(max(0, len(text) - TAIL_CHARS), int(len(text) * 0.85))
    tail_vals = {nv for pos, nv, _ in norm if pos >= tail_start}
    # sub-question markers in the PROBLEM are the reliable tell for a multi-answer task:
    # "(1) ... (2) ...", "a) ... b) ...". Parts can sit far apart in the trace, so the
    # tail window alone missed them (2026-09-09).
    problem = rec.get("problem") or ""
    out["multi_question"] = bool(re.search(r"\(\s*[2-9]\s*\)|\b[b-d]\s*\)", problem))
    out["multi_part"] = len(tail_vals) > 1 or out["multi_question"]
    # the earlier commitment must be genuinely earlier, not adjacent in the same block
    switch_pos = uniq[-1][0]
    earlier = [u for u in uniq[:-1] if u[1] != final and switch_pos - u[0] >= MIN_SEPARATION]
    # Numeric commitments only. A symbolic answer can be rewritten without changing
    # value (floor((n^2-2)/2) vs floor(n^2/2)-1), or be a case split ("1/c if ..., else
    # ..."), and both looked like reversals in v2. Numbers make "the value changed" a
    # decidable question, and a closed numeric answer set is what the readout needs.
    out["numeric"] = final.startswith("num:") and all(u[1].startswith("num:") for u in uniq)
    out["reversal"] = bool(earlier) and not out["multi_part"] and out["numeric"]
    if not out["reversal"]:
        return out
    window = text[max(0, switch_pos - 600):switch_pos]
    out.update({
        "from": earlier[-1][2], "to": uniq[-1][2],
        "from_norm": earlier[-1][1], "to_norm": final,
        "switch_char": switch_pos, "switch_frac": round(switch_pos / max(len(text), 1), 3),
        "n_commitments": len(uniq),
        "stated": bool(STATED.search(window)),
        "stated_phrase": (STATED.search(window).group(0) if STATED.search(window) else None),
        "context": window[-260:].replace("\n", " "),
    })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=str(ROOT / "results" / "reversals"))
    p.add_argument("--show", type=int, default=6)
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    stats = {"traces": 0, "has_backtracking": 0, "reversal": 0, "reversal_and_flagged": 0,
             "reversal_not_flagged": 0, "stated": 0, "single_boxed": 0, "no_boxed": 0,
             "reversal_correct_final": 0, "complete": 0, "multi_part_excluded": 0,
             "stated_and_flagged": 0, "non_numeric_excluded": 0}
    keep = []
    with open(out_dir / "reversal_rows.jsonl", "w") as f:
        for rec in iter_records(args.input, args.limit):
            a = analyse(rec)
            n += 1
            stats["traces"] += 1
            stats["has_backtracking"] += bool(a["has_backtracking"])
            stats["complete"] += bool(a["complete"])
            if a["n_boxed"] == 0:
                stats["no_boxed"] += 1
            elif a["n_distinct_runs"] == 1:
                stats["single_boxed"] += 1
            stats["multi_part_excluded"] += bool(a.get("multi_part"))
            stats["non_numeric_excluded"] += (a.get("numeric") is False)
            if a["reversal"]:
                stats["reversal"] += 1
                stats["reversal_and_flagged"] += bool(a["has_backtracking"])
                stats["reversal_not_flagged"] += not a["has_backtracking"]
                stats["stated"] += bool(a.get("stated"))
                stats["reversal_correct_final"] += bool(a["correct"])
                stats["stated_and_flagged"] += bool(a.get("stated")) and bool(a["has_backtracking"])
                row = {"problem": (rec.get("problem") or "")[:400], **a}
                f.write(json.dumps(row) + "\n")
                keep.append(row)

    flagged = stats["has_backtracking"] or 1
    summary = {
        "source": args.input, **stats,
        "rate_flagged_of_all": round(stats["has_backtracking"] / max(stats["traces"], 1), 4),
        "rate_reversal_of_all": round(stats["reversal"] / max(stats["traces"], 1), 4),
        "rate_reversal_of_flagged": round(stats["reversal_and_flagged"] / flagged, 4),
        "rate_stated_of_reversal": round(stats["stated"] / max(stats["reversal"], 1), 4),
        "tail_chars": TAIL_CHARS, "min_separation_chars": MIN_SEPARATION,
        "definition": ("commitment = a \\boxed{} payload; reversal = the last commitment differs "
                       "from an earlier one after normalising LaTeX/number formatting"),
    }
    write_json(out_dir / "reversal_summary.json", summary)
    print(json.dumps(summary, indent=1))
    print(f"\nfirst {args.show} reversals:")
    for r in keep[:args.show]:
        print(f"  {r['from']} -> {r['to']}  at {r['switch_frac']:.0%} of the trace, "
              f"{r['n_commitments']} commitments, flagged={r['has_backtracking']}, "
              f"stated={r.get('stated')} {r.get('stated_phrase') or ''}")
        print(f"     ...{r['context'][-180:]}")


if __name__ == "__main__":
    main()
