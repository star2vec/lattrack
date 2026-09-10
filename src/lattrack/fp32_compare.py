"""bf16 vs fp32 option readout on the same Huginn trajectories.

The fp32-letters rerun (huginn_lens.py --readout fp32-letters) recomputes the eight letter
logits in fp32 from lm_head's input and keeps the model's own bf16 logits per row as
`steps_bf16`. Two questions: (1) are the bf16 steps bit-identical to the earlier bf16 run,
so the only thing that changed is the readout; (2) which of the earlier numbers move when
the quantisation ties are gone — per-transition change counts, the within-ulp fraction, the
Cui-Ye event rate, real-pair vs distractor-pair crossing (RESULT 2), the last-change loop
(RESULTS 10/11), late movers, final answers and accuracy.

    .venv/bin/python src/lattrack/fp32_compare.py [--pairs huginn:huginn_fp32,huginn_easy:huginn_easy_fp32]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from commitment import commitment
from huginn_lens import derive
from sets import ROOT, write_json
from stats import bootstrap

ULP = 0.13
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def load(d):
    p = ROOT / "results" / d / "lens_rows.jsonl"
    return {r["qid"]: r for r in (json.loads(l) for l in open(p)) if r["condition"] == "base"} if p.exists() else {}


def stats(S, rows, tag):
    """S: {qid: (K,4) array}; rows for correct letters. One dict of the numbers we compare."""
    qids = sorted(S)
    D = {q: derive(S[q].tolist(), "ABCD".index(rows[q]["correct_letter"])) for q in qids}
    chg = [x for q in qids for x in D[q]["changes"]]
    cs = [commitment(["ABCD"[i] for i in S[q].argmax(1)]) for q in qids]
    K = len(S[qids[0]])
    f = lambda v: bootstrap([float(x) for x in v], np.mean)
    return {
        "n": len(qids), "K": K,
        "changes_per_question": f([D[q]["n_changes"] for q in qids]),
        "transitions_within_ulp": (sum(1 for x in chg if min(x["margin_before"], x["margin_after"]) < ULP), len(chg)),
        "cui_ye_event": f([D[q]["cui_ye_event"] for q in qids]),
        "real_pair_any_cross": f([any(D[q]["real_cross"]) for q in qids]),
        "distractor_pairs_any_cross": f([np.mean([any(c) for c in D[q]["distractor_cross"]]) for q in qids]),
        "last_change_loop": f(cs), "last_change_median": float(np.median(cs)),
        "differs_at_8": sum(int(S[q][8].argmax() != S[q][-1].argmax()) for q in qids),
        "differs_at_16": sum(int(S[q][16].argmax() != S[q][-1].argmax()) for q in qids),
        "accuracy_final": f([S[q][-1].argmax() == "ABCD".index(rows[q]["correct_letter"]) for q in qids]),
        "accuracy_by_loop": [float(np.mean([S[q][k].argmax() == "ABCD".index(rows[q]["correct_letter"]) for q in qids])) for k in range(K)],
    }


def fmt(b):
    return f"{b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"


def compare(old_dir, new_dir):
    old, new = load(old_dir), load(new_dir)
    qids = sorted(set(old) & set(new))
    if not qids:
        say(f"\n## {new_dir}: no rows yet"); return None
    say(f"\n## {old_dir} (bf16) vs {new_dir} (fp32 letters): {len(qids)} shared questions")
    ident = max(float(np.abs(np.array(new[q]["steps_bf16"]) - np.array(old[q]["steps"])).max()) for q in qids)
    diff = [float(np.abs(np.array(new[q]["steps"]) - np.array(new[q]["steps_bf16"])).max()) for q in qids]
    say(f"- bf16 steps in the rerun vs the earlier run: max |diff| {ident:.2e} (must be 0)")
    say(f"- fp32 vs bf16 option logits, per-question max |diff|: median {np.median(diff):.3f}, max {max(diff):.3f}")
    B = stats({q: np.array(new[q]["steps_bf16"]) for q in qids}, new, "bf16")
    F = stats({q: np.array(new[q]["steps"]) for q in qids}, new, "fp32")
    say(f"- final answer (argmax at the last loop) agrees between readouts on "
        f"{sum(int(np.array(new[q]['steps'])[-1].argmax() == np.array(new[q]['steps_bf16'])[-1].argmax()) for q in qids)}/{len(qids)}")
    say("\n| quantity | bf16 readout | fp32 readout |\n|---|---|---|")
    say(f"| leader changes per question | {fmt(B['changes_per_question'])} | {fmt(F['changes_per_question'])} |")
    say(f"| transitions with a side margin < one ulp | {B['transitions_within_ulp'][0]}/{B['transitions_within_ulp'][1]} | {F['transitions_within_ulp'][0]}/{F['transitions_within_ulp'][1]} |")
    say(f"| Cui-Ye event rate | {fmt(B['cui_ye_event'])} | {fmt(F['cui_ye_event'])} |")
    say(f"| (correct, top distractor) pair ever crosses | {fmt(B['real_pair_any_cross'])} | {fmt(F['real_pair_any_cross'])} |")
    say(f"| distractor pairs ever cross | {fmt(B['distractor_pairs_any_cross'])} | {fmt(F['distractor_pairs_any_cross'])} |")
    say(f"| last-change loop, mean (median) | {fmt(B['last_change_loop'])} ({B['last_change_median']:.0f}) | {fmt(F['last_change_loop'])} ({F['last_change_median']:.0f}) |")
    say(f"| answer at loop 8 differs from final | {B['differs_at_8']}/{B['n']} | {F['differs_at_8']}/{F['n']} |")
    say(f"| answer at loop 16 differs from final | {B['differs_at_16']}/{B['n']} | {F['differs_at_16']}/{F['n']} |")
    say(f"| final accuracy | {fmt(B['accuracy_final'])} | {fmt(F['accuracy_final'])} |")
    pk = int(np.argmax(F["accuracy_by_loop"]))
    say(f"| best loop (fp32) | — | loop {pk}: {F['accuracy_by_loop'][pk]:.2f}; loop 14: {F['accuracy_by_loop'][14]:.2f}; last: {F['accuracy_by_loop'][-1]:.2f} |")
    return {"identity_max_diff": ident, "bf16": B, "fp32": F}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default="huginn:huginn_fp32,huginn_easy:huginn_easy_fp32")
    args = p.parse_args()
    say("# bf16 vs fp32 option readout, same trajectories (fp32_compare.py)")
    out = {}
    for pair in args.pairs.split(","):
        o, n = pair.split(":")
        r = compare(o, n)
        if r:
            out[n] = r
    (ROOT / "results" / "fp32_compare.md").write_text("\n".join(OUT) + "\n")
    write_json(ROOT / "results" / "fp32_compare.json", {"schema": "fp32-compare-v1", "pairs": out})


if __name__ == "__main__":
    main()
