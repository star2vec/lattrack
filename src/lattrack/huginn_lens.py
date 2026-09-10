"""Per-loop lens on Huginn-3.5B (tomg-group-umd/huginn-0125) for four-choice QA.

The model is prelude (2 blocks) -> recurrent block (4 blocks, iterated) -> coda
(2 blocks) -> LM head. The recurrent state starts from a random draw
(`initialize_state`), so every forward is a sample unless torch is seeded; the
init seed is therefore a noise source we measure, like the serialization redraws
on the 2-layer model. The per-step readout is the model's own
`predict_from_latents` (ln_f -> coda -> ln_f -> lm_head) applied to the state
after each recurrent step, which is what Cui & Ye 2602.08100 call the belief at
step i (the distribution the model would output if halted there). We read the
logits at the last prompt position over the four option letters.

Per question and condition, per step k = 1..K: the four option logits, the
probabilities renormalised over the four, the leader, the margin (top-1 minus
top-2 logit), the rank of the correct option, KL(p_k || p_{k-1}) over the four.

Conditions per question: base (init seed 0, options in the file's order);
init_1, init_2 (init seeds 1 and 2, same prompt: the noise floor);
perm (init seed 0, options rotated by one place: the position cell, a prompt
change, reported apart from the noise).

Distractor-pair check (the arbitrary-pair null of the 2-layer run, adapted):
per transition, the crossing rate of the pair (correct option, strongest
distractor at the last step) against the mean crossing rate of the three
distractor-distractor pairs. Leader changes of the 4-way argmax are counted raw
and under Cui & Ye's event definition (argmax a for >= 3 consecutive steps, later
b != a for >= 3 consecutive steps, final answer b). No thresholds applied.

Tripwires on the first question: (1) two forwards under the same init seed give
identical logits; (2) the step-K lens equals the model's own forward with
num_steps=K under the same seed (max abs diff reported, must be < 1e-3).

    .venv/bin/python src/lattrack/huginn_lens.py --n 5 --device mps          # timing
    .venv/bin/python src/lattrack/huginn_lens.py --n 50 --device mps
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pyarrow.parquet as pq
import torch

from sets import ROOT, write_json
from stats import bootstrap

SCHEMA = "huginn-lens-v1"
MODEL_ID = "tomg-group-umd/huginn-0125"
LETTERS = ["A", "B", "C", "D"]
QUESTION_SEED = 20260908  # which ARC questions
SYSTEM = "You are a helpful assistant."


# --- data -----------------------------------------------------------------------

DATASET = "arc_challenge"   # set from --dataset


def load_questions(n, seed=QUESTION_SEED, dataset=None):
    """n ARC test questions (Challenge or Easy) with exactly four lettered options,
    drawn by a seeded shuffle of the file order."""
    dataset = dataset or DATASET
    if dataset == "gsm8k":   # built by mathopts.py, already in the ARC schema
        rows = json.load(open(ROOT / "data" / "gsm8k_options.json"))
    else:
        rows = pq.read_table(ROOT / "data" / f"{dataset}_test.parquet").to_pylist()
    ok = [r for r in rows if len(r["choices"]["text"]) == 4 and r["answerKey"] in LETTERS
          and r["choices"]["label"] == LETTERS]
    rng = random.Random(seed)
    rng.shuffle(ok)
    return ok[:n]


def render(q, options):
    body = q["question"].strip() + "\n" + "\n".join(f"{L}. {t}" for L, t in zip(LETTERS, options))
    return body + "\nAnswer with the letter only."


def conditions(q):
    """[(name, options, correct_letter, init_seed)]"""
    opts = list(q["choices"]["text"])
    correct = LETTERS.index(q["answerKey"])
    out = [("base", opts, LETTERS[correct], 0),
           ("init_1", opts, LETTERS[correct], 1),
           ("init_2", opts, LETTERS[correct], 2)]
    # cyclic rotations: option at index i moves to i-k; the correct one moves too. With all
    # three, every option appears in every position once across base+perm+perm2+perm3 (the
    # cheap stand-in for Cui & Ye's 25 random permutations).
    for k, name in ((1, "perm"), (2, "perm2"), (3, "perm3")):
        rot = opts[k:] + opts[:k]
        out.append((name, rot, LETTERS[(correct - k) % 4], 0))
    return out


# --- model ------------------------------------------------------------------------

def load_model(device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # low_cpu_mem_usage: materialise shard by shard in the target dtype (the checkpoint is
    # float32, 15.6 GB; the machine has 16 GB), needs `accelerate`
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, trust_remote_code=True,
                                                 low_cpu_mem_usage=True)
    model = model.to(device).eval()
    return tok, model


PROMPT_STYLE = "chat"   # set from --prompt-style; recorded in the summary conditions


def prompt_ids(tok, text, device, style=None):
    """chat: system + user turn, the letter is the first assistant token.
    chat_prefill: same, with the assistant turn prefilled "The correct answer is",
    the model's own phrasing under greedy decoding (checked 2026-09-08: under
    `chat` its first token is "The" with 40-60% mass and it writes "The correct
    answer is X."), so the letter follows as " A".
    plain: no template; "Question: ...\\nA. ...\\nAnswer:" as for a base model."""
    style = style or PROMPT_STYLE
    if style == "plain":
        s = tok.bos_token + "Question: " + text.replace("\nAnswer with the letter only.", "") + "\nAnswer:"
    else:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if style == "chat_prefill":
            s = s + "The correct answer is"
    return torch.tensor([tok(s, add_special_tokens=False)["input_ids"]], device=device)


class LetterHead:
    """fp32 option logits from the hidden state entering lm_head.

    The model's own path is bf16 end to end and one bf16 ulp at |logit|~16 is 0.125; in
    the bf16 runs 65-70% of leader changes had a margin under one ulp on one side (RESULT
    10 addendum), so per-transition counts were inflated by quantisation ties. A forward
    pre-hook captures lm_head's input (after the second ln_f) and dots it with the eight
    letter rows in fp32: 8 x 5280 weights, negligible memory. This removes the quantisation
    of the logits themselves; the hidden state feeding them is still bf16."""

    def __init__(self, model, letter_ids):
        self.W = model.lm_head.weight[letter_ids.flatten()].detach().float()   # (8, d)
        self.shape = tuple(letter_ids.shape)
        self.h = None
        self.handle = model.lm_head.register_forward_pre_hook(self._grab)

    def _grab(self, module, inputs):
        self.h = inputs[0][0, -1].detach()

    def letters(self):
        return (self.h.float() @ self.W.T).view(self.shape)                    # (2, 4) fp32


@torch.no_grad()
def lens_trajectory(model, ids, letter_ids, K, init_seed, head=None):
    """Per-step logits over the four letters at the last position, plus the full
    logit row at the last step (for the tripwire). With `head` (LetterHead) the option
    logits come from the fp32 letter readout and the model's own bf16 ones are returned
    alongside as `steps_bf16`; without it, steps are the bf16 ones and steps_bf16 is None."""
    torch.manual_seed(init_seed)
    input_embeds, block_idx = model.embed_inputs(ids)
    x = model.initialize_state(input_embeds)
    steps, steps_bf16, raw, mass, argmax_letter = [], [], [], [], []
    letter_set = set(letter_ids.flatten().tolist())
    last_row = None
    for k in range(K):
        x, block_idx, _ = model.iterate_one_step(input_embeds, x, block_idx=block_idx, current_step=k)
        out = model.predict_from_latents(x)
        row = out.logits[0, -1].float()
        both = row[letter_ids]                                   # (2, 4): "A".."D" and " A".." D"
        steps_bf16.append(torch.logsumexp(both, dim=0).tolist())
        if head is not None:
            both = head.letters()
        steps.append(torch.logsumexp(both, dim=0).tolist())     # combined option logit
        raw.append(both.tolist())
        pv = torch.softmax(row, -1)
        mass.append(float(pv[letter_ids].sum()))                 # vocab mass on the eight letter tokens
        argmax_letter.append(int(row.argmax()) in letter_set)    # is a letter the model's next token
        last_row = row
    return steps, last_row, raw, mass, argmax_letter, (steps_bf16 if head is not None else None)


@torch.no_grad()
def model_forward_last(model, ids, K, init_seed):
    torch.manual_seed(init_seed)
    out = model(ids, num_steps=int(K), output_details={"return_logits": True, "return_latents": False,
                                                       "return_head": False, "return_stats": False})
    return out.logits[0, -1].float()


# --- per-row derived quantities --------------------------------------------------

def derive(steps, correct_idx):
    """steps: list over k of the four letter logits."""
    L = np.asarray(steps, dtype=float)                      # (K, 4)
    P = np.exp(L - L.max(1, keepdims=True))
    P = P / P.sum(1, keepdims=True)                         # renormalised over the four
    leader = L.argmax(1)
    srt = np.sort(L, 1)
    margin = srt[:, -1] - srt[:, -2]
    rank_correct = (L > L[:, [correct_idx]]).sum(1) + 1
    kl = [None] + [float((P[k] * np.log((P[k] + 1e-12) / (P[k - 1] + 1e-12))).sum()) for k in range(1, len(L))]
    changes = [{"t": k, "from": int(leader[k]), "to": int(leader[k + 1]),
                "margin_before": float(margin[k]), "margin_after": float(margin[k + 1])}
               for k in range(len(L) - 1) if leader[k] != leader[k + 1]]
    final = int(leader[-1])
    # Cui & Ye event: argmax a for >= 3 consecutive steps, later b != a for >= 3 consecutive, final == b
    runs = []
    for k, v in enumerate(leader):
        if runs and runs[-1][0] == v:
            runs[-1][2] = k
        else:
            runs.append([int(v), k, k])
    long_runs = [(v, s, e) for v, s, e in runs if e - s + 1 >= 3]
    event = False
    for i, (a, _, ea) in enumerate(long_runs):
        for b, sb, _ in long_runs[i + 1:]:
            if b != a and b == final:
                event = True
    # pair crossings per transition: the correct option against the strongest distractor
    # at the last step, and the three distractor-distractor pairs
    order_last = [int(i) for i in np.argsort(-L[-1])]
    dis = [i for i in order_last if i != correct_idx]
    real_pair = (correct_idx, dis[0])
    dis_pairs = [(dis[0], dis[1]), (dis[0], dis[2]), (dis[1], dis[2])]

    def crossings(a, b):
        g = L[:, a] - L[:, b]
        return [bool(g[k] * g[k + 1] < 0) for k in range(len(g) - 1)]
    return {
        "probs": P.round(5).tolist(), "leader": leader.tolist(), "margin": margin.round(4).tolist(),
        "rank_correct": rank_correct.tolist(), "kl": [None if v is None else round(v, 6) for v in kl],
        "changes": changes, "n_changes": len(changes), "final": final, "correct": final == correct_idx,
        "cui_ye_event": event, "runs": runs,
        "real_pair": list(real_pair), "real_cross": crossings(*real_pair),
        "distractor_cross": [crossings(*p) for p in dis_pairs],
        "distractor_pairs": [list(p) for p in dis_pairs],
    }


# --- summary -----------------------------------------------------------------------

def summarize(rows, K):
    base = [r for r in rows if r["condition"] == "base"]
    by = {(r["qid"], r["condition"]): r for r in rows}
    out = {"schema": SCHEMA, "n_questions": len(base), "K": K}
    d = lambda r: r["derived"]
    q = lambda v: {f"q{p}": float(np.percentile(v, p)) for p in (10, 50, 90)} if v else None
    out["accuracy_base"] = bootstrap([float(d(r)["correct"]) for r in base], np.mean)
    if base and "letter_mass" in base[0]:
        out["readout_position_check"] = {
            "letter_mass_last_median": float(np.median([r["letter_mass"][-1] for r in base])),
            "letter_mass_step1_median": float(np.median([r["letter_mass"][0] for r in base])),
            "argmax_is_letter_last": bootstrap([float(r["argmax_is_letter"][-1]) for r in base], np.mean),
            "argmax_is_letter_by_step": [round(float(np.mean([r["argmax_is_letter"][k] for r in base])), 3)
                                         for k in range(K)],
        }
    out["accuracy_perm"] = bootstrap([float(d(by[(r["qid"], "perm")])["correct"]) for r in base if (r["qid"], "perm") in by], np.mean)
    rots = [c for c in ("base", "perm", "perm2", "perm3") if any((r["qid"], c) in by for r in base)]
    if len(rots) > 1:
        full = [r for r in base if all((r["qid"], c) in by for c in rots)]
        out["rotations"] = {
            "orders": rots, "n_questions_with_all": len(full),
            "accuracy_by_order": {c: bootstrap([float(d(by[(r["qid"], c)])["correct"]) for r in full], np.mean) for c in rots},
            "accuracy_mean_over_orders": bootstrap([np.mean([float(d(by[(r["qid"], c)])["correct"]) for c in rots]) for r in full], np.mean),
            "correct_under_every_order": bootstrap([float(all(d(by[(r["qid"], c)])["correct"] for c in rots)) for r in full], np.mean),
            "cui_ye_event_rate_mean_over_orders": bootstrap([np.mean([float(d(by[(r["qid"], c)])["cui_ye_event"]) for c in rots]) for r in full], np.mean),
            "leader_changes_mean_over_orders": bootstrap([np.mean([float(d(by[(r["qid"], c)])["n_changes"]) for c in rots]) for r in full], np.mean),
        }
    out["questions_with_any_leader_change"] = bootstrap([float(d(r)["n_changes"] > 0) for r in base], np.mean)
    out["mean_leader_changes_per_question"] = bootstrap([float(d(r)["n_changes"]) for r in base], np.mean)
    out["cui_ye_backtracking_rate"] = bootstrap([float(d(r)["cui_ye_event"]) for r in base], np.mean)
    ev = [r for r in base if d(r)["cui_ye_event"]]
    out["accuracy_on_backtracking_questions"] = bootstrap([float(d(r)["correct"]) for r in ev], np.mean) if ev else None
    out["accuracy_on_other_questions"] = bootstrap([float(d(r)["correct"]) for r in base if not d(r)["cui_ye_event"]], np.mean)
    # crossing rates by step bin (transitions k -> k+1, k = 0..K-2), real pair vs distractor pairs
    bins = [(0, 3), (3, 7), (7, 15), (15, K - 1)]
    tab = {}
    for lo, hi in bins:
        name = f"steps {lo + 1}→{hi + 1}"
        real = [float(any(d(r)["real_cross"][lo:hi])) for r in base]
        dis = [float(np.mean([any(c[lo:hi]) for c in d(r)["distractor_cross"]])) for r in base]
        tab[name] = {"real_pair_any_crossing": bootstrap(real, np.mean),
                     "distractor_pairs_any_crossing": bootstrap(dis, np.mean)}
    out["crossing_by_step_bin"] = tab
    per_step_real = [np.mean([d(r)["real_cross"][k] for r in base]) for k in range(K - 1)]
    per_step_dis = [np.mean([np.mean([c[k] for c in d(r)["distractor_cross"]]) for r in base]) for k in range(K - 1)]
    out["crossing_per_step"] = {"real": [round(float(v), 4) for v in per_step_real],
                                "distractor": [round(float(v), 4) for v in per_step_dis]}
    # answering phase: transitions where a letter is the model's next token at both steps
    # (early steps have ~0 vocab mass on the letters; changes there are between near-equal
    # tiny logits, not answers)
    if base and "argmax_is_letter" in base[0]:
        def valid(r):
            a = r["argmax_is_letter"]
            return [a[k] and a[k + 1] for k in range(len(a) - 1)]
        onset = [next((k + 1 for k, v in enumerate(r["argmax_is_letter"]) if v), None) for r in base]
        out["letter_onset_step"] = {"median": float(np.median([o for o in onset if o])) if any(onset) else None,
                                    "q90": float(np.percentile([o for o in onset if o], 90)) if any(onset) else None,
                                    "never": int(sum(o is None for o in onset))}
        real_v = [float(any(c for c, v in zip(d(r)["real_cross"], valid(r)) if v)) for r in base]
        dis_v = [float(np.mean([any(c for c, v in zip(cc, valid(r)) if v) for cc in d(r)["distractor_cross"]]))
                 for r in base]
        chg_v = [float(sum(1 for c in d(r)["changes"] if valid(r)[c["t"]])) for r in base]
        mb_v = [c["margin_before"] for r in base for c in d(r)["changes"] if valid(r)[c["t"]]]
        ma_v = [c["margin_after"] for r in base for c in d(r)["changes"] if valid(r)[c["t"]]]
        out["answering_phase"] = {
            "definition": "transitions where a letter is the model's next token at both steps",
            "real_pair_any_crossing": bootstrap(real_v, np.mean),
            "distractor_pairs_any_crossing": bootstrap(dis_v, np.mean),
            "mean_leader_changes": bootstrap(chg_v, np.mean),
            "questions_with_leader_change": bootstrap([float(x > 0) for x in chg_v], np.mean),
            "margin_at_changes": {"before": q(mb_v), "after": q(ma_v), "n": len(mb_v)},
        }
    # where leader changes happen (last change step), and margins at changes
    last_change = [max(c["t"] for c in d(r)["changes"]) + 1 for r in base if d(r)["changes"]]
    out["last_leader_change_step"] = {"median": float(np.median(last_change)) if last_change else None,
                                      "q90": float(np.percentile(last_change, 90)) if last_change else None,
                                      "n": len(last_change)}
    mb = [c["margin_before"] for r in base for c in d(r)["changes"]]
    ma = [c["margin_after"] for r in base for c in d(r)["changes"]]
    out["margin_at_changes"] = {"before": q(mb), "after": q(ma), "n": len(mb)}
    # init-seed noise: |Δlogit gap| of the real pair per step, base vs init_1/init_2; leader agreement; change-set agreement
    diffs, agree, same_set = [], [], []
    for r in base:
        a, b = d(r)["real_pair"]
        L0 = np.asarray(r["steps"])
        for c in ("init_1", "init_2"):
            x = by.get((r["qid"], c))
            if x is None:
                continue
            L1 = np.asarray(x["steps"])
            diffs.extend(np.abs((L0[:, a] - L0[:, b]) - (L1[:, a] - L1[:, b])).tolist())
            agree.append(float(np.mean(np.asarray(d(r)["leader"]) == np.asarray(d(x)["leader"]))))
            same_set.append(float({(c_["t"], c_["from"], c_["to"]) for c_ in d(r)["changes"]}
                                  == {(c_["t"], c_["from"], c_["to"]) for c_ in d(x)["changes"]}))
    out["init_seed_noise"] = {"abs_gap_change": q(diffs), "leader_agreement_per_step": bootstrap(agree, np.mean),
                              "change_set_identical": bootstrap(same_set, np.mean), "n_pairs": len(agree)}
    # position cell
    agree_p, same_final = [], []
    for r in base:
        x = by.get((r["qid"], "perm"))
        if x is None:
            continue
        same_final.append(float(d(r)["correct"] == d(x)["correct"]))
        agree_p.append(float(d(r)["n_changes"] == d(x)["n_changes"]))
    out["perm_cell"] = {"same_correctness": bootstrap(same_final, np.mean),
                        "same_number_of_changes": bootstrap(agree_p, np.mean)}
    # exploration end per Cui & Ye: first step after which KL <= 0.01 for 3 consecutive steps
    ends = []
    for r in base:
        kl = d(r)["kl"]
        end = None
        for k in range(1, len(kl) - 2):
            if all(kl[j] is not None and kl[j] <= 0.01 for j in (k, k + 1, k + 2)):
                end = k
                break
        ends.append(end)
    out["exploration_end_step"] = {"median": float(np.median([e for e in ends if e is not None])) if any(e is not None for e in ends) else None,
                                   "never": int(sum(e is None for e in ends))}
    return out


def fmt(b):
    return "—" if b is None else f"{b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}] (n={b['n']})"


def table(s, cond):
    ds = (s.get("conditions") or {}).get("dataset", cond.get("dataset", "ARC"))
    lines = [f"**Huginn-0125 per-loop lens, {ds}, {s['n_questions']} questions, K={s['K']}** "
             f"({cond['device']}, {cond['dtype']}, {cond['seconds_per_trajectory']:.1f} s per trajectory of {s['K']} steps)",
             f"accuracy: base {fmt(s['accuracy_base'])}, options rotated {fmt(s['accuracy_perm'])}",
             (f"readout position: vocab mass on the letter tokens at the last step median "
              f"{s['readout_position_check']['letter_mass_last_median']:.3f} (step 1: "
              f"{s['readout_position_check']['letter_mass_step1_median']:.3f}); a letter is the model's next "
              f"token at the last step {fmt(s['readout_position_check']['argmax_is_letter_last'])}"
              if s.get("readout_position_check") else "readout position: not recorded"),
             f"questions with any 4-way leader change {fmt(s['questions_with_any_leader_change'])}; "
             f"mean changes per question {fmt(s['mean_leader_changes_per_question'])}; "
             f"last change at step median {s['last_leader_change_step']['median']}, q90 {s['last_leader_change_step']['q90']}",
             f"Cui & Ye backtracking events {fmt(s['cui_ye_backtracking_rate'])}; accuracy on those "
             f"{fmt(s['accuracy_on_backtracking_questions'])} vs others {fmt(s['accuracy_on_other_questions'])}",
             "", "| transitions | (correct, top distractor) crossing | distractor-distractor pairs crossing |", "|---|---|---|"]
    for k, v in s["crossing_by_step_bin"].items():
        lines.append(f"| {k} | {fmt(v['real_pair_any_crossing'])} | {fmt(v['distractor_pairs_any_crossing'])} |")
    ap = s.get("answering_phase")
    if ap:
        o = s["letter_onset_step"]
        lines.append(f"answering phase ({ap['definition']}); letter onset at step median {o['median']}, q90 {o['q90']}, "
                     f"never {o['never']}: questions with a leader change {fmt(ap['questions_with_leader_change'])}; "
                     f"mean changes {fmt(ap['mean_leader_changes'])}; (correct, top distractor) crossing "
                     f"{fmt(ap['real_pair_any_crossing'])} vs distractor pairs {fmt(ap['distractor_pairs_any_crossing'])}")
        mv = ap["margin_at_changes"]
        if mv["before"]:
            lines.append(f"  margins at answering-phase changes: before q10/50/90 {mv['before']['q10']:.2f}/"
                         f"{mv['before']['q50']:.2f}/{mv['before']['q90']:.2f}, after {mv['after']['q10']:.2f}/"
                         f"{mv['after']['q50']:.2f}/{mv['after']['q90']:.2f} (n={mv['n']})")
    m = s["margin_at_changes"]
    if m["before"]:
        lines.append(f"margins at leader changes (top-1 minus top-2 logit): before q10/50/90 "
                     f"{m['before']['q10']:.2f}/{m['before']['q50']:.2f}/{m['before']['q90']:.2f}, after "
                     f"{m['after']['q10']:.2f}/{m['after']['q50']:.2f}/{m['after']['q90']:.2f} (n={m['n']})")
    n = s["init_seed_noise"]
    if n["abs_gap_change"]:
        lines.append(f"init-seed noise (2 reseeds): |Δgap| q10/50/90 {n['abs_gap_change']['q10']:.2f}/"
                     f"{n['abs_gap_change']['q50']:.2f}/{n['abs_gap_change']['q90']:.2f}; leader agreement per step "
                     f"{fmt(n['leader_agreement_per_step'])}; change set identical {fmt(n['change_set_identical'])}")
    lines.append(f"position cell (options rotated): same correctness {fmt(s['perm_cell']['same_correctness'])}; "
                 f"same number of changes {fmt(s['perm_cell']['same_number_of_changes'])}")
    ro = s.get("rotations")
    if ro:
        lines.append(f"all {len(ro['orders'])} option orders (n={ro['n_questions_with_all']}): accuracy by order "
                     + ", ".join(f"{c} {v['point']:.2f}" for c, v in ro["accuracy_by_order"].items())
                     + f"; mean over orders {fmt(ro['accuracy_mean_over_orders'])}; correct under every order "
                     f"{fmt(ro['correct_under_every_order'])}; Cui & Ye event rate mean over orders "
                     f"{fmt(ro['cui_ye_event_rate_mean_over_orders'])}; leader changes mean over orders "
                     f"{fmt(ro['leader_changes_mean_over_orders'])}")
    lines.append(f"exploration end (KL <= 0.01 for 3 steps): median step {s['exploration_end_step']['median']}, "
                 f"never {s['exploration_end_step']['never']}")
    return "\n".join(lines)


# --- driver -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--conditions", default="base,init_1,init_2,perm")
    p.add_argument("--out", default=str(ROOT / "results" / "huginn"))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--prompt-style", default="chat", choices=("chat", "chat_prefill", "plain"))
    p.add_argument("--summarize-only", action="store_true", help="recompute the summary from the rows; no model")
    p.add_argument("--dataset", default="arc_challenge", choices=("arc_challenge", "arc_easy", "gsm8k"))
    p.add_argument("--readout", default="bf16", choices=("bf16", "fp32-letters"),
                   help="bf16: the model's own logits; fp32-letters: LetterHead, lm_head input x letter rows in fp32")
    args = p.parse_args()
    global PROMPT_STYLE, DATASET
    PROMPT_STYLE = args.prompt_style
    DATASET = args.dataset
    if args.summarize_only:
        out_dir = Path(args.out)
        rows = [json.loads(l) for l in open(out_dir / "lens_rows.jsonl") if l.strip()]
        summary = summarize(rows, args.K)
        prev = out_dir / "lens_summary.json"
        summary["conditions"] = json.load(open(prev)).get("conditions") if prev.exists() else {"note": "rows only"}
        write_json(prev, summary)
        c = summary["conditions"]
        print(table(summary, {"device": c.get("device", "?"), "dtype": c.get("dtype", "?"),
                              "seconds_per_trajectory": c.get("seconds_per_trajectory") or float("nan")}))
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "lens_rows.jsonl"
    if args.fresh and rows_path.exists():
        rows_path.unlink()
    existing = {}
    if rows_path.exists():
        for line in open(rows_path):
            r = json.loads(line)
            if r.get("schema") != SCHEMA:
                sys.exit(f"[stop] {rows_path} holds schema {r.get('schema')!r}; rerun with --fresh")
            existing[(r["qid"], r["condition"])] = r

    dtype = getattr(torch, args.dtype)
    t0 = time.time()
    tok, model = load_model(args.device, dtype)
    load_s = time.time() - t0
    # each option is read as logsumexp over its two single-token spellings, "A" and " A"
    nospace = [tok.encode(L, add_special_tokens=False) for L in LETTERS]
    space = [tok.encode(" " + L, add_special_tokens=False) for L in LETTERS]
    assert all(len(x) == 1 for x in nospace + space), f"letter tokens are not single tokens: {nospace} {space}"
    letter_ids = torch.tensor([[x[0] for x in nospace], [x[0] for x in space]], device=args.device)  # (2, 4)
    qs = load_questions(args.n)
    wanted = args.conditions.split(",")
    head = LetterHead(model, letter_ids) if args.readout == "fp32-letters" else None

    # tripwires on the first question
    ids0 = prompt_ids(tok, render(qs[0], qs[0]["choices"]["text"]), args.device)
    t1 = time.time()
    s_a, row_a, *_, sb_a = lens_trajectory(model, ids0, letter_ids, args.K, 0, head)
    first_traj_s = time.time() - t1
    s_b, row_b, *_ = lens_trajectory(model, ids0, letter_ids, args.K, 0, head)
    det = float((row_a - row_b).abs().max())
    fwd = model_forward_last(model, ids0, args.K, 0)
    consist = float((row_a - fwd).abs().max())
    trip = {"determinism_max_abs_diff": det, "lens_vs_forward_max_abs_diff": consist,
            "prompt_tokens": int(ids0.shape[1]), "first_trajectory_seconds": round(first_traj_s, 1)}
    if head is not None:
        # the fp32 letter logits must sit within bf16 rounding of the model's own
        trip["fp32_vs_bf16_letters_max_abs_diff"] = float(np.abs(np.array(s_a) - np.array(sb_a)).max())
    print("tripwires:", trip, flush=True)
    if det > 0 or consist > 1e-3 or trip.get("fp32_vs_bf16_letters_max_abs_diff", 0) > 0.25:
        sys.exit("[stop] tripwire failed")

    n_new, t2 = 0, time.time()
    with open(rows_path, "a") as f:
        for qi, q in enumerate(qs):
            for name, opts, correct, seed in conditions(q):
                if name not in wanted or (q["id"], name) in existing:
                    continue
                ids = prompt_ids(tok, render(q, opts), args.device)
                steps, _, raw, mass, aml, sb = lens_trajectory(model, ids, letter_ids, args.K, seed, head)
                r = {"schema": SCHEMA, "qid": q["id"], "qi": qi, "condition": name, "init_seed": seed,
                     "correct_letter": correct, "n_tokens": int(ids.shape[1]), "steps": steps,
                     "steps_raw_nospace_space": raw, "letter_mass": mass, "argmax_is_letter": aml,
                     "derived": derive(steps, LETTERS.index(correct))}
                if sb is not None:
                    r["readout"], r["steps_bf16"] = args.readout, sb
                f.write(json.dumps(r) + "\n")
                f.flush()
                existing[(q["id"], name)] = r
                n_new += 1
                if n_new % 10 == 0:
                    print(f"  {n_new} trajectories, {(time.time() - t2) / n_new:.1f} s each", flush=True)
    per_traj = (time.time() - t2) / max(n_new, 1)

    rows = list(existing.values())
    summary = summarize(rows, args.K)
    summary["conditions"] = {"model": MODEL_ID, "device": args.device, "dtype": args.dtype, "K": args.K,
                             "dataset": f"allenai/ai2_arc {args.dataset} test, four-option questions, seeded shuffle",
                             "question_seed": QUESTION_SEED, "prompt_style": args.prompt_style,
                             "prompt": "question + lettered options; see prompt_ids() for the three styles",
                             "readout": "predict_from_latents at each recurrent step, last position, logits of ' A'..' D'"
                                        + ("; option logits from LetterHead: lm_head input x letter rows in fp32" if head is not None else ""),
                             "readout_mode": args.readout,
                             "tripwires": trip, "model_load_seconds": round(load_s, 1),
                             "seconds_per_trajectory": round(per_traj, 1) if n_new else None,
                             "rows_new": n_new, "rows_total": len(rows), "torch": torch.__version__,
                             "date": time.strftime("%Y-%m-%d")}
    write_json(out_dir / "lens_summary.json", summary)
    print()
    print(table(summary, summary["conditions"] | {"seconds_per_trajectory": per_traj if n_new else float("nan")}))


if __name__ == "__main__":
    main()
