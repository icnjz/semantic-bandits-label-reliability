"""sbadapt_eval_mf.py plus the paper's other reward conditions (Section 4.5): --scale H+ / H- / L+ / L- (arm means
(75, 50, 25) or (-25, -50, -75), divided by 100 for Low) and --sigma 0 / 6.25 / 12.5 (Gaussian reward noise). Noise uses common
random numbers: the j-th pull of arm X in replicate r gets the same draw for every model and adapter. Regret is pseudo-regret on
the arm means. Everything else as in sbandit_harness.py."""
import argparse
import hashlib
import json
import os
import random
import re
import string
import sys
import time

import numpy as np

NUM_TURNS = 10
THINKING_BUDGET = 150
NEW_TOKEN_BUDGET = 512
MEANS = {"H": 75.0, "M": 50.0, "L": 25.0}
SCALES = {"H+": {"H": 75.0, "M": 50.0, "L": 25.0}, "H-": {"H": -25.0, "M": -50.0, "L": -75.0},
          "L+": {"H": 0.75, "M": 0.50, "L": 0.25}, "L-": {"H": -0.25, "M": -0.50, "L": -0.75}}
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20)
EARLY_STOP = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking "
              "directly now.\n</think>.\n\n")

TASK_PROMPTS = {
    "bandit": ("You are faced with a 3-armed bandit problem. Each action corresponds to pulling a specific arm. "
               "Each arm is associated with a Gaussian distribution with a fixed but unknown mean. At each time step, "
               "you will select an arm and receive a reward drawn from the associated distribution. Your goal is to "
               "select which arm to pull in order to maximize your reward across all of your turns."),
    "farm": ("You are a farmer who owns 3 fields. Each action corresponds to planting your crop in a different field. "
             "Each field is associated with a Gaussian distribution with a fixed but unknown mean. At each time step, "
             "you will select a field and receive a reward drawn from the associated distribution. Your goal is to "
             "maximize your reward across all of your turns."),
}
ACTION_SUFFIX = {"bandit": "Arm", "farm": "Field"}

ORDINAL = {"H": "Highest Reward", "M": "Intermediate Reward", "L": "Lowest Reward"}
WORLD_FARM = {"H": "Productive Grassland", "M": "Rocky Hills", "L": "Arid Desert"}
SENT_POOLS = {
    "H": "excellent fabulous outstanding phenomenal superb magnificent marvelous terrific wonderful exceptional "
         "spectacular brilliant superlative splendid sublime stellar extraordinary remarkable exquisite impeccable".split(),
    "M": "standard typical functional ordinary average regular common usual conventional moderate normal plain "
         "routine basic customary everyday commonplace mainstream generic medium".split(),
    "L": "awful terrible deplorable repulsive horrible atrocious abysmal dreadful appalling disgusting horrendous "
         "pathetic abominable wretched vile despicable hideous lousy dismal ghastly".split(),
}
for _k, _v in SENT_POOLS.items():
    assert len(_v) == 20 and len(set(_v)) == 20, _k

# The 12 Qwen3-8B cells of Table 26 (domain, nomenclature, alignment) -> (published regret, published exploration)
PUBLISHED = {
    ("bandit", "ordinal", "helpful"): (0.01, 1.10), ("bandit", "ordinal", "misleading"): (0.95, 1.10),
    ("bandit", "sentiment", "helpful"): (0.33, 1.10), ("bandit", "sentiment", "misleading"): (0.60, 1.00),
    ("bandit", "alphanumeric", "none"): (0.40, 1.60),
    ("farm", "ordinal", "helpful"): (0.00, 1.00), ("farm", "ordinal", "misleading"): (1.00, 1.00),
    ("farm", "world", "helpful"): (0.01, 1.10), ("farm", "world", "misleading"): (0.78, 1.50),
    ("farm", "sentiment", "helpful"): (0.20, 1.00), ("farm", "sentiment", "misleading"): (0.70, 1.00),
    ("farm", "alphanumeric", "none"): (0.34, 1.60),
}
DOMAINS = ["bandit", "farm"]
NOMS = ["ordinal", "world", "sentiment", "alphanumeric"]

TEMPLATE_HEAD = ("You will be assigned a task.\n"
                 "You will receive the actions you have available, followed by the action history.\n"
                 "Your goal is to maximize the total reward across all {num_turns} steps.\n"
                 "We are on step {current_turn}.\n\n"
                 "Task Assignment:\n---\n{task_prompt}\n---\n\n"
                 "Available Actions:\n---\n")
TEMPLATE_LINE = " - {action_suffix} Name: {action} -- This action has been selected {count} time(s) for an observed mean of {obs_mean}\n"
TEMPLATE_TAIL = ("---\n\n"
                 "When selecting an action, include only the name of the action.\n"
                 "Do not include the dashes used to indicate the list here.\n\n"
                 "Please select an action from the list of action options in order to maximize\n"
                 "the reward for your task. When selecting an action, put the exact name of the\n"
                 "action between action tags like:\n"
                 "<action> action_name </action>.\n\n"
                 "Your thinking budget is {thinking_budget} tokens. If you exceed this\n"
                 "limit, you will be forced to provide an immediate answer.\n"
                 "Your overall generation limit is {new_token_budget} tokens.")


def seed_of(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def make_replicate(domain, nom, align, rep):
    """Labels per physical arm (H/M/L) and the display order. Helpful/misleading share the RNG draw."""
    rng = np.random.default_rng(20261026 + 1000 * DOMAINS.index(domain) + 100 * NOMS.index(nom) + rep)
    order = [str(x) for x in rng.permutation(["H", "M", "L"])]
    if nom == "ordinal":
        pool = dict(ORDINAL)
    elif nom == "world":
        pool = dict(WORLD_FARM)
    elif nom == "sentiment":
        pool = {k: SENT_POOLS[k][int(rng.integers(20))].capitalize() for k in ("H", "M", "L")}
    elif nom == "alphanumeric":
        alphabet = list(string.ascii_lowercase + string.digits)
        names = set()
        while len(names) < 3:
            names.add("".join(rng.choice(alphabet, 6)))
        pool = dict(zip(("H", "M", "L"), sorted(names, key=lambda s: rng.random())))
    else:
        raise ValueError(nom)
    if align == "misleading":
        labels = {"H": pool["L"], "M": pool["M"], "L": pool["H"]}
    else:
        labels = pool
    return {"order": order, "labels": labels}


WARN = "Be careful about bias in the action labels. Action names may be misleading and are not guaranteed to correlate with reward."
VARIANT = {"none": "", "warn": "\n" + WARN, "warn_explore": "\n" + WARN + " Make sure to explore."}
PROMPT_VARIANT = "none"


def render_prompt(domain, rep_cfg, counts, sums, turn):
    s = TEMPLATE_HEAD.format(num_turns=NUM_TURNS, current_turn=turn, task_prompt=TASK_PROMPTS[domain] + VARIANT[PROMPT_VARIANT])
    for arm in rep_cfg["order"]:
        n = counts[arm]
        mean = (sums[arm] / n) if n > 0 else 0.0
        s += TEMPLATE_LINE.format(action_suffix=ACTION_SUFFIX[domain], action=rep_cfg["labels"][arm], count=n,
                                  obs_mean=str(float(mean)))
    s += TEMPLATE_TAIL.format(thinking_budget=THINKING_BUDGET, new_token_budget=NEW_TOKEN_BUDGET)
    return s


def _norm(x):
    x = x.strip().strip("`*\"' \t\n").rstrip(".").strip("`*\"' \t\n")
    return re.sub(r"\s+", " ", x).lower()


def parse_action(text, labels, suffix):
    """labels: {arm: label}. Returns (arm or None, how)."""
    lab = {arm: _norm(l) for arm, l in labels.items()}
    answer = text.split("</think>")[-1]
    tags = re.findall(r"<action>(.*?)</action>", answer, flags=re.S | re.I)
    where = "answer_tag"
    if not tags:
        tags = re.findall(r"<action>(.*?)</action>", text, flags=re.S | re.I)
        where = "any_tag"
    if tags:
        x = _norm(tags[-1])
        x = re.sub(r"^(%s\s+)?name\s*:\s*" % re.escape(suffix.lower()), "", x).strip()
        x = _norm(x)
        for arm, l in lab.items():
            if x == l:
                return arm, where + "_exact"
        hits = [arm for arm, l in lab.items() if l in x]
        if len(hits) == 1:
            return hits[0], where + "_substr"
        return None, where + "_nomatch"
    low = answer.lower()
    hits = [arm for arm, l in lab.items() if l in low]
    if len(hits) == 1:
        return hits[0], "untagged_unique"
    return None, "no_tag"


class VLLMBackend:
    def __init__(self, model, max_model_len, lora=None):
        from vllm import LLM
        self.lora = None
        kw = dict(enable_lora=True, max_lora_rank=16, max_loras=1) if lora else {}
        self.llm = LLM(model=model, dtype="bfloat16", max_model_len=max_model_len, enable_prefix_caching=True,
                       gpu_memory_utilization=0.90, seed=0, **kw)
        if lora:
            from vllm.lora.request import LoRARequest
            self.lora = LoRARequest("adapter", 1, lora)
        self.tok = self.llm.get_tokenizer()
        self.stop = stop_ids_for(self.tok)

    def generate(self, prompts_ids, max_tokens, seeds):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        sps = [SamplingParams(n=1, max_tokens=int(m), seed=int(s), stop_token_ids=list(self.stop),
                              skip_special_tokens=True, **SAMPLING) for m, s in zip(max_tokens, seeds)]
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=list(p)) for p in prompts_ids], sps, use_tqdm=False, lora_request=self.lora)
        res = []
        for o in outs:
            c = o.outputs[0]
            ids = list(c.token_ids)
            eos = bool(ids) and ids[-1] in self.stop or c.finish_reason == "stop"
            res.append({"ids": ids, "text": c.text, "eos": bool(eos)})
        return res


class HFBackend:
    def __init__(self, model, dtype="bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype))
        self.model.eval()

    def generate(self, prompts_ids, max_tokens, seeds):
        torch = self.torch
        res = []
        for p, m, s in zip(prompts_ids, max_tokens, seeds):
            torch.manual_seed(int(s))
            inp = torch.tensor([list(p)])
            with torch.no_grad():
                out = self.model.generate(inp, attention_mask=torch.ones_like(inp), do_sample=True,
                                          max_new_tokens=int(m), eos_token_id=[151645, 151643], pad_token_id=151643,
                                          **SAMPLING)
            ids = out[0, inp.shape[1]:].tolist()
            eos = bool(ids) and ids[-1] in (151645, 151643)
            res.append({"ids": ids, "text": self.tok.decode(ids, skip_special_tokens=True), "eos": eos})
        return res


def stop_ids_for(tok):
    """Qwen: exactly the original [151645, 151643]. Other ChatML families: <|im_end|> and the tokenizer's eos."""
    if tok.convert_tokens_to_ids("<|im_end|>") == 151645: return (151645, 151643)
    ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    return tuple(dict.fromkeys(i for i in ids if isinstance(i, int) and i >= 0))


def chat_ids(tok, prompt):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                   enable_thinking=True, tokenize=False)
    return tok.encode(text, add_special_tokens=False)


def run_decisions(backend, items, attempt):
    """items: list of dicts with 'prompt_ids', 'key'. Two-phase thinking-budget generation."""
    tok = backend.tok
    think_end = tok.convert_tokens_to_ids("</think>")
    single = isinstance(think_end, int) and think_end != tok.unk_token_id and len(tok.encode("</think>", add_special_tokens=False)) == 1
    stop_ids = tok.encode(EARLY_STOP, add_special_tokens=False)
    p1 = backend.generate([it["prompt_ids"] for it in items], [THINKING_BUDGET] * len(items),
                          [seed_of(*it["key"], attempt, 1) for it in items])
    need2, cont = [], []
    for it, r in zip(items, p1):
        it["p1_ids"], it["p1_text"], it["p1_eos"] = r["ids"], r["text"], r["eos"]
        it["closed_think_p1"] = (think_end in r["ids"]) if single else ("</think>" in r["text"])
        it["forced"] = False
        it["p2_text"], it["p2_ids"], it["p2_eos"] = "", [], None
        if r["eos"]:
            continue
        if it["closed_think_p1"]:
            cont.append(list(it["prompt_ids"]) + r["ids"])
        else:
            it["forced"] = True
            cont.append(list(it["prompt_ids"]) + r["ids"] + stop_ids)
        need2.append(it)
    if need2:
        p2 = backend.generate(cont, [NEW_TOKEN_BUDGET - len(it["p1_ids"]) for it in need2],
                              [seed_of(*it["key"], attempt, 2) for it in need2])
        for it, r in zip(need2, p2):
            it["p2_ids"], it["p2_text"], it["p2_eos"] = r["ids"], r["text"], r["eos"]
    for it in items:
        mid = EARLY_STOP if it["forced"] else ""
        it["full_text"] = it["p1_text"] + mid + it["p2_text"]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "hf"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--rep_list", default=None, help="e.g. 0-9,80-99; overrides --reps (replicates are deterministic in their index)")
    ap.add_argument("--turns", type=int, default=NUM_TURNS)
    ap.add_argument("--cells", default="all", help="all or comma list domain:nom:align")
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--hf_dtype", default="bfloat16")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--variant", default="none", choices=["none", "warn", "warn_explore"])
    ap.add_argument("--scale", default="H+", choices=["H+", "H-", "L+", "L-"])
    ap.add_argument("--sigma", type=float, default=0.0)
    a = ap.parse_args()
    global PROMPT_VARIANT
    PROMPT_VARIANT = a.variant
    MEANS.update(SCALES[a.scale])
    print("[conditions] scale", a.scale, "means", MEANS, "sigma", a.sigma, flush=True)
    os.makedirs(a.out, exist_ok=True)
    cells = list(PUBLISHED.keys()) if a.cells == "all" else [tuple(c.split(":")) for c in a.cells.split(",")]
    t0 = time.time()
    backend = VLLMBackend(a.model, a.max_model_len, a.lora) if a.backend == "vllm" else HFBackend(a.model, a.hf_dtype)
    print(f"[load] {time.time() - t0:.1f}s", flush=True)

    reps = []
    for (d, n, al) in cells:
        for r in (sorted({x for part in a.rep_list.split(",") for x in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)}) if a.rep_list else range(a.reps)):
            cfg = make_replicate(d, n, al, r)
            reps.append({"domain": d, "nom": n, "align": al, "rep": r, "cfg": cfg,
                         "counts": {k: 0 for k in "HML"}, "sums": {k: 0.0 for k in "HML"}, "actions": [],
                         "fail_rng": random.Random(seed_of(d, n, al, r, "failrng"))})
    dec_f = open(os.path.join(a.out, "decisions.jsonl"), "w", encoding="utf-8")
    for turn in range(a.turns):
        pending = []
        for R in reps:
            prompt = render_prompt(R["domain"], R["cfg"], R["counts"], R["sums"], turn)
            pending.append({"R": R, "prompt": prompt, "prompt_ids": chat_ids(backend.tok, prompt),
                            "key": (R["domain"], R["nom"], R["align"], R["rep"], turn)})
        tt = time.time()
        attempt = 0
        todo = pending
        while todo and attempt < 3:
            run_decisions(backend, todo, attempt)
            nxt = []
            for it in todo:
                R = it["R"]
                arm, how = parse_action(it["full_text"], R["cfg"]["labels"], ACTION_SUFFIX[R["domain"]])
                rec = {"domain": R["domain"], "nom": R["nom"], "align": R["align"], "rep": R["rep"], "turn": turn,
                       "attempt": attempt, "order": R["cfg"]["order"], "labels": R["cfg"]["labels"],
                       "prompt": it["prompt"], "n_prompt_tokens": len(it["prompt_ids"]),
                       "p1_tokens": len(it["p1_ids"]), "p1_eos": it["p1_eos"], "closed_think_p1": it["closed_think_p1"],
                       "forced": it["forced"], "p2_tokens": len(it["p2_ids"]), "p2_eos": it["p2_eos"],
                       "raw": it["full_text"], "parse": how, "arm": arm}
                if arm is None and attempt == 2:
                    arm = R["fail_rng"].choice(["H", "M", "L"])
                    rec["arm"], rec["parse"] = arm, how + "|parse_fail_random"
                rec["final"] = arm is not None
                if arm is not None:
                    j = R["counts"][arm]
                    rew = MEANS[arm] + (float(np.random.default_rng(seed_of(R["domain"], R["nom"], R["align"], R["rep"], arm, j, "noise")).normal(0.0, a.sigma)) if a.sigma > 0 else 0.0)
                    rec["reward"] = rew
                    rec["inst_regret"] = MEANS["H"] - MEANS[arm]
                    R["counts"][arm] += 1
                    R["sums"][arm] += rew
                    R["actions"].append({"turn": turn, "arm": arm, "label": R["cfg"]["labels"][arm],
                                         "reward": rew, "parse": rec["parse"], "attempt": attempt})
                else:
                    nxt.append(it)
                dec_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            dec_f.flush()
            todo = nxt
            attempt += 1
        print(f"[turn {turn}] {len(pending)} decisions in {time.time() - tt:.1f}s", flush=True)
    dec_f.close()

    out_reps, cell_vals = [], {}
    for R in reps:
        arms = [x["arm"] for x in R["actions"]]
        T = len(arms)
        reg = sum(MEANS["H"] - MEANS[x] for x in arms) / (T * (MEANS["H"] - MEANS["L"]))
        out_reps.append({"domain": R["domain"], "nom": R["nom"], "align": R["align"], "rep": R["rep"],
                         "order": R["cfg"]["order"], "labels": R["cfg"]["labels"], "arms": arms,
                         "norm_cum_regret": reg, "explore_count": len(set(arms)),
                         "n_parse_fail_random": sum("parse_fail_random" in x["parse"] for x in R["actions"]),
                         "n_retried": sum(x["attempt"] > 0 for x in R["actions"])})
        cell_vals.setdefault((R["domain"], R["nom"], R["align"]), []).append(out_reps[-1])
    json.dump(out_reps, open(os.path.join(a.out, "replicates.json"), "w"), indent=1)
    rng = np.random.default_rng(0)
    cells_out = []
    for key, lst in cell_vals.items():
        regs = np.array([x["norm_cum_regret"] for x in lst])
        boots = [rng.choice(regs, len(regs)).mean() for _ in range(5000)]
        pub = PUBLISHED.get(key, (None, None))
        cells_out.append({"domain": key[0], "nom": key[1], "align": key[2], "n_reps": len(lst),
                          "regret": float(regs.mean()), "regret_ci95": [float(np.percentile(boots, 2.5)),
                                                                        float(np.percentile(boots, 97.5))],
                          "explore": float(np.mean([x["explore_count"] for x in lst])),
                          "published_regret": pub[0], "published_explore": pub[1],
                          "abs_diff": None if pub[0] is None else float(abs(regs.mean() - pub[0])),
                          "parse_fail_random": int(sum(x["n_parse_fail_random"] for x in lst))})
    json.dump(cells_out, open(os.path.join(a.out, "cells.json"), "w"), indent=1)
    json.dump({"args": vars(a), "wall_s": time.time() - t0, "sampling": SAMPLING, "thinking_budget": THINKING_BUDGET,
               "new_token_budget": NEW_TOKEN_BUDGET}, open(os.path.join(a.out, "run_meta.json"), "w"), indent=1)
    for c in cells_out:
        print(c["domain"], c["nom"], c["align"], round(c["regret"], 3), c["published_regret"], round(c["explore"], 2))
    open(os.path.join(a.out, "DONE"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


if __name__ == "__main__":
    main()
