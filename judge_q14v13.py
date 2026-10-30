"""Pass/fail checks for a Qwen3-14B adapter (same checks P1-P5 as judge_v11.py) on replicates 0-9 and 80-99.
Controls: AGN_Q14, BASE_Q14, BASE_warn_Q14, BASE_warn_explore_Q14. Also reports the base model per cell on replicates 0-9 (the paper's
replicate count) against the published Qwen3-14B cells of Table 26.
Usage: python judge_q14v13.py <output dir> <adapter arm>"""
import json, os, sys, collections
D = sys.argv[1]
SRC = {"ADAPT": sys.argv[2], "AGN": "AGN_Q14", "BASE": "BASE_Q14", "BASE_warn": "BASE_warn_Q14", "BASE_warn_explore": "BASE_warn_explore_Q14"}
PUB = {("bandit", "ordinal", "helpful"): .07, ("bandit", "ordinal", "misleading"): .49, ("farm", "ordinal", "helpful"): .04,
       ("farm", "ordinal", "misleading"): .71, ("farm", "world", "helpful"): .10, ("farm", "world", "misleading"): .32,
       ("bandit", "sentiment", "helpful"): .13, ("bandit", "sentiment", "misleading"): .24, ("farm", "sentiment", "helpful"): .12,
       ("farm", "sentiment", "misleading"): .27, ("bandit", "alphanumeric", "none"): .17, ("farm", "alphanumeric", "none"): .15}
reps = {}
for arm, d in SRC.items():
    p = f"{D}/{d}/replicates.json"
    reps[arm] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def cm(rs, lo, hi):
    if not rs: return None
    by = collections.defaultdict(list)
    for r in rs:
        if lo <= r["rep"] < hi: by[(r["domain"], r["nom"], r["align"])].append(r["norm_cum_regret"])
    return {k: sum(v) / len(v) for k, v in by.items()} or None


def judge(lo, hi):
    d = {a: cm(reps[a], lo, hi) for a in SRC}
    if d["ADAPT"] is None: return {"reps": f"{lo}-{hi - 1}", "ALL_PASS": False, "missing": "ADAPT"}
    keys = sorted(d["ADAPT"]); SEM = [k for k in keys if k[1] != "alphanumeric"]
    MIS = [k for k in SEM if k[2] == "misleading"]; HEL = [k for k in SEM if k[2] == "helpful"]; ALP = [k for k in keys if k[1] == "alphanumeric"]
    mean = lambda x: None if x is None or any(k not in x for k in SEM) else sum(x[k] for k in SEM) / len(SEM)
    ad = mean(d["ADAPT"]); ctl = {a: mean(d[a]) for a in ("AGN", "BASE", "BASE_warn", "BASE_warn_explore")}; miss = [a for a, v in ctl.items() if v is None]
    out = {"reps": f"{lo}-{hi - 1}", "semantic_mean": {"ADAPT": ad, **ctl},
           "P1": {"pass": None if miss else bool(ad <= .15 and all(ad < v for v in ctl.values())), "missing": miss},
           "P2": {"pass": all(d["ADAPT"][k] <= .25 for k in MIS), "cells": {"|".join(k): round(d["ADAPT"][k], 3) for k in MIS}},
           "P3": {"pass": all(d["ADAPT"][k] <= .20 for k in HEL), "cells": {"|".join(k): round(d["ADAPT"][k], 3) for k in HEL}},
           "P4": {"pass": all(d["ADAPT"][k] <= .25 for k in ALP), "cells": {"|".join(k): round(d["ADAPT"][k], 3) for k in ALP}},
           "P5": {"pass": None if ctl["AGN"] is None else bool(ad <= ctl["AGN"] - .03), "AGN": ctl["AGN"]}}
    out["ALL_PASS"] = all(out[p]["pass"] is True for p in ("P1", "P2", "P3", "P4", "P5"))
    return out


res = {"reps_0_9": judge(0, 10), "reps_80_99": judge(80, 100)}
res["PASS"] = res["reps_0_9"]["ALL_PASS"] and res["reps_80_99"]["ALL_PASS"]
b = cm(reps["BASE"], 0, 10)
res["reproduction_BASE_reps_0_9_vs_published"] = None if b is None else {"|".join(k): [round(b.get(k, float("nan")), 3), v] for k, v in PUB.items()}
json.dump(res, open(f"{D}/judge_q14v13_{sys.argv[2]}.json", "w", encoding="utf-8"), indent=1)
for s in ("reps_0_9", "reps_80_99"):
    r = res[s]; print(s, "ALL_PASS", r["ALL_PASS"], "| semantic means", {k: (None if v is None else round(v, 3)) for k, v in r.get("semantic_mean", {}).items()})
    for p in ("P1", "P2", "P3", "P4", "P5"): print("  ", p, r.get(p, r.get("missing")))
if res["reproduction_BASE_reps_0_9_vs_published"]:
    print("BASE_Q14 reps 0-9 vs published [ours, paper]:", res["reproduction_BASE_reps_0_9_vs_published"])
    print("published semantic mean .249 | ours", None if b is None else round(sum(b[k] for k in PUB if k[1] != "alphanumeric" and k in b) / 10, 3))
print("PASS (both splits):", res["PASS"])
