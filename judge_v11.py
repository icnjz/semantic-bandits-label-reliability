"""Pass/fail checks for one trained adapter, on the development replicates (0-9) and on the fresh replicates (80-99),
against the base model with the paper's three prompts and the label-ignoring control.
P1: semantic mean <= 0.15 and below every control. P2: every misleading cell <= 0.25. P3: every helpful cell <= 0.20.
P4: both alphanumeric cells <= 0.25. P5: semantic mean at least 0.03 below the label-ignoring control.
Usage: python judge_v11.py <output dir> qwen|olmo <adapter arm>   (control arm names: see README)"""
import collections, json, os, sys
D, model = sys.argv[1], sys.argv[2]
SRC = ({"ADAPT": [sys.argv[3]], "AGN": ["AGN_V7_QWEN"], "BASE": ["BASE_QWEN7"], "BASE_warn": ["BASE_warn_QWEN7"],
        "BASE_warn_explore": ["BASE_warn_explore_QWEN7"]} if model == "qwen" else
       {"ADAPT": [sys.argv[3]], "AGN": ["AGN_V7_OLMO"], "BASE": ["BASE_OLMO", "BASE_OLMO_R8099"],
        "BASE_warn": ["BASE_warn_OLMO", "BASE_warn_OLMO_R8099"], "BASE_warn_explore": ["BASE_warn_explore_OLMO", "BASE_warn_explore_OLMO_R8099"]})
reps = {}
for arm, dirs in SRC.items():
    rs = []
    for d in dirs:
        p = f"{D}/{d}/replicates.json"
        if os.path.exists(p): rs += json.load(open(p, encoding="utf-8"))
    reps[arm] = rs or None


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


res = {"dev_reps_0_9": judge(0, 10), "fresh_reps_80_99": judge(80, 100)}
res["PASS"] = res["dev_reps_0_9"]["ALL_PASS"] and res["fresh_reps_80_99"]["ALL_PASS"]
json.dump(res, open(f"{D}/judge_v11_{sys.argv[3]}.json", "w", encoding="utf-8"), indent=1)
for s in ("dev_reps_0_9", "fresh_reps_80_99"):
    r = res[s]; print(s, "ALL_PASS", r["ALL_PASS"], "| semantic means", {k: (None if v is None else round(v, 3)) for k, v in r.get("semantic_mean", {}).items()})
    for p in ("P1", "P2", "P3", "P4", "P5"): print("  ", p, r.get(p, r.get("missing")))
print("PASS (dev AND fresh):", res["PASS"])
