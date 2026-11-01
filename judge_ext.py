"""Judgement of the five-arm decision lines (fixed before the run). Usage: python judge_ext.py OUTDIR
Metric: mean normalised regret over the non-alphanumeric cells (equal replicates per cell). Paired comparisons resample replicate
indices within each cell (10,000 draws, seed 0); probe comparisons resample the 22 disagreement states (cluster bootstrap)."""
import json, os, sys
import numpy as np

D = sys.argv[1]
DEV, FRESH60, FRESH20 = list(range(0, 10)), list(range(80, 140)), list(range(80, 100))
rng = np.random.default_rng(0)


def reps(arm):
    p = os.path.join(D, arm, "replicates.json")
    if not os.path.exists(p):
        return None
    out = {}
    for r in json.load(open(p)):
        if r["nom"] != "alphanumeric":
            out.setdefault((r["domain"], r["nom"], r["align"]), {})[r["rep"]] = r["norm_cum_regret"]
    return out


def sem(R, rs, align=None):
    cells = [c for c in R if align is None or c[2] == align]
    return float(np.mean([np.mean([R[c][r] for r in rs]) for c in cells]))


def paired(RA, RB, rs, align=None, n=10000):
    cells = [c for c in RA if align is None or c[2] == align]
    d = np.array([[RA[c][r] - RB[c][r] for r in rs] for c in cells])          # cells x reps
    point = float(d.mean())
    idx = rng.integers(0, d.shape[1], size=(n, d.shape[0], d.shape[1]))
    boots = np.take_along_axis(np.broadcast_to(d, (n,) + d.shape), idx, axis=2).mean(axis=(1, 2))
    return point, [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def probe(arm):
    p = os.path.join(D, arm, "probes.json")
    return json.load(open(p))["decisions"] if os.path.exists(p) else None


def dp_rate_diff(PA, PB, n=10000):
    sa = {}; sb = {}
    for x in PA: sa.setdefault(x["state"], []).append(x["match"] == "dp")
    for x in PB: sb.setdefault(x["state"], []).append(x["match"] == "dp")
    states = sorted(set(sa) & set(sb))
    a = np.array([np.mean(sa[s]) for s in states]); b = np.array([np.mean(sb[s]) for s in states])
    point = float(a.mean() - b.mean())
    idx = rng.integers(0, len(states), size=(n, len(states)))
    boots = (a[idx] - b[idx]).mean(axis=1)
    return float(a.mean()), float(b.mean()), point, [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], float((boots <= 0).mean())


res = {"lines": {}, "means": {}}
A = {n: reps(n) for n in os.listdir(D) if os.path.isdir(os.path.join(D, n))}
A = {k: v for k, v in A.items() if v}
for n, R in A.items():
    some = next(iter(R.values()))
    res["means"][n] = {s: round(sem(R, rs), 4) for s, rs in (("dev", DEV), ("fresh20", FRESH20), ("fresh60", FRESH60)) if all(r in some for r in rs)}
for model in ("QWEN", "OLMO"):
    for cond in ("E5", "EC"):
        for seed in ("S0", "S1"):
            v = A.get(f"{cond}_V13_{model}_{seed}")
            if not v: continue
            ctrls = {c: A.get(f"{cond}_{c}_{model}") for c in ("AGN", "BASE", "WX")}
            if not all(ctrls.values()): continue
            x1 = all(sem(v, rs) < sem(ctrls[c], rs) for c in ctrls for rs in (DEV, FRESH20))
            x2 = all(sem(v, rs) <= sem(ctrls["AGN"], rs) - .03 for rs in (DEV, FRESH20))
            res["lines"][f"X1_{cond}_{model}_{seed}"] = x1; res["lines"][f"X2_{cond}_{model}_{seed}"] = x2
v0, s0, agn, v1, ad = (A.get(x) for x in ("E5_V13_QWEN_S0", "E5_SCRIPT_QWEN_S0", "E5_AGN_QWEN", "E5_V13_QWEN_S1", "E5_ADPO_QWEN_S1"))
if v0 and s0:
    d, ci = paired(v0, s0, FRESH60)
    m = sem(v0, FRESH60)
    res["lines"]["L1_teacher_beyond_script"] = {"pass": bool(d <= -.025 and ci[1] < 0 and m < .100), "v13_minus_script": round(d, 4), "ci95": [round(x, 4) for x in ci], "v13_fresh60": round(m, 4), "script_fresh60": round(sem(s0, FRESH60), 4)}
if v0 and agn:
    d, ci = paired(v0, agn, FRESH60); dm, cim = paired(v0, agn, FRESH60, "misleading")
    res["lines"]["L2_semantic_information"] = {"pass": bool(ci[1] < 0 and cim[1] < 0), "v13_minus_agn": round(d, 4), "ci95": [round(x, 4) for x in ci], "misleading_diff": round(dm, 4), "misleading_ci95": [round(x, 4) for x in cim]}
pv0, ps0, pad, pv1 = (probe(x) for x in ("P5_V13_QWEN_S0", "P5_SCRIPT_QWEN_S0", "P5_ADPO_QWEN_S1", "P5_V13_QWEN_S1"))
if pv0 and ps0:
    a, b, d, ci, p = dp_rate_diff(pv0, ps0)
    res["lines"]["L3_policy_identification"] = {"pass": bool(d >= .10 and ci[0] > 0), "v13_dp_rate": round(a, 4), "script_dp_rate": round(b, 4), "diff": round(d, 4), "ci95": [round(x, 4) for x in ci], "p_boot": round(p, 4)}
if v1 and ad:
    d, ci = paired(v1, ad, FRESH60)
    fid = dp_rate_diff(pv1, pad) if (pv1 and pad) else None
    adv = (d <= -.01 and ci[1] < 0) or (fid is not None and fid[2] >= .10 and fid[3][0] > 0)
    worse = (d >= .01 and ci[0] > 0) or (fid is not None and fid[2] <= -.10 and fid[3][1] < 0)
    res["lines"]["L4_v13_vs_agentic_dpo"] = {"verdict": "V13_OBJECTIVE_ADVANTAGE" if adv and not worse else ("AGENTIC_DPO_WINS" if worse and not adv else "MATCH"),
                                            "v13_minus_adpo": round(d, 4), "ci95": [round(x, 4) for x in ci], "fidelity": None if fid is None else {"v13": round(fid[0], 4), "adpo": round(fid[1], 4), "diff": round(fid[2], 4), "ci95": [round(x, 4) for x in fid[3]]}}
if v0 and s0 and "L3_policy_identification" in res["lines"]:
    L1, L3 = res["lines"]["L1_teacher_beyond_script"], res["lines"]["L3_policy_identification"]
    res["lines"]["L5_failure_criterion"] = bool(L1["v13_fresh60"] >= .100 or not L1["pass"] or not L3["pass"])
json.dump(res, open(os.path.join(D, "judge_ext.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
