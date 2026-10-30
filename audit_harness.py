"""Usage: python audit_harness.py ARM [ARM ...]"""
import collections, json, re, sys

MEANS = {"H": 75.0, "M": 50.0, "L": 25.0}
PUB8 = {("bandit", "ordinal", "helpful"): .01, ("bandit", "ordinal", "misleading"): .95, ("bandit", "sentiment", "helpful"): .33,
        ("bandit", "sentiment", "misleading"): .60, ("bandit", "alphanumeric", "none"): .40, ("farm", "ordinal", "helpful"): .00,
        ("farm", "ordinal", "misleading"): 1.00, ("farm", "world", "helpful"): .01, ("farm", "world", "misleading"): .78,
        ("farm", "sentiment", "helpful"): .20, ("farm", "sentiment", "misleading"): .70, ("farm", "alphanumeric", "none"): .34}
LINE = re.compile(r"^ - (?:Arm|Field) Name: (.+?) -- This action has been selected (\d+) time\(s\) for an observed mean of ([\d.]+)$", re.M)

def audit(arm):
    reps = {(r["domain"], r["nom"], r["align"], r["rep"]): r for r in json.load(open(f"out/{arm}/replicates.json", encoding="utf-8"))}
    dec = collections.defaultdict(list)
    for l in open(f"out/{arm}/decisions.jsonl", encoding="utf-8"):
        d = json.loads(l)
        if d.get("final", True) and d.get("arm") is not None: dec[(d["domain"], d["nom"], d["align"], d["rep"])].append(d)
    n_reg_bad = n_prompt_bad = n_turn_bad = n_mis_bad = 0; labels_seen = set()
    for key, ds in dec.items():
        ds.sort(key=lambda d: d["turn"])
        reg = sum(MEANS["H"] - MEANS[d["arm"]] for d in ds) / (len(ds) * (MEANS["H"] - MEANS["L"]))
        if abs(reg - reps[key]["norm_cum_regret"]) > 1e-9 or len(ds) != 10: n_reg_bad += 1
        lab = ds[0]["labels"]; labels_seen |= set(lab.values())
        if key[2] == "misleading" and key[1] in ("ordinal", "world"):
            best_sounding = {"ordinal": "Highest Reward", "world": "Productive Grassland"}[key[1]]
            if lab["L"] != best_sounding: n_mis_bad += 1
        counts = {a: 0 for a in "HML"}; sums = {a: 0.0 for a in "HML"}
        for t, d in enumerate(ds):
            if d["turn"] != t or f"We are on step {t}." not in d["prompt"]: n_turn_bad += 1
            shown = {name: (int(c), float(m)) for name, c, m in LINE.findall(d["prompt"])}
            for a in "HML":
                exp = (counts[a], (sums[a] / counts[a]) if counts[a] else 0.0)
                if shown.get(lab[a]) != exp: n_prompt_bad += 1; break
            counts[d["arm"]] += 1; sums[d["arm"]] += MEANS[d["arm"]]
    print(f"{arm:26s} replicates {len(dec)}: regret mismatches {n_reg_bad}, prompt/history mismatches {n_prompt_bad}, "
          f"turn-number mismatches {n_turn_bad}, misleading-label errors {n_mis_bad}")
    return labels_seen

if __name__ == "__main__":
    all_labels = set()
    for arm in sys.argv[1:]: all_labels |= audit(arm)
    train = open("v13/train_adaptive_v13.jsonl", encoding="utf-8").read() + open("v7/train_adaptive_v7.jsonl", encoding="utf-8").read()
    hits = sorted(l for l in all_labels if re.search(r"(?<![A-Za-z0-9])" + re.escape(l) + r"(?![A-Za-z0-9])", train))
    print(f"evaluation label strings: {len(all_labels)}; found in training data: {len(hits)} {hits[:10]}")
    base = json.load(open("out/BASE_QWEN7/replicates.json", encoding="utf-8"))
    by = collections.defaultdict(list)
    for r in base:
        if r["rep"] < 10: by[(r["domain"], r["nom"], r["align"])].append(r["norm_cum_regret"])
    print("Qwen3-8B base, reps 0-9 (paper's replicate count): cell  ours  published")
    for k, p in PUB8.items(): print(f"  {k[0]:6s} {k[1]:12s} {k[2]:10s} {sum(by[k]) / len(by[k]):.2f}  {p:.2f}")
