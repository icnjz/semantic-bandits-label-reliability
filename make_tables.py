"""Builds the per-replicate and per-cell regret tables in results/ from the raw replicates.json of every evaluated arm.
Usage: python make_tables.py <output dir with one folder per arm> <results dir>
Arm folder names are the ones used by judge_v11.py / judge_q14v13.py (see README)."""
import csv, json, os, sys

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

CELLS = [("bandit", "ordinal", "helpful"), ("bandit", "ordinal", "misleading"), ("bandit", "sentiment", "helpful"),
         ("bandit", "sentiment", "misleading"), ("bandit", "alphanumeric", "none"), ("farm", "ordinal", "helpful"),
         ("farm", "ordinal", "misleading"), ("farm", "world", "helpful"), ("farm", "world", "misleading"),
         ("farm", "sentiment", "helpful"), ("farm", "sentiment", "misleading"), ("farm", "alphanumeric", "none")]
SEM = [c for c in CELLS if c[1] != "alphanumeric"]
PAPER = {  # Semantic Bandits, Table 26 (base models, 10 replicates)
    "Qwen3-8B": dict(zip(CELLS, [.01, .95, .33, .60, .40, .00, 1.00, .01, .78, .20, .70, .34])),
    "Qwen3-14B": dict(zip(CELLS, [.07, .49, .13, .24, .17, .04, .71, .10, .32, .12, .27, .15])),
}
COND0 = "3 arms, 10 turns, no reward noise"
# (model, column name, description, source directories, condition)
ARMS = []
for m, t in (("Qwen3-8B", "QWEN"), ("OLMo-3-7B", "OLMO")):
    b7 = {"QWEN": ["BASE_QWEN7"], "OLMO": ["BASE_OLMO", "BASE_OLMO_R8099"]}[t]
    w7 = {"QWEN": ["BASE_warn_QWEN7"], "OLMO": ["BASE_warn_OLMO", "BASE_warn_OLMO_R8099"]}[t]
    we7 = {"QWEN": ["BASE_warn_explore_QWEN7"], "OLMO": ["BASE_warn_explore_OLMO", "BASE_warn_explore_OLMO_R8099"]}[t]
    ARMS += [(m, "base", "Base model, paper's prompt", b7, COND0),
             (m, "warning", "Base model, paper's warning prompt", w7, COND0),
             (m, "warning_explore", "Base model, paper's warning + explore prompt", we7, COND0),
             (m, "label_ignoring", "Fine-tuned on a planner that ignores labels", [f"AGN_V7_{t}"], COND0),
             (m, "imitation_seed0", "Fine-tuned on the planner's decisions, plain imitation (v7 data), seed 0", [f"ADAPT_V7_{t}"], COND0),
             (m, "imitation_seed1", "Same, seed 1", [f"ADAPT_V7_{t}_S1"], COND0),
             (m, "method_seed0", "This method (v13 data: cost-sensitive copies + class balance), seed 0", [f"ADAPT_V13_{t}_S0"], COND0),
             (m, "method_seed1", "This method, seed 1", [f"ADAPT_V13_{t}_S1"], COND0),
             (m, "shuffled_copies_seed1", "Control: copy counts shuffled within label kind, class balance kept, seed 1", [f"ADAPT_V13SHUF_{t}_S1"], COND0)]
    ARMS += [(m, f"agentic_dpo_round{r}", f"Agentic-DPO, round {r}" + (" (SFT warm start)" if r == 0 else ""), [f"ADAPT_ADPO_R{r}_{t}_S1"], COND0)
             for r in range(6)]
ARMS += [("Qwen3-14B", "base", "Base model, paper's prompt", ["BASE_Q14"], COND0),
         ("Qwen3-14B", "warning", "Base model, paper's warning prompt", ["BASE_warn_Q14"], COND0),
         ("Qwen3-14B", "warning_explore", "Base model, paper's warning + explore prompt", ["BASE_warn_explore_Q14"], COND0),
         ("Qwen3-14B", "label_ignoring", "Fine-tuned on a planner that ignores labels", ["AGN_Q14"], COND0),
         ("Qwen3-14B", "method", "This method (v13 data)", ["ADAPT_V13_Q14"], COND0)]
NOISE = {"LP0": "low reward scale (L+), no noise", "HM0": "high scale, negative values (H-), no noise",
         "HP6": "high scale (H+), sigma 6.25", "HP12": "high scale (H+), sigma 12.5"}
for c, desc in NOISE.items():
    for a, col, d in (("BASE", "base", "Base model"), ("warn", "warning", "Base model, warning prompt"),
                      ("warn_explore", "warning_explore", "Base model, warning + explore prompt"),
                      ("AGN", "label_ignoring", "Fine-tuned on a planner that ignores labels"),
                      ("ADAPT", "imitation_seed0", "Fine-tuned on the planner's decisions, plain imitation (v7 data), seed 0")):
        ARMS.append(("Qwen3-8B", col, d, [f"SV_{c}_{a}"], "3 arms, 10 turns, " + desc))

SPLITS = {"dev (replicates 0-9)": range(0, 10), "fresh (replicates 80-99)": range(80, 100)}
rows_rep, per_cell = [], {}
for m, col, desc, dirs, cond in ARMS:
    seen = {}
    for d in dirs:
        for r in json.load(open(f"{SRC}/{d}/replicates.json", encoding="utf-8")):
            if r["rep"] < 10 or r["rep"] >= 80:
                seen[(r["domain"], r["nom"], r["align"], r["rep"])] = r["norm_cum_regret"]
    for (dom, nom, al, rep), v in sorted(seen.items(), key=lambda kv: (CELLS.index(kv[0][:3]), kv[0][3])):
        rows_rep.append([m, col, cond, dom, nom, al, rep, round(v, 6)])
    for s, rr in SPLITS.items():
        for c in CELLS:
            vals = [seen[c + (i,)] for i in rr if c + (i,) in seen]
            per_cell[(m, col, cond, s, c)] = (len(vals), sum(vals) / len(vals) if vals else None)
    for s, rr in SPLITS.items():
        n = [per_cell[(m, col, cond, s, c)][0] for c in CELLS]
        assert all(k == len(rr) for k in n), (m, col, cond, s, n)

with open(f"{OUT}/per_replicate_regret.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["model", "arm", "condition", "domain", "label_kind", "alignment", "replicate", "normalized_cumulative_regret"])
    w.writerows(rows_rep)
with open(f"{OUT}/per_cell_regret.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["model", "arm", "description", "condition", "replicates", "domain", "label_kind", "alignment", "n", "mean_normalized_regret", "paper_base_model"])
    for m, col, desc, dirs, cond in ARMS:
        for s in SPLITS:
            for c in CELLS:
                n, v = per_cell[(m, col, cond, s, c)]
                pub = PAPER.get(m, {}).get(c, "") if (col == "base" and cond == COND0) else ""
                w.writerow([m, col, desc, cond, s, *c, n, round(v, 4), pub])

def sem_mean(m, col, cond, s): return sum(per_cell[(m, col, cond, s, c)][1] for c in SEM) / len(SEM)
def fmt(v): return "" if v is None or v == "" else f"{v:.3f}" if isinstance(v, float) else str(v)

L = ["# Per-cell results", "",
     "Metric: normalized cumulative regret, sum over turns of (75 - mean of the chosen arm) / 500, so 0 is optimal and 1 is always the worst arm. "
     "Arm means are 75 / 50 / 25. Each cell is the mean over its replicates.",
     "", "Replicates 0-9 were used during development. Replicates 80-99 were never run until each model was trained. "
     "The paper column is the base model in Semantic Bandits, Table 26 (10 replicates).",
     "", "Semantic mean = mean over the 10 cells with meaningful labels (ordinal, world, sentiment; helpful and misleading). "
     "Alphanumeric labels carry no ordering information.", ""]
MAIN = {"Qwen3-8B": ["base", "warning", "warning_explore", "label_ignoring", "imitation_seed0", "method_seed0", "method_seed1", "shuffled_copies_seed1", "agentic_dpo_round2"],
        "OLMo-3-7B": ["base", "warning", "warning_explore", "label_ignoring", "imitation_seed0", "method_seed0", "method_seed1", "shuffled_copies_seed1", "agentic_dpo_round5"],
        "Qwen3-14B": ["base", "warning", "warning_explore", "label_ignoring", "method"]}
HEAD = {"base": "Base", "warning": "Warning", "warning_explore": "Warning + explore", "label_ignoring": "Label-ignoring planner",
        "imitation_seed0": "Planner imitation", "method_seed0": "This method s0", "method_seed1": "This method s1", "method": "This method",
        "shuffled_copies_seed1": "Shuffled copies s1", "agentic_dpo_round2": "Agentic-DPO (round 2)", "agentic_dpo_round5": "Agentic-DPO (round 5)"}
L += ["Columns:", "- Base, Warning, Warning + explore: the base model with the paper's prompts.",
      "- Label-ignoring planner: the same fine-tuning on decisions of a planner that ignores the labels.",
      "- Planner imitation: fine-tuning on the label-reliability planner's decisions, one copy each (v7 data).",
      "- This method: v13 data, with cost-sensitive copies and class balance. s0 / s1 are training seeds.",
      "- Shuffled copies: the v13 copy counts shuffled within each label kind, with class balance kept.",
      "- Agentic-DPO: the round selected on replicates 0-9 (Qwen3-8B round 2, OLMo-3-7B round 5). All rounds are in per_cell_regret.csv.", ""]
for m, cols in MAIN.items():
    for s in SPLITS:
        hdr = ["Cell"] + (["Paper"] if m in PAPER else []) + [HEAD[c] for c in cols]
        L += [f"## {m}, {s}", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for c in CELLS:
            L.append(f"| {' / '.join(c)} | " + (f"{PAPER[m][c]:.2f} | " if m in PAPER else "") + " | ".join(fmt(per_cell[(m, col, COND0, s, c)][1]) for col in cols) + " |")
        L.append("| **Semantic mean** | " + (f"{sum(PAPER[m][c] for c in SEM) / 10:.3f} | " if m in PAPER else "") + " | ".join(f"**{sem_mean(m, col, COND0, s):.3f}**" for col in cols) + " |")
        L.append("")
L += ["## Agentic-DPO by round (seed 1), semantic mean", "", "| Model | Replicates | " + " | ".join(f"Round {r}" for r in range(6)) + " |", "|---|---|" + "---|" * 6]
for m in ("Qwen3-8B", "OLMo-3-7B"):
    for s in SPLITS:
        L.append(f"| {m} | {s} | " + " | ".join(f"{sem_mean(m, f'agentic_dpo_round{r}', COND0, s):.3f}" for r in range(6)) + " |")
L += ["", "## Reward scale and noise (Qwen3-8B, planner imitation adapter, semantic mean)", "",
      "These runs use the plain-imitation adapter (v7 data, seed 0), not the final recipe.", "",
      "| Condition | Replicates | Base | Warning | Warning + explore | Label-ignoring planner | Planner imitation |", "|---|---|---|---|---|---|---|"]
for c, desc in NOISE.items():
    cond = "3 arms, 10 turns, " + desc
    for s in SPLITS:
        L.append(f"| {desc} | {s} | " + " | ".join(f"{sem_mean('Qwen3-8B', col, cond, s):.3f}" for col in ("base", "warning", "warning_explore", "label_ignoring", "imitation_seed0")) + " |")
L += ["", "## Files", "", "- `per_cell_regret.csv`: every arm, condition, replicate split and cell.",
      "- `per_replicate_regret.csv`: the regret of every single episode, from which all tables here are computed.",
      "- `judge_*.json`: the pass/fail checks of each trained adapter (see the main README).", ""]
open(f"{OUT}/PER_CELL_RESULTS.md", "w", encoding="utf-8", newline="\n").write("\n".join(L))
print(len(rows_rep), "episodes,", len(ARMS), "arm / condition pairs")
for m in MAIN:
    for s in SPLITS:
        print(m, s, {col: round(sem_mean(m, col, COND0, s), 3) for col in MAIN[m]})
for c in NOISE:
    print(c, {s[:5]: round(sem_mean("Qwen3-8B", "imitation_seed0", "3 arms, 10 turns, " + NOISE[c], s) - sem_mean("Qwen3-8B", "label_ignoring", "3 arms, 10 turns, " + NOISE[c], s), 3) for s in SPLITS})
b = {c: per_cell[("Qwen3-8B", "base", COND0, "dev (replicates 0-9)", c)][1] for c in CELLS}
print("8B base dev vs paper:", {"/".join(c): (round(b[c], 3), PAPER["Qwen3-8B"][c]) for c in CELLS if abs(b[c] - PAPER["Qwen3-8B"][c]) > .15})
