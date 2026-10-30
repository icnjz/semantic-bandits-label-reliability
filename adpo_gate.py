"""Round selection and comparison for the Agentic-DPO comparator, for one model.
Selection: among rounds 1-5, the round with the lowest semantic mean on the development replicates (0-9); ties go to the earlier round.
Comparison on the fresh replicates (80-99) only, against this method's fresh semantic mean:
MATCH if the selected round passes all checks of judge_v11.py and makes fewer than 5 exploit-instead-of-check errors (exploit_count.py);
V13_BETTER if it fails a check, makes 10 or more such errors, or its semantic mean is at least 0.02 higher; otherwise CLOSE.
Usage: python adpo_gate.py QWEN|OLMO METHOD_FRESH_SEMANTIC_MEAN   (judge outputs are read from ./out, or from $JDIR)"""
import json, os, sys
from teacher_agreement import run
T, v13 = sys.argv[1], float(sys.argv[2])
rows = []
for R in range(1, 6):
    arm = f"ADAPT_ADPO_R{R}_{T}_S1"
    try: j = json.load(open(os.path.join(os.environ.get("JDIR", "out"), f"judge_v11_{arm}.json")))
    except FileNotFoundError: continue
    rows.append((j["dev_reps_0_9"]["semantic_mean"]["ADAPT"], R, arm, j))
assert rows, "no judged rounds"
dev, R, arm, j = min(rows, key=lambda x: (x[0], x[1]))
fr = j["fresh_reps_80_99"]; sm = fr["semantic_mean"]["ADAPT"]
tot, bad, kinds = run(arm)
E = sum(v for (s, st, kd), v in kinds.items() if s == "fresh" and kd == "exploit-instead-of-check" and st in ("codes/tried2", "labels/tried2/contradiction"))
if (not fr["ALL_PASS"]) or E >= 10 or sm >= v13 + 0.02: verdict = "V13_BETTER"
elif fr["ALL_PASS"] and E < 5: verdict = "MATCH"
else: verdict = "CLOSE"
print(f"{verdict} selected=R{R} dev_sem={dev:.3f} fresh_sem={sm:.3f} fresh_pass={fr['ALL_PASS']} E_fresh={E} rounds_judged={len(rows)}")
