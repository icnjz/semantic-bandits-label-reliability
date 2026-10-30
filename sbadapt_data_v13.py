"""Class balance ('v13', the final recipe): the v11 lines with one change: every meaningless-label ('code') decision
gets round(copies x R) copies, R = ranked-label lines / code lines in v11 (3.49), so the two classes of the episode-level decision
'are these labels meaningful?' are balanced.
Why: Qwen3-8B decides label meaning at the first generated rationale token ('Label' -> ranking vs 'Labels' -> no meaning). In v11
the two classes are 7,243 : 2,077 lines (3.49 : 1) at that token, and a v11 adapter invented a ranking for random codes in all 600
alphanumeric decisions.
Usage: python sbadapt_data_v13.py v11/train_adaptive_v11.jsonl v13/train_adaptive_v13.jsonl"""
import json, sys
src, dst = sys.argv[1], sys.argv[2]
lines = [l for l in open(src, encoding="utf-8")]
recs, i = [], 0
while i < len(lines):                       # v11 writes each decision 'copies' times consecutively
    r = json.loads(lines[i]); recs.append(r); i += r["meta"]["copies"]
assert sum(r["meta"]["copies"] for r in recs) == len(lines)
code = sum(r["meta"]["copies"] for r in recs if r["meta"]["kind"] == "code"); ranked = len(lines) - code; R = ranked / code
n = 0
with open(dst, "w", encoding="utf-8") as f:
    for r in recs:
        c = int(round(r["meta"]["copies"] * R)) if r["meta"]["kind"] == "code" else r["meta"]["copies"]
        r["meta"]["copies_v13"] = c
        for _ in range(c): f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
print("decisions", len(recs), "v11 lines", len(lines), "ranked", ranked, "code", code, "R", round(R, 3), "-> v13 lines", n)
