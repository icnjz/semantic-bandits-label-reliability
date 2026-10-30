"""Control for v13: the cost-sensitive copy counts shuffled within each label kind. Each v11 decision keeps its
prompt, rationale and action; the copy counts round(1 + gap / mean_gap) are permuted among the decisions of the same kind (ordinal /
sentiment / world / code), so every kind keeps its line total and the copy histogram is unchanged, but a decision's weight no longer
depends on its own action gap. The v13 class balance is then applied unchanged.
If this control matches v13, the gain comes from extra copies and class balance, not from the exact decision gap.
Usage: python sbadapt_data_v13shuf.py v11/train_adaptive_v11.jsonl v13shuf/train_adaptive_v13shuf.jsonl"""
import json, sys
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
lines = [l for l in open(src, encoding="utf-8")]
recs, i = [], 0
while i < len(lines):                       # v11 writes each decision 'copies' times consecutively
    r = json.loads(lines[i]); recs.append(r); i += r["meta"]["copies"]
assert sum(r["meta"]["copies"] for r in recs) == len(lines)
rng = np.random.default_rng(2026)
for kind in sorted({r["meta"]["kind"] for r in recs}):
    idx = [j for j, r in enumerate(recs) if r["meta"]["kind"] == kind]
    perm = rng.permutation([recs[j]["meta"]["copies"] for j in idx])
    for j, c in zip(idx, perm): recs[j]["meta"]["copies_shuf"] = int(c)
code = sum(r["meta"]["copies_shuf"] for r in recs if r["meta"]["kind"] == "code"); ranked = len(lines) - code; R = ranked / code
corr = np.corrcoef([r["meta"]["action_gap"] for r in recs], [r["meta"]["copies_shuf"] for r in recs])[0, 1]
n = 0
with open(dst, "w", encoding="utf-8") as f:
    for r in recs:
        c = int(round(r["meta"]["copies_shuf"] * R)) if r["meta"]["kind"] == "code" else r["meta"]["copies_shuf"]
        r["meta"]["copies_v13shuf"] = c
        for _ in range(c): f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
print("decisions", len(recs), "ranked", ranked, "code", code, "R", round(R, 3), "corr(gap, shuffled copies)", round(float(corr), 3),
      "-> lines", n)
