"""Matched script-distilled control for the five-arm test: v13 training data with every five-arm semantic decision relabelled
by the CONTRA contradiction script (k5_policy_sets.contra_action) wherever it differs from the exact DP teacher. Same prompts, same
rationale (state description), same per-decision copies, same line count; only the final decision sentence and the <action> change.
Three-arm decisions and meaningless-label ("code") decisions are untouched, so the control differs from v13 only where the five-arm
DP and the script disagree. Each relabelled record's original action is first checked against a recomputed DP action.
Usage: python make_script_control.py v13/train_adaptive_v13.jsonl train_script_k5.jsonl"""
import gzip, json, re, sys, collections
import k5_policy_sets as P

src, dst = sys.argv[1], sys.argv[2]
op = gzip.open if src.endswith(".gz") else open
recs = [json.loads(l) for l in op(src, "rt", encoding="utf-8")]
stats = collections.Counter()
out = []
for r in recs:
    m = r["meta"]
    if m["k"] != 5 or m["kind"] == "code":
        out.append(r); stats["unchanged_other"] += 1; continue
    think, rest = r["target"].split("</think>", 1)
    labels = re.search(r"Label ranking: (.+?)\. (?:Nothing observed yet|Observed:)", think).group(1).split(" > ")
    assert len(labels) == 5, labels
    obs = []
    om = re.search(r"Observed: (.+?)\. Tried", think)
    if om:
        for pos, lab in enumerate(labels):
            mm = re.search(re.escape(lab) + r" (-?[0-9.]+(?:e-?[0-9]+)?) \((\d+)\)", om.group(1))
            if mm:
                obs.append((mm.start(), pos))              # the generator lists observed arms best -> worst (display rounding can tie)
    order = [pos for _, pos in sorted(obs)]
    left = P.T - m["t"]
    dp = P.dp_action(order, left)
    body = think.rstrip()
    hits = [(kind, pos, body[: -len(s)]) for pos, lab in enumerate(labels)
            for kind, s in (("probe", f"Check {lab} next."), ("commit", f"{lab} is very likely best; exploit it.")) if body.endswith(s)]
    assert len(hits) == 1, body[-200:]
    rec_kind, rec_pos, head = hits[0]
    if (rec_kind, rec_pos) != dp and not (dp[0] == "commit" and rec_kind == "commit" and rec_pos == dp[1]):
        stats["dp_recompute_mismatch"] += 1; out.append(r); continue
    c = P.contra_action(order, set(order))
    if c == dp:
        out.append(r); stats["k5_same_action"] += 1; continue
    lab = labels[c[1]]
    new_sent = f"Check {lab} next." if c[0] == "probe" else f"{lab} is very likely best; exploit it."
    new_think = head + new_sent + "\n"
    nr = dict(r); nr["target"] = new_think + "</think>\n\n<action> " + lab + " </action>"
    nr["meta"] = dict(m, script_relabelled=True, dp_action=list(dp), script_action=list(c))
    out.append(nr); stats["k5_relabelled"] += 1
with open(dst, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(dict(stats), "lines", len(out))
