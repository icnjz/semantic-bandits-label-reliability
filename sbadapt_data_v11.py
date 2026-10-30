"""Cost-sensitive copies ('v11'): the v7 decisions (prompts, rationales and actions unchanged), each repeated
round(1 + gap / mean_gap) times, where gap is the planner's exact action gap: the expected regret of the best alternative action
minus that of the planner's action, from the same Bayes-adaptive DP.
Why: almost all regret-causing student errors are one decision, exploiting after two tried labels when the planner checks the third.
The v7 data hold 1,219 k = 3 'exploit at two tried' decisions against 231 'check the third' (5.3 : 1), while the action gap of a
'check the third' decision is about 4 times that of an 'exploit' decision (labels: 3.7-4.1 vs 0.9-1.0; codes: 1.2 vs 0.3)."""
import argparse, collections, itertools, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sbadapt_data_v7 as V7
D = V7.D
import sbadapt_oracle_bayes as B


def qvals(k, informative, order, left):
    pr = dict(B.prior_sem(k, informative, 0.0)); cons = {}
    for p, w in pr.items():
        rk = {pos: i for i, pos in enumerate(p)}
        if all(rk[order[i]] < rk[order[i + 1]] for i in range(len(order) - 1)): cons[p] = w
    z = sum(cons.values()); q = {p: w / z for p, w in cons.items()}
    val = lambda pos: sum(qq * p.index(pos) for p, qq in q.items())
    Q = {}
    for pos in order: Q[("commit", pos)] = left * val(pos)
    if not order:
        for pos in range(k): Q[("commit", pos)] = left * val(pos)
    for u in range(k):
        if u in order: continue
        groups = {}
        for p, qq in q.items():
            new = tuple(sorted(order + (u,), key=lambda pos: p.index(pos))); groups.setdefault(new, []).append((p, qq))
        Q[("probe", u)] = sum(sum(qq * p.index(u) for p, qq in lst) + sum(qq for _, qq in lst) * B.V(k, informative, 0.0, new, left - 1)[0]
                              for new, lst in groups.items())
    return Q


_GAPS = []


class _GapOracle:
    @staticmethod
    def act(k, s0, counts, means):
        arm, pb, mode = B.act(k, s0, counts, means)
        informative = s0 is not None; sem = list(s0) if informative else list(range(k)); pos_of = {a: i for i, a in enumerate(sem)}
        pulled = [a for a in range(k) if counts[a] > 0]
        order = tuple(pos_of[a] for a in sorted(pulled, key=lambda a: -means[a])); left = max(B.T - sum(counts), 1)
        Q = qvals(k, informative, order, left); chosen = Q.get(("commit" if mode == "exploit" else "probe", pos_of[arm]))
        if chosen is None: chosen = min(Q.values())
        alts = [v for key, v in Q.items() if key[1] != pos_of[arm]]           # any action that pulls a different arm
        _GAPS.append(max(min(alts) - chosen, 0.0) if alts else 0.0)
        return arm, pb, mode


D.O = _GapOracle

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["adaptive", "agnostic"], required=True)
    ap.add_argument("--episodes", type=int, default=450); ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--out", required=True)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); recs = []
    for e in range(a.episodes):
        _GAPS.clear(); ep = D.episode(rng, a.mode == "adaptive"); assert len(ep) == len(_GAPS)
        for r, g in zip(ep, list(_GAPS)): r["meta"]["action_gap"] = round(float(g), 4); recs.append(r)
    G = float(np.mean([r["meta"]["action_gap"] for r in recs]))
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for r in recs:
            c = int(round(1 + r["meta"]["action_gap"] / G)); r["meta"]["copies"] = c
            for _ in range(c): f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    print("decisions", len(recs), "mean gap G", round(G, 3), "written (with copies)", n)
