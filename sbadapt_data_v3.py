"""Training decisions, step 2: the generator of sbadapt_data.py (same scenarios, pools, regimes, rewards, seed and
episode count), with actions from the exact Bayes-adaptive planner (sbadapt_oracle_bayes.act, EPS = 0) instead of the greedy
'verify the most probable untried arm' rule. Labels that contain a word of the paper's sentiment pools are renamed
(pools_v2.RENAME) and 'average-ish' is dropped, so every label is token-disjoint from the paper's pools (asserted)."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sbadapt_data as D
import sbadapt_oracle_bayes as B
import pools_v2 as P

D.ORD = [[P.RENAME.get(x, x) for x in p] for p in D.ORD]
D.SCEN = [(s[0], s[1], [[P.RENAME.get(x, x) for x in p] for p in s[2]]) for s in D.SCEN]
D.SENT_NEU = [w for w in D.SENT_NEU if w.lower() not in P.DROP_WORDS]
for p in D.ORD + [p for s in D.SCEN for p in s[2]]:
    for lab in p:
        bad = [w for w in P.tokens(lab) if w in D.PAPER_WORDS]; assert not bad, (lab, bad)
for w in D.SENT_POS + D.SENT_NEU + D.SENT_NEG:
    assert not [t for t in P.tokens(w) if t in D.PAPER_WORDS], w


class _BayesOracle:          # drop-in for the 'O' module used inside sbadapt_data.episode
    act = staticmethod(lambda k, s0, counts, means: B.act(k, s0, counts, means))


D.O = _BayesOracle


def rationale(labels_sem, labels_by_arm, counts, means, pb, mode_name, a, informative):
    obs = [f"{labels_by_arm[i]} {means[i]:.1f} ({counts[i]})" for i in range(len(counts)) if counts[i] > 0]
    s = ("Label ranking: " + " > ".join(labels_sem) + ". ") if informative else "Labels carry no meaning. "
    s += ("Observed: " + ", ".join(obs) + ". ") if obs else "Nothing observed yet. "
    s += f"{labels_by_arm[a]} is very likely best; exploit it." if mode_name == "exploit" else f"Not sure yet; check {labels_by_arm[a]} next."
    return s


D.rationale = rationale

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["adaptive", "agnostic"], required=True)
    ap.add_argument("--episodes", type=int, default=450); ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--out", required=True)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for e in range(a.episodes):
            for r in D.episode(rng, a.mode == "adaptive"):
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    print("wrote", n, "decisions to", a.out)
