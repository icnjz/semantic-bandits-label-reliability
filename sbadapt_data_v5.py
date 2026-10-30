"""Training decisions, step 3: v3 with one change: the rationale states the decision-relevant state before the
action: how many labels are tried, which are untried, and whether the observed order agrees with the label ranking. For k = 3 these
three facts determine the Bayes action exactly (checked by --check: action purity per stated state >= .95), so the action is
conditioned on a label-independent description of the state."""
import argparse, json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sbadapt_data_v3 as V3
D = V3.D


def state_clause(labels_sem, labels_by_arm, counts, means, informative):
    k = len(counts); pulled = [i for i in range(k) if counts[i] > 0]
    order = labels_sem if informative else [labels_by_arm[i] for i in range(k)]
    untried = [l for l in order if counts[[i for i in range(k) if labels_by_arm[i] == l][0]] == 0]
    s = f"Tried {len(pulled)} of {k}; untried: {', '.join(untried) if untried else 'none'}. "
    inv = None
    if informative and len(pulled) >= 2:
        pos = {lab: i for i, lab in enumerate(labels_sem)}
        invs = [(i, j) for i in pulled for j in pulled if pos[labels_by_arm[i]] > pos[labels_by_arm[j]] and means[i] > means[j]]
        if invs:
            i, j = max(invs, key=lambda p: means[p[0]] - means[p[1]])
            inv = True; s += f"Order check: {labels_by_arm[i]} beats the higher-ranked {labels_by_arm[j]}. "
        else:
            inv = False; s += "Order check: agrees with the ranking. "
    return s, (k, len(pulled), len(untried) > 0, inv)


def rationale(labels_sem, labels_by_arm, counts, means, pb, mode_name, a, informative):
    obs = [f"{labels_by_arm[i]} {means[i]:.1f} ({counts[i]})" for i in range(len(counts)) if counts[i] > 0]
    s = ("Label ranking: " + " > ".join(labels_sem) + ". ") if informative else "Labels carry no meaning. "
    s += ("Observed: " + ", ".join(obs) + ". ") if obs else "Nothing observed yet. "
    s += state_clause(labels_sem, labels_by_arm, counts, means, informative)[0]
    s += f"{labels_by_arm[a]} is very likely best; exploit it." if mode_name == "exploit" else f"Check {labels_by_arm[a]} next."
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
