"""Five-arm policy identification set, CPU only, noise-free order-only information, T = 10.
Policies on semantic positions (0 = top label):
  DP      exact Bayes-adaptive teacher (sbadapt_oracle_bayes, label-reliability prior), the policy v13 was distilled from;
  CONTRA  the contradiction rule as a state rule: probe position 0, then 1; if the observed order of {0, 1}
          agrees with the labels commit to the best observed; otherwise keep probing untried positions in label order and commit to
          the best observed once all are tried.
Outputs (k5_policy_sets.json): official-cell regret of both policies (helpful = labels in true order, misleading = top and bottom
labels swapped, means 75 / 62.5 / 50 / 37.5 / 25), and every state reached by either policy on any of the 120 true orderings where
the two policies choose different actions (the mechanism probe set), with one consistent true ordering per state for rendering."""
import itertools, json
import sbadapt_oracle_bayes as B

K, T = 5, 10


def dp_action(order, left):
    _, (kind, pos) = B.V(K, True, 0.0, tuple(order), max(left, 1))
    if kind == "commit" and not order:
        kind = "probe"
    return ("commit" if kind == "commit" else "probe"), pos


def contra_action(order, tried):
    """order: tried positions sorted best -> worst by observed mean."""
    if not tried:
        return "probe", 0
    if tried == {0}:
        return "probe", 1
    agree01 = order.index(0) < order.index(1) if {0, 1} <= tried else True
    if agree01 and tried == {0, 1}:
        return "commit", order[0]
    untried = [p for p in range(K) if p not in tried]
    if untried:
        return "probe", untried[0]
    return "commit", order[0]


def trajectory(true, policy):
    order, tried, left, states, seq = [], set(), T, [], []
    while left:
        a = dp_action(order, left) if policy == "DP" else contra_action(order, tried)
        states.append((tuple(order), left))
        kind, pos = a
        if kind == "commit":
            seq += [pos] * left
            break
        seq.append(pos); tried.add(pos); left -= 1
        order = sorted(tried, key=lambda p: -true[p])
    return seq, states


def regret(seq, true):
    return sum((K - 1) - true[p] for p in seq) / (len(seq) * (K - 1))


if __name__ == "__main__":
    helpful = [4, 3, 2, 1, 0]                      # true[pos] = rank value of the arm under label position pos (4 = best)
    misleading = [0, 3, 2, 1, 4]                   # top and bottom labels swapped
    cells = {}
    for name, true in (("helpful", helpful), ("misleading", misleading)):
        cells[name] = {p: round(regret(trajectory(true, p)[0], true), 4) for p in ("DP", "CONTRA")}
        cells[name]["DP_seq"] = trajectory(true, "DP")[0]; cells[name]["CONTRA_seq"] = trajectory(true, "CONTRA")[0]
    dis = {}
    for perm in itertools.permutations(range(K)):
        true = [0] * K
        for rank, pos in enumerate(perm):
            true[pos] = (K - 1) - rank
        for pol in ("DP", "CONTRA"):
            for order, left in trajectory(true, pol)[1]:
                tried = set(order)
                d, c = dp_action(list(order), left), contra_action(list(order), tried)
                if d != c and (order, left) not in dis:
                    dis[(order, left)] = {"order_best_to_worst": list(order), "turns_left": left, "step": T - left,
                                          "dp": d, "contra": c, "example_true": true, "reached_by": pol}
    out = {"official_cells": cells, "semantic_mean": {p: round((cells["helpful"][p] + cells["misleading"][p]) / 2, 4) for p in ("DP", "CONTRA")},
           "disagreement_states": list(dis.values())}
    json.dump(out, open("k5_policy_sets.json", "w"), indent=1)
    print(json.dumps({"official_cells": cells, "semantic_mean": out["semantic_mean"], "n_disagreement_states": len(dis)}))
    for s in list(dis.values())[:12]:
        print(s["order_best_to_worst"], "step", s["step"], "DP", s["dp"], "CONTRA", s["contra"], "via", s["reached_by"])
