"""Error decomposition on CPU: for every evaluation decision of an adapter, recompute what the planner (exact
order-only Bayes-adaptive DP, sigma 0) would do in exactly that state, and classify the model's disagreements by state type.
States are rebuilt from decisions.jsonl: final decisions of each replicate in turn order (reward = arm mean, sigma 0).
Semantic order: helpful -> labels follow H, M, L; misleading -> top and bottom labels swapped (the top label sits on arm L);
alphanumeric -> labels carry no order (planner = label-ignoring DP). Usage: python teacher_agreement.py ARM [ARM ...]"""
import collections, json, sys
import sbadapt_oracle_bayes as B

IDX = {"H": 0, "M": 1, "L": 2}; MEAN = {"H": 75.0, "M": 50.0, "L": 25.0}


def state_type(counts, means, s0):
    k = 3; pulled = [a for a in range(k) if counts[a] > 0]; n = len(pulled)
    if s0 is None: return f"codes/tried{n}"
    pos = {a: i for i, a in enumerate(s0)}
    inv = any(pos[a] > pos[b] and means[a] > means[b] for a in pulled for b in pulled)
    return f"labels/tried{n}/" + ("contradiction" if inv else "agrees" if n >= 2 else "-")


def run(arm):
    recs = collections.defaultdict(list)
    for l in open(f"out/{arm}/decisions.jsonl", encoding="utf-8"):
        d = json.loads(l)
        if d.get("final", True) and d.get("arm") is not None: recs[(d["domain"], d["nom"], d["align"], d["rep"])].append(d)
    tot = collections.Counter(); bad = collections.Counter(); kinds = collections.Counter()
    for (dom, nom, al, rep), ds in recs.items():
        ds.sort(key=lambda d: d["turn"])
        s0 = None if nom == "alphanumeric" else ([0, 1, 2] if al == "helpful" else [2, 1, 0])
        counts = [0, 0, 0]; sums = [0.0, 0.0, 0.0]
        for d in ds:
            means = [sums[a] / counts[a] if counts[a] else 0.0 for a in range(3)]
            teacher = B.act(3, s0, counts, means)[0]
            st = state_type(counts, means, s0); split = "dev" if rep < 10 else "fresh" if rep >= 80 else "seen"
            tot[(split, st)] += 1
            a = IDX[d["arm"]]
            if a != teacher:
                bad[(split, st)] += 1
                tried_a = counts[a] > 0
                kinds[(split, st, "exploit-instead-of-check" if tried_a and counts[teacher] == 0 else
                       "check-instead-of-exploit" if not tried_a and counts[teacher] > 0 else "other")] += 1
            counts[a] += 1; sums[a] += MEAN[d["arm"]]
    return tot, bad, kinds


if __name__ == "__main__":
    for arm in sys.argv[1:]:
        tot, bad, kinds = run(arm)
        for split in ("dev", "fresh"):
            T = sum(v for k, v in tot.items() if k[0] == split); E = sum(v for k, v in bad.items() if k[0] == split)
            if not T: continue
            worst = sorted(((k[1], bad[k], tot[k]) for k in tot if k[0] == split and bad[k]), key=lambda x: -x[1])[:4]
            print(f"{arm:22s} {split:5s} disagree {E:4d}/{T} = {E / T:.3f} | top states: " + "; ".join(f"{s} {b}/{t}" for s, b, t in worst))
        ks = sorted(((k, v) for k, v in kinds.items() if k[0] in ("dev", "fresh")), key=lambda x: -x[1])[:4]
        print(" " * 23, "kinds:", "; ".join(f"{k[0]} {k[1]} {k[2]} {v}" for k, v in ks))
