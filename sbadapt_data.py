"""Training decisions for semantic-prior-adaptive exploration.

Episodes use scenarios, label pools and label formats that are DISJOINT from the Semantic Bandits paper (no Bandit / Farm /
Clothing task prompt, none of the paper's ordinal / sentiment / world / alphanumeric pools). The prompt layout is the paper's
budget_summhist_v1 template (also used by sbandit_harness.py), so evaluation on the official cells is unchanged.
Each decision is labelled by sbadapt_oracle.act (semantic-regime posterior policy) with a short rationale inside <think>.
--mode adaptive : oracle sees the pool's semantic order (the method)
--mode agnostic : oracle treats every label set as uninformative (label-agnostic control; same episodes, same prompts)
"""
import argparse, json, os, random, string, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sbadapt_oracle as O

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "CHEUNG_SBANDIT_work"))
import sbandit_harness as H   # template strings only (TEMPLATE_HEAD / LINE / TAIL, THINKING_BUDGET, NEW_TOKEN_BUDGET)

SCEN = [  # (task prompt with {k}, action suffix, world-knowledge pools best -> worst)
    ("You are a fishing guide choosing among {k} fishing spots for today's trip. Each action corresponds to taking clients to a different spot. Each spot is associated with a Gaussian distribution with a fixed but unknown mean catch value. At each time step, you will select a spot and receive a reward drawn from the associated distribution. Your goal is to maximize your reward across all of your turns.",
     "Spot", [["Teeming Coral Reef", "Quiet Harbour Cove", "Polluted Drainage Outlet"], ["Salmon Spawning Run", "Muddy River Bend", "Dry Creek Bed"]]),
    ("You manage a delivery company and choose among {k} routes for each shipment. Each action corresponds to sending the truck along a different route. Each route is associated with a Gaussian distribution with a fixed but unknown mean on-time score. At each time step, you will select a route and receive a reward drawn from the associated distribution. Your goal is to maximize your reward across all of your turns.",
     "Route", [["Open Express Highway", "Busy Suburban Avenue", "Flooded Mountain Track"], ["Toll-free Motorway", "Downtown Side Streets", "Closed Construction Zone"]]),
    ("You run a mining operation with {k} candidate dig sites. Each action corresponds to working a different site for one shift. Each site is associated with a Gaussian distribution with a fixed but unknown mean ore value. At each time step, you will select a site and receive a reward drawn from the associated distribution. Your goal is to maximize your reward across all of your turns.",
     "Site", [["Rich Gold Vein", "Mixed Copper Seam", "Exhausted Gravel Pit"], ["Untouched Diamond Pipe", "Average Iron Deposit", "Collapsed Abandoned Shaft"]]),
    ("You are an advertising planner choosing among {k} ad placements. Each action corresponds to buying a different placement for one day. Each placement is associated with a Gaussian distribution with a fixed but unknown mean sales lift. At each time step, you will select a placement and receive a reward drawn from the associated distribution. Your goal is to maximize your reward across all of your turns.",
     "Placement", [["Prime-time Television Slot", "Regional Radio Spot", "Expired Paper Flyer"], ["Stadium Big Screen", "Bus Shelter Poster", "Spam Folder Email"]]),
    ("You operate a cloud service and choose among {k} server regions for incoming jobs. Each action corresponds to routing a job to a different region. Each region is associated with a Gaussian distribution with a fixed but unknown mean throughput score. At each time step, you will select a region and receive a reward drawn from the associated distribution. Your goal is to maximize your reward across all of your turns.",
     "Region", [["New Fiber Datacenter", "Shared Hosting Rack", "Overheating Legacy Closet"], ["Dedicated GPU Cluster", "Standard Virtual Machine", "Dial-up Basement Server"]]),
    ("You are a coach choosing among {k} training drills for your team each day. Each action corresponds to running a different drill. Each drill is associated with a Gaussian distribution with a fixed but unknown mean improvement score. At each time step, you will select a drill and receive a reward drawn from the associated distribution. Your goal is to maximize your reward across all of your turns.",
     "Drill", [["Elite Interval Session", "Light Jogging Loop", "Sitting on the Bench"], ["Olympic Coach Program", "Generic Stretching Routine", "Skipping Practice Entirely"]]),
]
ORD = [["Top Tier", "Middle Tier", "Bottom Tier"], ["Best Choice", "Average Choice", "Worst Choice"], ["Gold Rank", "Silver Rank", "Bronze Rank"],
       ["First Place", "Second Place", "Third Place"], ["Premium Grade", "Standard Grade", "Reject Grade"], ["Maximum Payoff", "Moderate Payoff", "Minimum Payoff"]]
SENT_POS = "great amazing fantastic glorious delightful admirable lovely awesome incredible perfect ideal wondrous dazzling heavenly sensational".split()
SENT_NEU = "fair okay adequate middling passable unremarkable acceptable neutral tolerable so-so average-ish mild unexceptional decent fine".split()
SENT_NEG = "bad poor dire inferior shoddy miserable rotten horrid nasty crummy subpar faulty woeful dismaying grim".split()
PAPER_WORDS = set(w.lower() for p in H.SENT_POOLS.values() for w in p) | set(w.lower() for v in list(H.ORDINAL.values()) + list(H.WORLD_FARM.values()) for w in v.split())
for w in SENT_POS + SENT_NEU + SENT_NEG: assert w.lower() not in PAPER_WORDS, w


def code_label(rng):
    return rng.choice(["X", "Q", "Z", "K"]) + "-" + "".join(rng.choice(list(string.digits), 4))


def make_pool(rng, kind, k, scen):
    """Returns labels best -> worst (semantic order) and whether they are informative."""
    if kind == "ordinal":
        base = rng.choice(ORD)
        if k == 3: return list(base), True
        return [f"Rank {i + 1} of {k}" for i in range(k)], True
    if kind == "sentiment":
        pools = [SENT_POS, SENT_NEU, SENT_NEG] if k == 3 else [SENT_POS, SENT_POS, SENT_NEU, SENT_NEG, SENT_NEG]
        out, used = [], set()
        for p in pools:
            w = rng.choice([x for x in p if x not in used]); used.add(w); out.append(w.capitalize())
        if k == 5:   # within-polarity order is not semantically defined: treat the 5-arm sentiment set as 3 tiers -> only top-vs-bottom is informative
            return out, True
        return out, True
    if kind == "world":
        return list(rng.choice(scen[2])), True
    labels = set()
    while len(labels) < k: labels.add(code_label(rng))
    return list(labels), False


def rationale(labels_sem, labels_by_arm, counts, means, pb, mode_name, a, informative):
    obs = [f"{labels_by_arm[i]} {means[i]:.1f} ({counts[i]})" for i in range(len(counts)) if counts[i] > 0]
    s = ("Label ranking: " + " > ".join(labels_sem) + ". ") if informative else "Labels carry no meaning. "
    s += ("Observed: " + ", ".join(obs) + ". ") if obs else "Nothing observed yet. "
    if mode_name == "exploit":
        s += f"{labels_by_arm[a]} is very likely best; exploit it."
    elif mode_name == "verify":
        s += f"Not sure yet; check {labels_by_arm[a]}, the most likely best arm not yet tried."
    else:
        s += f"Uncertain; {labels_by_arm[a]} has the highest chance of being best."
    return s


def episode(rng, adaptive):
    k = 3 if rng.random() < 0.85 else 5
    scen = SCEN[rng.integers(len(SCEN))]
    kind = rng.choice(["ordinal", "sentiment", "world", "code"], p=[0.3, 0.3, 0.2, 0.2])
    if k == 5 and kind in ("world", "sentiment"): kind = "ordinal"
    pool, informative = make_pool(rng, kind, k, scen)
    gap = float(rng.choice([1, 1, 2, 5, 10, 25, 50, 100, 0.25, 0.01]))
    center = float(rng.choice([0.0, 2.0, -2.0, 1.0, -1.0])) * gap * k
    means_sorted = [center + gap * (k - 1 - i) + (rng.normal(0, 0.15 * gap) if rng.random() < 0.3 else 0.0) for i in range(k)]   # best -> worst
    means_sorted = sorted(means_sorted, reverse=True)
    sigma = float(rng.choice([0.0, 0.0, 0.25, 0.5])) * gap
    regime = rng.choice(["aligned", "swapped", "reversed", "random"], p=[0.45, 0.30, 0.10, 0.15]) if informative else "none"
    # label for the arm of true rank r (0 = best)
    idx = list(range(k))
    if regime == "swapped": idx[0], idx[-1] = idx[-1], idx[0]
    elif regime == "reversed": idx = idx[::-1]
    elif regime == "random": idx = list(rng.permutation(k))
    label_of_rank = [pool[idx[r]] for r in range(k)]          # true rank r gets semantic position idx[r]
    arms = list(range(k))                                      # arm id = true rank
    s0 = [r for _, r in sorted((idx[r], r) for r in range(k))] if (informative and adaptive) else None   # arm ids by semantic position
    display = list(rng.permutation(k))
    counts, sums, recs = [0] * k, [0.0] * k, []
    task = scen[0].format(k=k)
    for t in range(H.NUM_TURNS):
        means = [sums[a] / counts[a] if counts[a] else 0.0 for a in arms]
        a, pb, mname = O.act(k, s0, counts, means)
        prompt = H.TEMPLATE_HEAD.format(num_turns=H.NUM_TURNS, current_turn=t, task_prompt=task)
        for arm in display:
            prompt += H.TEMPLATE_LINE.format(action_suffix=scen[1], action=label_of_rank[arm], count=counts[arm], obs_mean=str(float(means[arm])))
        prompt += H.TEMPLATE_TAIL.format(thinking_budget=H.THINKING_BUDGET, new_token_budget=H.NEW_TOKEN_BUDGET)
        sem = [pool[i] for i in range(k)]
        why = rationale(sem, label_of_rank, counts, means, pb, mname, a, informative and adaptive)
        recs.append({"prompt": prompt, "target": f"<think>\n{why}\n</think>\n\n<action> {label_of_rank[a]} </action>",
                     "meta": {"k": k, "kind": kind, "regime": regime, "sigma": sigma, "gap": gap, "t": t, "arm_rank": int(a), "mode": mname}})
        r = means_sorted[a] + (rng.normal(0, sigma) if sigma > 0 else 0.0)
        counts[a] += 1; sums[a] += r
    return recs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["adaptive", "agnostic"], required=True)
    ap.add_argument("--episodes", type=int, default=450); ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--out", required=True)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for e in range(a.episodes):
            for r in episode(rng, a.mode == "adaptive"):
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    print("wrote", n, "decisions to", a.out)
