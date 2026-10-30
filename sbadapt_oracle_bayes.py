"""Exact Bayes-adaptive planner for the summarised-history semantic bandit, k = 3 or 5, T = 10.
Hidden variable: how reliable the label ordering is. Prior over orderings of the arm means = the regime mixture of
sbadapt_oracle.py (aligned .45 / reversed .15 / swapped .15 / uniform .25; uniform only when the labels carry no order), optionally
with rank noise EPS on adjacent transpositions of each regime ordering. Information model: scale- and offset-invariant, so only the
ORDER of observed means is informative (with a log-uniform spacing prior a rank gap of 1 or 2 is equally likely given an observed
gap). The policy minimises expected cumulative regret (rank-value units) over the remaining turns by exact dynamic programming over
(observed order, turns left); exploiting = committing to an arm forever is optimal once no new arm will be probed, because
re-pulling an observed arm gives no information without reward noise.
Interface: act(k, s0, counts, means) -> (arm, pb, mode), as in sbadapt_oracle.py."""
import itertools, functools
import numpy as np
import sbadapt_oracle as V1

PRIOR, EPS, T = V1.PRIOR, 0.0, 10


def noisy(order, eps):
    out = {tuple(order): 1.0 - eps * (len(order) - 1)}
    for j in range(len(order) - 1):
        o = list(order); o[j], o[j + 1] = o[j + 1], o[j]; out[tuple(o)] = out.get(tuple(o), 0.0) + eps
    return out


@functools.lru_cache(None)
def prior_sem(k, informative, eps):
    """Prior over orderings expressed in SEMANTIC-POSITION space: an ordering is a tuple p (best -> worst) of semantic positions."""
    perms = list(itertools.permutations(range(k)))
    w = {p: (PRIOR["uniform"] if informative else 1.0) / len(perms) for p in perms}
    if informative:
        al = list(range(k)); rv = al[::-1]; sw = list(al); sw[0], sw[-1] = sw[-1], sw[0]
        for name, order in (("aligned", al), ("reversed", rv), ("swapped", sw)):
            for p, m in noisy(order, eps).items(): w[p] += PRIOR[name] * m
    return tuple(sorted(w.items()))


@functools.lru_cache(None)
def V(k, informative, eps, order, left):
    """order: observed semantic positions sorted by observed mean, best first. Returns (expected regret, action)."""
    pr = dict(prior_sem(k, informative, eps))
    cons = {}
    for p, w in pr.items():
        if w <= 0: continue
        rk = {pos: i for i, pos in enumerate(p)}
        if all(rk[order[i]] < rk[order[i + 1]] for i in range(len(order) - 1)): cons[p] = w
    z = sum(cons.values())
    if left == 0 or z == 0: return 0.0, ("commit", order[0] if order else 0)
    q = {p: w / z for p, w in cons.items()}
    val = lambda pos: sum(qq * p.index(pos) for p, qq in q.items())          # expected rank (0 = best) of semantic position pos
    best_commit = min(range(k), key=lambda pos: (val(pos), pos))
    best = (left * val(best_commit), ("commit", best_commit))
    for u in range(k):
        if u in order: continue
        cost = 0.0
        groups = {}
        for p, qq in q.items():
            new = tuple(sorted(order + (u,), key=lambda pos: p.index(pos)))
            groups.setdefault(new, []).append((p, qq))
        for new, lst in groups.items():
            mass = sum(qq for _, qq in lst)
            imm = sum(qq * p.index(u) for p, qq in lst)
            cost += imm + mass * V(k, informative, eps, new, left - 1)[0]
        if cost < best[0] - 1e-12: best = (cost, ("probe", u))
    return best


def act(k, s0, counts, means, eps=None):
    eps = EPS if eps is None else eps
    informative = s0 is not None
    sem = list(s0) if informative else list(range(k))                       # semantic position -> arm id
    pos_of = {a: i for i, a in enumerate(sem)}
    pulled = [a for a in range(k) if counts[a] > 0]
    order = tuple(pos_of[a] for a in sorted(pulled, key=lambda a: -means[a]))
    left = T - sum(counts)
    _, (kind, pos) = V(k, informative, eps, order, max(left, 1))
    if kind == "commit" and not pulled: kind = "probe"
    arm = sem[pos]
    pb = np.zeros(k); pb[arm] = 1.0
    return arm, pb, ("exploit" if kind == "commit" else "verify")


def run(mu, s0, eps=None, sigma=0.0, rng=None):
    k = len(mu); counts = [0] * k; sums = [0.0] * k; arms = []
    for t in range(T):
        means = [sums[a] / counts[a] if counts[a] else 0.0 for a in range(k)]
        a = act(k, s0, counts, means, eps)[0]
        counts[a] += 1; sums[a] += mu[a] + (rng.normal(0, sigma) if sigma > 0 else 0.0); arms.append(a)
    return sum(max(mu) - mu[a] for a in arms) / (T * (max(mu) - min(mu))), "".join(("HML"[a] if k == 3 else str(a)) for a in arms)


if __name__ == "__main__":
    mu = [75, 50, 25]
    cases = [("helpful, correct", [0, 1, 2]), ("misleading, correct", [2, 1, 0]), ("misleading, v1 failure (bottom pair wrong)", [2, 0, 1]),
             ("misleading, top pair wrong", [1, 2, 0]), ("helpful, bottom pair wrong", [0, 2, 1]), ("helpful, top pair wrong", [1, 0, 2])]
    for eps in (0.0, 0.0655):
        print(f"--- EPS {eps}")
        for name, s0 in cases:
            r, seq = run(mu, s0, eps); r1 = V1.run_episode(mu, s0, 0.0)[0]
            print(f"{name:45s} bayes {r:.3f} {seq} | v1 greedy {r1:.3f}")
        print("uninformative                                 bayes %.3f %s" % run(mu, None, eps))
        allp = [run([[75, 50, 25][p.index(a)] for a in range(3)], [0, 1, 2], eps)[0] for p in itertools.permutations(range(3))]
        print("all 6 true orders under s0=[0,1,2]: mean %.3f max %.3f" % (np.mean(allp), max(allp)))
        mu5 = [75, 62.5, 50, 37.5, 25]
        for name, s0 in (("k5 helpful", [0, 1, 2, 3, 4]), ("k5 swapped", [4, 1, 2, 3, 0]), ("k5 reversed", [4, 3, 2, 1, 0]), ("k5 uninformative", None)):
            print(f"{name:20s} bayes {run(mu5, s0, eps)[0]:.3f}" + ("" if s0 is None else f" | v1 greedy {V1.run_episode(mu5, s0, 0.0)[0]:.3f}"))
    rng = np.random.default_rng(0)
    for sigma in (6.25, 12.5):
        for name, s0 in (("helpful", [0, 1, 2]), ("misleading", [2, 1, 0]), ("v1 failure", [2, 0, 1])):
            print(f"sigma {sigma:5} {name:11s} bayes {np.mean([run(mu, s0, 0.0, sigma, rng)[0] for _ in range(300)]):.3f}")
