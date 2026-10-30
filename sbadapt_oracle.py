"""Semantic-regime posterior policy for semantic bandits (Austin, Suleman, Cheung, COLM 2026 setting).

Hidden variable: the ordering pi of the arm means. Prior over pi = mixture of
 aligned (pi = semantic order s0),
 reversed (pi = s0 reversed),
 swapped (s0 with its top and bottom exchanged; equals 'reversed' when k = 3),
 uniform (any ordering; the only component when the labels carry no semantic order).
Likelihood: pairwise Gaussian comparisons of the observed per-arm means (summarised history: count, observed mean).
Policy: exploit the arm whose posterior probability of being best is >= TAU; otherwise pull the most probable
best arm among arms not yet pulled (verify a candidate); if every candidate is already pulled, pull the most probable best arm.
"""
import itertools, math
import numpy as np

PRIOR = {"aligned": 0.45, "reversed": 0.15, "swapped": 0.15, "uniform": 0.25}
TAU = 0.90


def _phi(z): return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def perm_prior(k, s0):
    """s0: list of arm ids from most to least semantically favoured, or None (uninformative labels)."""
    perms = list(itertools.permutations(range(k)))
    w = {p: 0.0 for p in perms}
    uni = PRIOR["uniform"] if s0 is not None else 1.0
    for p in perms: w[p] += uni / len(perms)
    if s0 is not None:
        al = tuple(s0); rv = tuple(reversed(s0)); sw = list(s0); sw[0], sw[-1] = sw[-1], sw[0]; sw = tuple(sw)
        w[al] += PRIOR["aligned"]; w[rv] += PRIOR["reversed"]; w[sw] += PRIOR["swapped"]
    return w


def posterior_best(k, s0, counts, means, sd_floor_frac=0.02):
    """P(arm a has the highest mean | data) under the mixture prior. counts / means indexed by arm id."""
    pulled = [a for a in range(k) if counts[a] > 0]
    scale = max([abs(means[a]) for a in pulled] + [1e-9])
    sd = sd_floor_frac * scale   # rewards are noisy at most at this relative level unless repeats show more (summarised history hides spread)
    w = perm_prior(k, s0); post = {}
    for p, pr in w.items():
        if pr == 0: continue
        rank = {a: i for i, a in enumerate(p)}   # 0 = best
        lik = 1.0
        for i, j in itertools.combinations(pulled, 2):
            hi, lo = (i, j) if rank[i] < rank[j] else (j, i)
            se = math.sqrt(sd ** 2 / counts[hi] + sd ** 2 / counts[lo])
            lik *= _phi((means[hi] - means[lo]) / max(se, 1e-12))
        post[p] = pr * lik
    z = sum(post.values()) or 1.0
    pb = np.zeros(k)
    for p, v in post.items(): pb[p[0]] += v / z
    return pb


def act(k, s0, counts, means):
    pb = posterior_best(k, s0, counts, means)
    if pb.max() >= TAU: return int(np.argmax(pb)), pb, "exploit"
    unpulled = [a for a in range(k) if counts[a] == 0]
    if unpulled:
        cand = max(unpulled, key=lambda a: (pb[a], -(s0.index(a) if s0 is not None else a)))
        if pb[cand] > 0: return int(cand), pb, "verify"
    return int(np.argmax(pb)), pb, "best_guess"


def ucb1(k, counts, means, t):
    for a in range(k):
        if counts[a] == 0: return a
    return int(np.argmax([means[a] + math.sqrt(2 * math.log(t + 1) / counts[a]) * max(1e-9, np.ptp(means)) for a in range(k)]))


def run_episode(mu, s0, sigma, T=10, policy="oracle", rng=None):
    k = len(mu); counts = [0] * k; sums = [0.0] * k; arms = []
    for t in range(T):
        means = [sums[a] / counts[a] if counts[a] else 0.0 for a in range(k)]
        a = act(k, s0, counts, means)[0] if policy == "oracle" else ucb1(k, counts, means, t)
        r = mu[a] + (rng.normal(0, sigma) if sigma > 0 else 0.0)
        counts[a] += 1; sums[a] += r; arms.append(a)
    best = max(mu); gap = max(mu) - min(mu)
    return sum(best - mu[a] for a in arms) / (T * gap), arms


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for k, mus in ((3, [75, 50, 25]), (5, [75, 62.5, 50, 37.5, 25])):
        for sigma in (0.0, 6.25, 12.5):
            for scale_name, sc, sh in (("High+", 1, 0), ("High-", 1, -100), ("Low+", 0.01, 0)):
                mu = [m * sc + sh * sc for m in mus]
                res = {}
                for regime in ("helpful", "misleading", "uninformative"):
                    regs = []
                    for rep in range(400):
                        order = list(rng.permutation(k))               # display order is irrelevant to the policy; arm ids carry the means
                        true_rank = list(np.argsort(-np.array(mu)))   # arm ids best -> worst
                        if regime == "helpful": s0 = true_rank
                        elif regime == "misleading": s0 = list(true_rank); s0[0], s0[-1] = s0[-1], s0[0]
                        else: s0 = None
                        regs.append(run_episode(mu, s0, sigma * sc, policy="oracle", rng=rng)[0])
                    res[regime] = round(float(np.mean(regs)), 3)
                ucb = round(float(np.mean([run_episode(mu, None, sigma * sc, policy="ucb", rng=rng)[0] for _ in range(400)])), 3)
                print(f"k={k} sigma={sigma:5} {scale_name:6} oracle helpful {res['helpful']:.3f} misleading {res['misleading']:.3f} uninformative {res['uninformative']:.3f} | UCB1 {ucb:.3f}")
