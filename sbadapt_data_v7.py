"""Training decisions, step 4 (plain imitation data, 'v7'): v5 with one change: meaningless ('code') labels are drawn
from a broad family of random-identifier formats instead of the single 'X-1234' format, so that 'meaningless' does not look like one
format. The evaluation's 6-character lowercase alphanumerics are one member of this family. Semantic episodes (ordinal / sentiment /
world) are byte-identical to v5."""
import argparse, json, os, string, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sbadapt_data_v5 as V5
D = V5.D
LOW, UP, DIG = string.ascii_lowercase, string.ascii_uppercase, string.digits


def _rand(rng, alphabet, n): return "".join(rng.choice(list(alphabet), n))


FORMATS = [
    lambda r: rng_pick(r, ["X", "Q", "Z", "K"]) + "-" + _rand(r, DIG, 4),                      # v1 format, kept
    lambda r: _rand(r, LOW + DIG, int(r.integers(4, 9))),                                        # lowercase alnum, 4-8
    lambda r: _rand(r, UP + DIG, int(r.integers(3, 8))),                                         # uppercase alnum, 3-7
    lambda r: _rand(r, LOW + UP + DIG, int(r.integers(5, 10))),                                  # mixed-case alnum, 5-9
    lambda r: _rand(r, "0123456789abcdef", int(r.integers(6, 9))),                               # hex, 6-8
    lambda r: _rand(r, UP, 2) + "-" + _rand(r, DIG, int(r.integers(2, 5))),                      # AB-123
    lambda r: _rand(r, LOW, 3) + "_" + _rand(r, DIG, 3),                                         # abc_123
    lambda r: "id" + _rand(r, DIG, int(r.integers(3, 6))),                                       # id48213
]


def rng_pick(r, lst): return lst[int(r.integers(len(lst)))]


_V1_CODE_LABEL = D.code_label


def code_label(rng):
    """Consume exactly the v1 draws from the episode RNG (so every other episode stays byte-identical to v5), then build the new
    label from an independent generator seeded by that v1 label."""
    v1 = _V1_CODE_LABEL(rng)
    local = np.random.default_rng(int.from_bytes(v1.encode(), "little") % (2 ** 63))
    return FORMATS[int(local.integers(len(FORMATS)))](local)


D.code_label = code_label

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["adaptive", "agnostic"], required=True)
    ap.add_argument("--episodes", type=int, default=450); ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--out", required=True)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for e in range(a.episodes):
            for r in D.episode(rng, a.mode == "adaptive"):
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    print("wrote", n, "decisions to", a.out)
