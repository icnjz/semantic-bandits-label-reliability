"""Five-arm mechanism probe: on every reachable state where the exact DP teacher and the CONTRA script choose
different actions (k5_policy_sets.json, 22 states), render the state in the official prompt (sbadapt_eval_ext.py, k = 5) and record
which action the model takes. Each state is shown with three label families - bandit / ordinal, bandit / sentiment, farm / world -
and 4 display orders (seeded), tried arms pulled once with the 5-arm means of their true rank. Output: probes.json with every
decision and the share of decisions matching the DP action, the CONTRA action, or neither.
Usage: python sbadapt_probe_k5.py --backend vllm --model M [--lora DIR] --out OUTDIR"""
import argparse, json, os, sys, time
import numpy as np
import sbadapt_eval_ext as E

COMBOS = [("bandit", "ordinal"), ("bandit", "sentiment"), ("farm", "world")]
N_ORDERS = 4


def labels_by_position(domain, nom, rng):
    if nom == "ordinal":
        return [E.ORDINAL5[a] for a in E.ARMS[5]]
    if nom == "world":
        return [E.WORLD_FARM5[a] for a in E.ARMS[5]]
    return [E.SENT_POOLS[p][int(rng.integers(20))].capitalize() for p in E.SENT5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="vllm")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--states", default="k5_policy_sets.json")
    ap.add_argument("--max_model_len", type=int, default=2048)
    a = ap.parse_args()
    E.K = 5
    E.MEANS.clear(); E.MEANS.update(E.scale_means(5, "H+"))
    os.makedirs(a.out, exist_ok=True)
    states = json.load(open(a.states))["disagreement_states"]
    t0 = time.time()
    backend = E.VLLMBackend(a.model, a.max_model_len, a.lora)
    items = []
    for si, st in enumerate(states):
        true = st["example_true"]
        arm_at = {pos: E.ARMS[5][4 - true[pos]] for pos in range(5)}          # semantic position -> arm key (A = best)
        pos_of = {v: k for k, v in arm_at.items()}
        for domain, nom in COMBOS:
            for o in range(N_ORDERS):
                rng = np.random.default_rng(E.seed_of("probe", si, domain, nom, o) )
                labs = labels_by_position(domain, nom, rng)
                cfg = {"order": [str(x) for x in rng.permutation(E.ARMS[5])], "labels": {arm_at[p]: labs[p] for p in range(5)}}
                counts = {x: 0 for x in E.ARMS[5]}; sums = {x: 0.0 for x in E.ARMS[5]}
                for p in st["order_best_to_worst"]:
                    counts[arm_at[p]] = 1; sums[arm_at[p]] = E.MEANS[arm_at[p]]
                prompt = E.render_prompt(domain, cfg, counts, sums, st["step"])
                items.append({"si": si, "domain": domain, "nom": nom, "o": o, "cfg": cfg, "pos_of": pos_of, "st": st, "prompt": prompt,
                              "prompt_ids": E.chat_ids(backend.tok, prompt), "key": ("probe", si, domain, nom, o)})
    todo, attempt, recs = items, 0, []
    while todo and attempt < 3:
        E.run_decisions(backend, todo, attempt)
        nxt = []
        for it in todo:
            arm, how = E.parse_action(it["full_text"], it["cfg"]["labels"], E.ACTION_SUFFIX[it["domain"]])
            if arm is None and attempt < 2:
                nxt.append(it); continue
            pos = it["pos_of"].get(arm) if arm else None
            dp, ct = it["st"]["dp"][1], it["st"]["contra"][1]
            recs.append({"state": it["si"], "domain": it["domain"], "nom": it["nom"], "order_idx": it["o"], "step": it["st"]["step"],
                         "observed_best_to_worst": it["st"]["order_best_to_worst"], "dp": it["st"]["dp"], "contra": it["st"]["contra"],
                         "chosen_position": pos, "match": "dp" if pos == dp else ("contra" if pos == ct else ("none" if pos is not None else "parse_fail")),
                         "parse": how, "raw": it["full_text"]})
        todo, attempt = nxt, attempt + 1
    summ = {}
    for key in [("all",)] + [(d, n) for d, n in COMBOS]:
        rs = [r for r in recs if key == ("all",) or (r["domain"], r["nom"]) == key]
        summ["/".join(key)] = {m: round(sum(r["match"] == m for r in rs) / max(1, len(rs)), 4) for m in ("dp", "contra", "none", "parse_fail")} | {"n": len(rs)}
    json.dump({"summary": summ, "decisions": recs, "args": vars(a), "wall_s": time.time() - t0}, open(os.path.join(a.out, "probes.json"), "w"), indent=1)
    print(json.dumps(summ))
    open(os.path.join(a.out, "DONE"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


if __name__ == "__main__":
    main()
