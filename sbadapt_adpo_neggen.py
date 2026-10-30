"""Agentic-DPO comparator (Chen & Yuille, arXiv 2607.10601; released code github.com/Schuture/Agentic-DPO), negative refresh for one round.
Expert states = the v7 teacher decisions (train_adaptive_v7.jsonl, one line per decision). Per round: a slice of 1,200 states
(= 75 steps x effective batch 16, the release's strict round budget) drawn without replacement with seed 100 * seed + round; K = 4
one-step samples from the current student (temperature 0.7, max 256 tokens, as in the release); candidates whose parsed action is
the expert's arm or that do not parse to one of the listed actions are dropped; the highest-log-prob remaining one is the negative;
states with none are kept with no negative (SFT term only, weight 0 for DPO, as in the release).
Usage: python sbadapt_adpo_neggen.py --model M --lora ADAPTER --data train_adaptive_v7.jsonl --round R --seed S --out pairs_rR.jsonl"""
import argparse, json, re
import numpy as np


def labels_of(prompt):
    return [re.sub(r"^.*?Name:\s*", "", m).strip() for m in re.findall(r"^ - (.+?) -- This action has been selected", prompt, flags=re.M)]


def norm(x):
    x = x.strip().strip("`*\"' \t\n").rstrip(".").strip("`*\"' \t\n")
    return re.sub(r"\s+", " ", x).lower()


def action_of(text, labels):
    tags = re.findall(r"<action>(.*?)</action>", text.split("</think>")[-1], flags=re.S | re.I)
    if not tags: return None
    x = norm(re.sub(r"^.*?name\s*:\s*", "", norm(tags[-1])))
    hits = [l for l in labels if norm(l) == x]
    return hits[0] if len(hits) == 1 else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--lora", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--round", type=int, required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n_visits", type=int, default=1200); ap.add_argument("--K", type=int, default=4)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    idx = np.random.default_rng(100 * a.seed + a.round).choice(len(rows), size=min(a.n_visits, len(rows)), replace=False)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=2048, enable_prefix_caching=True, gpu_memory_utilization=0.90, seed=0,
              enable_lora=True, max_lora_rank=16, max_loras=1)
    tok = llm.get_tokenizer()
    ids = [tok.encode(tok.apply_chat_template([{"role": "user", "content": rows[i]["prompt"]}], add_generation_prompt=True,
                                              enable_thinking=True, tokenize=False), add_special_tokens=False) for i in idx]
    # Stop at the end of the assistant turn exactly as the evaluation harness does (<|im_end|> and the tokenizer's eos). Without this,
    # models whose eos is not <|im_end|> (OLMo-3) keep generating invented later turns, and the last <action> tag is a fake action.
    stop = [t for t in dict.fromkeys([tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]) if isinstance(t, int) and t >= 0]
    sps = [SamplingParams(n=a.K, temperature=0.7, max_tokens=256, logprobs=0, seed=int(1000003 * a.round + j), skip_special_tokens=True,
                          stop_token_ids=stop) for j in range(len(idx))]
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ids], sps, use_tqdm=False, lora_request=LoRARequest("student", 1, a.lora))
    n_neg = n_par = n_same = n_runon = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for i, o in zip(idx, outs):
            r = rows[int(i)]; labs = labels_of(r["prompt"]); expert = action_of(r["target"], labs); assert expert is not None, r["target"]
            cands = []
            for c in o.outputs:
                if c.text.count("<action>") > 1 or "\nassistant" in c.text or "\nuser" in c.text: n_runon += 1; continue
                act = action_of(c.text, labs)
                if act is None: continue
                n_par += 1
                if act == expert: n_same += 1; continue
                lp = c.cumulative_logprob
                if lp is None: lp = sum(d[t].logprob for t, d in zip(c.token_ids, c.logprobs))
                cands.append((lp, c.text, act))
            neg = max(cands, key=lambda x: x[0]) if cands else None
            n_neg += neg is not None
            f.write(json.dumps({"i": int(i), "prompt": r["prompt"], "target": r["target"], "expert_action": expert,
                                "negative": neg[1] if neg else None, "negative_action": neg[2] if neg else None,
                                "negative_logprob": neg[0] if neg else None, "meta": r["meta"]}, ensure_ascii=False) + "\n")
    print(f"NEGGEN round {a.round}: states {len(idx)}, parsed candidates {n_par}/{len(idx) * a.K}, same-as-expert {n_same}, "
          f"states with a negative {n_neg}, run-on candidates dropped {n_runon} (stop ids {stop})", flush=True)
    assert n_runon <= 0.01 * len(idx) * a.K, f"too many run-on candidates ({n_runon}); stop tokens not working"
