"""Agentic-DPO comparator (arXiv 2607.10601, released trainer src/training/agentic_dpo_trainer.py), one refresh round.
Loss per the release: L = sft_weight * L_SFT(expert) + dpo_weight * mean_i[-w_i log sigmoid(beta_eff_i * ((lp+_i - lp-_i) - 0))],
beta_eff = beta / max(|a+|, |a-|)^alpha, beta = 0.008, alpha = 0.5, sft_weight = 0.5, zero reference (Delta_ref = 0), log-probs summed
over the response tokens, w_i = 0 for states without a negative or with identical expert / negative tokens. 75 optimizer steps at
effective batch 16, lr 2e-4 (release values). Continues the current student's LoRA adapter (deviation: the release merges each
round and adds a fresh r64 adapter; here the r16 adapter of the SFT warm-up is trained further so vLLM can serve it unchanged).
SFT targets / chat template / response suffix exactly as sbadapt_train.py.
Usage: python sbadapt_adpo_train.py --model M --adapter_in A --pairs pairs_rR.jsonl --out A_next --seed S"""
import argparse, json, os, random, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--adapter_in", required=True); ap.add_argument("--pairs", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=75); ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--micro", type=int, default=2); ap.add_argument("--accum", type=int, default=8); ap.add_argument("--max_len", type=int, default=1536)
ap.add_argument("--beta", type=float, default=0.008); ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--sft_weight", type=float, default=0.5); ap.add_argument("--dpo_weight", type=float, default=1.0)
ap.add_argument("--cpu_smoke", type=int, default=0)
a = ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.model)
DEV = "cpu" if a.cpu_smoke else "cuda"
if a.cpu_smoke:
    model = AutoModelForCausalLM.from_pretrained(a.model)
else:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = PeftModel.from_pretrained(model, a.adapter_in, is_trainable=True)
pairs = [json.loads(l) for l in open(a.pairs, encoding="utf-8")]
ex = []
for r in pairs:
    p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True, enable_thinking=True, tokenize=False)
    pi = tok.encode(p, add_special_tokens=False)
    ti = tok.encode(r["target"] + "<|im_end|>\n", add_special_tokens=False)
    ni = tok.encode(r["negative"] + "<|im_end|>\n", add_special_tokens=False) if r.get("negative") else None
    if len(pi) + max(len(ti), len(ni or [])) > a.max_len: continue
    w = 0.0 if ni is None or ni == ti else 1.0
    ex.append((pi, ti, ni if w else None, w))
print(f"pairs {len(pairs)} usable {len(ex)} with negative {sum(e[3] > 0 for e in ex)}", flush=True)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / a.steps))
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def resp_logps(seqs):
    """seqs: list of (prompt_ids, resp_ids). Returns per-sequence summed log-prob and per-token log-probs of the response."""
    L = max(len(p) + len(r) for p, r in seqs)
    ids = torch.full((len(seqs), L), PAD); att = torch.zeros((len(seqs), L), dtype=torch.long)
    for b, (p, r) in enumerate(seqs):
        ids[b, :len(p) + len(r)] = torch.tensor(p + r); att[b, :len(p) + len(r)] = 1
    logits = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).logits
    sums, toks = [], []
    for b, (p, r) in enumerate(seqs):
        lg = logits[b, len(p) - 1:len(p) + len(r) - 1].float()
        lp = torch.log_softmax(lg, -1).gather(-1, torch.tensor(r, device=lg.device).unsqueeze(-1)).squeeze(-1)
        sums.append(lp.sum()); toks.append(lp)
    return torch.stack(sums), toks


model.train(); order = []; t0 = time.time(); log = []
steps = a.steps if not a.cpu_smoke else a.cpu_smoke
for step in range(steps):
    tot = {"loss": 0.0, "sft": 0.0, "dpo": 0.0, "n_dpo": 0, "margin": 0.0}
    for _ in range(a.accum):
        if not order: order = list(range(len(ex))); random.shuffle(order)
        batch = [ex[order.pop()] for _ in range(min(a.micro, len(order)))]
        lp_e, tok_e = resp_logps([(pi, ti) for pi, ti, _, _ in batch])
        sft = -torch.cat(tok_e).mean()
        loss = a.sft_weight * sft
        dj = [j for j, e in enumerate(batch) if e[3] > 0]
        if dj and a.dpo_weight > 0:
            lp_n, _ = resp_logps([(batch[j][0], batch[j][2]) for j in dj])
            lens = torch.tensor([float(max(len(batch[j][1]), len(batch[j][2]))) for j in dj], device=lp_n.device)
            beta_eff = a.beta / lens.clamp(min=1) ** a.alpha
            margin = beta_eff * (lp_e[dj] - lp_n)
            dpo = -torch.nn.functional.logsigmoid(margin).sum() / len(batch)     # mean over the batch, w = 0 rows contribute 0
            loss = loss + a.dpo_weight * dpo
            tot["dpo"] += dpo.item() / a.accum; tot["n_dpo"] += len(dj); tot["margin"] += (lp_e[dj] - lp_n).sum().item()
        (loss / a.accum).backward(); tot["loss"] += loss.item() / a.accum; tot["sft"] += sft.item() / a.accum
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    tot["margin"] = tot["margin"] / max(tot["n_dpo"], 1); log.append({"step": step, **tot, "s": time.time() - t0})
    if step % 10 == 0 or step == steps - 1:
        print(f"step {step}/{steps} loss {tot['loss']:.4f} sft {tot['sft']:.4f} dpo {tot['dpo']:.4f} n_dpo {tot['n_dpo']} "
              f"logp_margin {tot['margin']:.2f} {time.time() - t0:.0f}s", flush=True)
model.save_pretrained(a.out); tok.save_pretrained(a.out)
json.dump({"args": vars(a), "n_pairs": len(pairs), "n_usable": len(ex), "log": log}, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)
print("TRAIN_DONE", a.out, f"{time.time() - t0:.0f}s", flush=True)
