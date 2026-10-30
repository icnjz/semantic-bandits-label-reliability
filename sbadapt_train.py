"""QLoRA supervised fine-tuning on planner decisions (label-reliability planner, or the label-ignoring control).
The loss is on the assistant target only.
Usage: python sbadapt_train.py --model Qwen/Qwen3-8B --data v13/train_adaptive_v13.jsonl --out lora_v13 --seed 0"""
import argparse, json, math, os, random, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--micro", type=int, default=4); ap.add_argument("--accum", type=int, default=4); ap.add_argument("--max_len", type=int, default=1024)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--cpu_smoke", type=int, default=0)
a = ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.model)
DEV = "cpu" if a.cpu_smoke else "cuda"
if a.cpu_smoke:   # local CPU smoke test of the loop / masking only (tiny model, no quantisation)
    model = AutoModelForCausalLM.from_pretrained(a.model)
else:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
ex = []
for r in rows:
    p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True, enable_thinking=True, tokenize=False)
    pi = tok.encode(p, add_special_tokens=False); ti = tok.encode(r["target"] + "<|im_end|>\n", add_special_tokens=False)
    ids = (pi + ti)[: a.max_len]; lab = ([-100] * len(pi) + ti)[: a.max_len]
    ex.append((ids, lab))
steps = math.ceil(len(ex) * a.epochs / (a.micro * a.accum)) if not a.cpu_smoke else a.cpu_smoke
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
model.train(); order = []; t0 = time.time(); log = []
for step in range(steps):
    tot = 0.0
    for _ in range(a.accum):
        if not order: order = list(range(len(ex))); random.shuffle(order)
        batch = [ex[order.pop()] for _ in range(min(a.micro, len(order)))]
        L = max(len(i) for i, _ in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id or 151643); lab = torch.full((len(batch), L), -100); att = torch.zeros((len(batch), L), dtype=torch.long)
        for b, (i, l) in enumerate(batch):
            ids[b, :len(i)] = torch.tensor(i); lab[b, :len(l)] = torch.tensor(l); att[b, :len(i)] = 1
        out = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV), labels=lab.to(DEV))
        (out.loss / a.accum).backward(); tot += float(out.loss) / a.accum
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    log.append({"step": step, "loss": tot, "s": time.time() - t0})
    if step % 10 == 0 or step == steps - 1: print(f"step {step}/{steps} loss {tot:.4f} {time.time() - t0:.0f}s", flush=True)
model.save_pretrained(a.out); tok.save_pretrained(a.out)
json.dump({"args": vars(a), "n_examples": len(ex), "steps": steps, "log": log}, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)
print("TRAIN_DONE", a.out, f"{time.time() - t0:.0f}s", flush=True)
