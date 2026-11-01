# Learning when to trust action labels: distilling a Bayes-adaptive planner into LLM agents

extends **Semantic Bandits** (Austin, Suleman and Cheung, arXiv 2608.16707, 2026), which shows that LLM agents exploring a multi-armed bandit are biased by the semantic labels of the actions. With misleading labels, Qwen3-8B's normalized regret reaches 0.60–1.00, and the paper's warning prompts reduce this only partly.

The question here is how an agent can infer, within one episode, whether the labels help or hurt.

## Method

1. **Teacher.** A Bayes-adaptive dynamic-programming planner whose hidden variable is the reliability of the label ordering. The ordering can be correct, reversed, partly swapped or uninformative; the prior is aligned .45, reversed .15, swapped .15, uniform .25.
2. **Synthetic tasks.** Bandit tasks whose scenarios and label words do not appear in the benchmark, solved exactly by the teacher. The label pools are token-disjoint from the paper's words (`pools_v2.py`).
3. **Student.** QLoRA (r = 16) fine-tuning of the LLM on 4,500 teacher decisions. Each decision is preceded by a one-line state summary: the label ranking, which arms were tried, and whether the observed order agrees with the labels.
4. **Cost-sensitive copies.** Each decision is repeated round(1 + gap / mean gap) times, where the gap is the teacher's exact action gap (`sbadapt_data_v11.py`).
5. **Class balance.** The decision "do these labels carry meaning?" is made at the first rationale token, so the two classes are balanced at that token (`sbadapt_data_v13.py`).

## Results

Protocol: the paper's 12 Qwen3-8B bandit/farm cells, 10 turns, no variance, summarized history. Metric: mean normalized regret over the semantic cells. Results are reported on 10 development replicates and on 20 replicates never run during development.

| Model | This method | Base model | Paper's warning + explore prompt | Agent trained on a label-ignoring planner |
|---|---|---|---|---|
| Qwen3-8B (2 training seeds) | 0.100–0.103 | 0.42–0.49 | 0.29–0.35 | 0.15 |
| OLMo-3-7B (2 training seeds) | 0.101–0.104 | 0.45–0.50 | 0.42–0.47 | 0.15 |
| Qwen3-14B | 0.10 | 0.43–0.51 (paper's table: 0.25) | 0.24–0.28 | 0.15 |

Per-cell results:
- **Misleading cells:** 0.15–0.19, about the level of the label-ignoring agent.
- **Helpful cells:** 0.05–0.08.
- **Alphanumeric cells:** 0.15 at 8B / 7B.

**What each component does.** The control is the same recipe with the cost-based copy counts shuffled within each label kind. It keeps the same copy histogram and class balance (`sbadapt_data_v13shuf.py`).
- **Cost-sensitive copies** fix the contradiction action: checking the third arm when the observed order contradicts the labels. With the copies shuffled, OLMo seed 1 makes 48 such errors, against 0 with the full recipe and 70 with plain imitation.
- **Class balance** fixes invented rankings for meaningless labels. On Qwen seed 1 it brings them from 41–49% of decisions down to about 1%.

**Comparator.** Agentic-DPO (Chen and Yuille, arXiv 2607.10601) is re-implemented from the released code: `sbadapt_adpo_neggen.py` and `sbadapt_adpo_train.py`.
- **Setup.** It starts from plain imitation of the same planner decisions (seed 1) and runs 5 refresh rounds. The round is chosen on the development replicates, as in the release.
- **Final result: matches this method on both models.**
  - Qwen3-8B: round 2, semantic mean 0.100, 0 contradiction errors on the fresh replicates.
  - OLMo-3-7B: round 5, 0.102, 0 contradiction errors.
- **Unstable across rounds.** Its rounds alternate between this level and clear failures:
  - Qwen rounds 1 and 3: 0.150 and 0.120.
  - OLMo round 2: 0.120, with 72 contradiction errors on the fresh replicates.
  - OLMo round 4 fixes the contradiction action, but after trying all three arms it no longer exploits the best one.
- This method reaches the same level in a single fine-tuning run.

**Note on the ceiling.** In this noise-free setting, an agent that follows the planner's rule scores exactly 0.05 on every helpful replicate and 0.15 on every misleading one, so the semantic mean is (5 × 0.05 + 5 × 0.15) / 10 = 0.100. Methods that learn the rule therefore cannot be separated by mean regret. They are compared by how often they make the critical errors and by how stable they are across runs. `audit_harness.py` independently recomputes every metric from the raw model outputs and checks every prompt against the history so far.

**Per-cell results.** `results/PER_CELL_RESULTS.md` has every cell for every model and arm, on both replicate sets, and the Agentic-DPO rounds. It also has the reward-scale and reward-noise conditions. `results/per_cell_regret.csv` and `results/per_replicate_regret.csv` hold the same numbers down to single episodes. `make_tables.py` regenerates all three files from the raw evaluation outputs. The `results/judge_*.json` files hold the pass/fail checks of each trained adapter.

## Five arms, clothing and reward noise

The adapters above, without retraining, on the paper's other conditions. Outputs are in `results/ext/`. The pass/fail lines were fixed before the run (`judge_ext.py`, `results/ext/judge_ext.json`).

**Assumptions.** The paper does not give the five-arm means or labels (Appendix B.4). Used here: means 75 / 62.5 / 50 / 37.5 / 25, ordinal labels Highest / High / Intermediate / Low / Lowest Reward, and misleading labels that swap the labels of the best and the worst arm. The five-arm farm labels, two extra sentiment labels and the clothing context keys (Climate, Looking for) are not from the paper (`sbadapt_eval_ext.py` header).

**Five arms.** The planner is no longer the 3-arm rule. On 22 reachable states it acts differently from the contradiction rule (try the top two labels; on a contradiction, keep trying untried labels in label order). On 19 of them the planner pulls the best observed arm; on the other 3 it tries the bottom-ranked label (`k5_policy_sets.py`). Semantic mean: planner 0.075, contradiction rule 0.1375.

Semantic mean, replicates 0–9 / 80–99:

| Arm | Qwen3-8B | OLMo-3-7B |
|---|---|---|
| This method s0 | 0.203 / 0.195 | 0.233 / 0.225 |
| This method s1 | 0.222 / 0.217 | 0.199 / 0.192 |
| Contradiction-rule control s0 | 0.248 / 0.237 | – |
| Agentic-DPO | 0.206 / 0.217 | 0.229 / 0.213 |
| Label-ignoring planner | 0.277 / 0.262 | 0.274 / 0.261 |
| Base | 0.450 / 0.435 | 0.421 / 0.452 |
| Warning + explore | 0.329 / 0.345 | 0.475 / 0.423 |

- **Contradiction-rule control:** the v13 data with every five-arm decision on which the planner and the rule differ relabelled with the rule's action (`make_script_control.py`), trained the same way.
- **Agentic-DPO:** retrained with the same code and data. The round chosen on the 3-arm dev replicates is Qwen3-8B round 4 (3-arm semantic mean 0.100 / 0.100) and OLMo-3-7B round 5 (0.101 / 0.103).

Differences in semantic mean on Qwen3-8B, replicates 80–139, with 95% bootstrap intervals:
- This method s0 (0.194) vs the contradiction-rule control (0.236): −0.042 [−0.049, −0.034].
- This method s0 vs the label-ignoring planner (0.268): −0.073 [−0.083, −0.064] overall, +0.037 [+0.024, +0.052] on the misleading cells.
- This method s1 (0.217) vs Agentic-DPO (0.213): +0.005 [−0.003, +0.012].
- The failure criterion fixed before the run is met: this method stays above 0.100. The five-arm results therefore do not show that the planner's policy was learned, and the 3-arm result is read as learning the contradiction rule.

**Where it fails.** In misleading cells the best arm carries the bottom-ranked label. Over replicates 0–9 and 80–139:
- Qwen3-8B s0 pulls the best arm in 7 of 350 misleading replicates; the label-ignoring planner in 156 of 350.
- In the misleading ordinal cells it tries Highest and then High Reward in 135 of 140 replicates. Its third pull is Intermediate Reward in 86, High Reward in 35 and Lowest Reward in none.
- On the 22 states (`sbadapt_probe_k5.py`), the share of decisions matching the planner is 0.78 (s0), 0.45 (s1), 0.66 (contradiction-rule control) and 0.80 (Agentic-DPO). On the 3 states where the planner tries the bottom-ranked label, every probed model does so in at most 4 of 36 decisions.

**Clothing (3 arms).** Semantic mean, replicates 0–9 / 80–99:

| Arm | Qwen3-8B | OLMo-3-7B |
|---|---|---|
| This method s0 | 0.133 / 0.137 | 0.234 / 0.327 |
| This method s1 | 0.147 / 0.141 | 0.102 / 0.105 |
| Agentic-DPO | 0.139 / 0.139 | 0.117 / 0.126 |
| Label-ignoring planner | 0.148 / 0.158 | 0.150 / 0.150 |
| Base | 0.475 / 0.505 | 0.508 / 0.471 |
| Warning + explore | 0.338 / 0.332 | 0.460 / 0.446 |

On Qwen3-8B this method is 0.001–0.021 below the label-ignoring planner. OLMo-3-7B s0 is above it.

**Reward noise (3 arms, Qwen3-8B s0).** Replicates 0–9 / 80–99. Sigma = 6.25: 0.102 / 0.101 (label-ignoring planner 0.150 / 0.150). Sigma = 12.5: 0.122 / 0.158 (label-ignoring planner 0.158 / 0.160).

## Limitations

- With 3 arms, 10 turns and no variance, the optimal policy reduces to a simple rule: try the two best-labelled arms, then check the third if the observed order contradicts the labels.
- With five arms, this method does not learn the planner's policy where it differs from the contradiction rule. On clothing it is at most 0.021 below the label-ignoring planner (Qwen3-8B), or above it (OLMo-3-7B s0).
- **Reproduction of the base models.**
  - Re-run of the 14B base model gives a semantic mean of 0.43–0.51, not the 0.25 in the paper's table.
  - In the 8B base-model run reported here, 11 of the 12 cells are within 0.15 of the paper's table. The exception is Farm sentiment helpful: 0.05 against 0.20.
  - An earlier run of the same harness under vLLM 0.10.0 gave 10 of 12. The two exceptions were Bandit sentiment misleading (0.40 against 0.60) and Farm sentiment helpful (0.00 against 0.20).
- At 14B, the two alphanumeric cells of this method are 0.17–0.25.
- **Reward noise.** At sigma = 12.5, the margin of this method over the label-ignoring planner is 0.036 on replicates 0–9 and 0.002 on 80–99. The reward-scale conditions (H−, L+) were run only with the plain-imitation adapter (`results/PER_CELL_RESULTS.md`).
- 32B was not run.

## Files

| File | Role |
|---|---|
| `sbandit_harness.py` | Re-implementation of the Semantic Bandits harness from the paper text |
| `sbadapt_eval_mf.py`, `sbadapt_eval_sv.py` | Evaluation with vLLM and optional LoRA; the paper's prompt variants; variance / reward-scale conditions |
| `sbadapt_oracle.py`, `sbadapt_oracle_bayes.py` | Label-reliability prior and the exact Bayes-adaptive planner |
| `pools_v2.py`, `sbadapt_data*.py` | Synthetic task and decision generation; `v7` = imitation data; `v11` = cost-sensitive copies; `v13` = + class balance; `v13shuf` = shuffled-copy control |
| `sbadapt_train.py` | QLoRA supervised fine-tuning (loss on the assistant target only) |
| `sbadapt_adpo_neggen.py`, `sbadapt_adpo_train.py` | Agentic-DPO comparator: student negatives with vLLM, then a DPO loss with an SFT anchor |
| `judge_v11.py`, `judge_q14v13.py` | Pass/fail lines on development and fresh replicates |
| `teacher_agreement.py`, `exploit_count.py` | Decision-level comparison with the teacher and error decomposition |
| `adpo_gate.py` | Round selection and comparison for the Agentic-DPO comparator |
| `audit_harness.py` | Independent audit: recomputes regret from raw outputs, checks every prompt against the history, misleading labels, and train / eval label overlap |
| `make_tables.py` | Per-cell and per-replicate tables in `results/` |
| `sbadapt_eval_ext.py` | Five-arm and clothing conditions (assumptions in the header); 3-arm bandit / farm replicates are identical to `sbadapt_eval_sv.py` |
| `k5_policy_sets.py`, `k5_policy_sets.json` | Five-arm values of the planner and the contradiction rule, and the 22 states where they differ |
| `sbadapt_probe_k5.py` | Actions of a model on those 22 states |
| `make_script_control.py` | Contradiction-rule control data |
| `judge_ext.py` | Pass/fail lines of the five-arm and clothing runs |
| `data/` | Training data of the reported adapters (gzipped JSONL): v13, shuffled copies, v7 imitation, label-ignoring, contradiction-rule control |
| `results/ext/` | Outputs of the five-arm (`E5_*`), clothing (`EC_*`), reward-noise (`N6_*`, `N12_*`) and probe (`P5_*`) runs, and `judge_ext.json` |

## How to run

```bash
python sbadapt_data_v7.py --mode adaptive --out v7/train_adaptive_v7.jsonl
python sbadapt_data_v11.py --mode adaptive --out v11/train_adaptive_v11.jsonl
python sbadapt_data_v13.py v11/train_adaptive_v11.jsonl v13/train_adaptive_v13.jsonl
python sbadapt_train.py --model Qwen/Qwen3-8B --data v13/train_adaptive_v13.jsonl --out lora_v13 --seed 0
python sbadapt_eval_mf.py --backend vllm --model Qwen/Qwen3-8B --lora lora_v13 --rep_list 0-9,80-99 --out out/ADAPT
python judge_v11.py out qwen ADAPT
python make_tables.py out results
```

The control arms are produced the same way:
- **Base model:** no `--lora`.
- **Paper's prompts:** `--variant warn` / `--variant warn_explore`.
- **Label-ignoring agent:** train on `--mode agnostic` data.

`judge_v11.py` looks for the control arms in the same output directory, under the names `BASE_QWEN7`, `BASE_warn_QWEN7`, `BASE_warn_explore_QWEN7` and `AGN_V7_QWEN` (OLMo: `BASE_OLMO`, `BASE_warn_OLMO`, `BASE_warn_explore_OLMO`, `AGN_V7_OLMO`). Every script's header documents its arguments.

Five arms, clothing and reward noise (`gunzip -k data/*.gz` to train from the included data):

```bash
python sbadapt_eval_ext.py --backend vllm --model Qwen/Qwen3-8B --lora lora_v13 --k 5 --rep_list 0-9,80-139 --out out_ext/E5_V13_QWEN_S0
python sbadapt_eval_ext.py --backend vllm --model Qwen/Qwen3-8B --lora lora_v13 --cells clothing --rep_list 0-9,80-99 --out out_ext/EC_V13_QWEN_S0
python sbadapt_eval_sv.py --backend vllm --model Qwen/Qwen3-8B --lora lora_v13 --sigma 12.5 --rep_list 0-9,80-99 --out out_ext/N12_V13_QWEN_S0
python k5_policy_sets.py
python sbadapt_probe_k5.py --model Qwen/Qwen3-8B --lora lora_v13 --out out_ext/P5_V13_QWEN_S0
python make_script_control.py v13/train_adaptive_v13.jsonl train_script_k5.jsonl
python judge_ext.py out_ext
```

`judge_ext.py` reads the folders `E5_*` and `EC_*` with `V13_QWEN_S0`, `V13_QWEN_S1`, `V13_OLMO_S0`, `V13_OLMO_S1`, `AGN_QWEN`, `AGN_OLMO`, `BASE_QWEN`, `BASE_OLMO`, `WX_QWEN`, `WX_OLMO`, plus `E5_SCRIPT_QWEN_S0`, `E5_ADPO_QWEN_S1` and `P5_V13_QWEN_S0`, `P5_V13_QWEN_S1`, `P5_SCRIPT_QWEN_S0`, `P5_ADPO_QWEN_S1`. `SCRIPT` = contradiction-rule control, `ADPO` = Agentic-DPO, `AGN` = label-ignoring planner, `WX` = warning + explore.
