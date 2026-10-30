# Per-cell results

Metric: normalized cumulative regret, sum over turns of (75 - mean of the chosen arm) / 500, so 0 is optimal and 1 is always the worst arm. Arm means are 75 / 50 / 25. Each cell is the mean over its replicates.

Replicates 0-9 were used during development. Replicates 80-99 were never run until each model was trained. The paper column is the base model in Semantic Bandits, Table 26 (10 replicates).

Semantic mean = mean over the 10 cells with meaningful labels (ordinal, world, sentiment; helpful and misleading). Alphanumeric labels carry no ordering information.

Columns:
- Base, Warning, Warning + explore: the base model with the paper's prompts.
- Label-ignoring planner: the same fine-tuning on decisions of a planner that ignores the labels.
- Planner imitation: fine-tuning on the label-reliability planner's decisions, one copy each (v7 data).
- This method: v13 data, with cost-sensitive copies and class balance. s0 / s1 are training seeds.
- Shuffled copies: the v13 copy counts shuffled within each label kind, with class balance kept.
- Agentic-DPO: the round selected on replicates 0-9 (Qwen3-8B round 2, OLMo-3-7B round 5). All rounds are in per_cell_regret.csv.

## Qwen3-8B, dev (replicates 0-9)

| Cell | Paper | Base | Warning | Warning + explore | Label-ignoring planner | Planner imitation | This method s0 | This method s1 | Shuffled copies s1 | Agentic-DPO (round 2) |
|---|---|---|---|---|---|---|---|---|---|---|
| bandit / ordinal / helpful | 0.01 | 0.000 | 0.045 | 0.255 | 0.150 | 0.050 | 0.050 | 0.050 | 0.050 | 0.050 |
| bandit / ordinal / misleading | 0.95 | 0.960 | 0.960 | 0.450 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| bandit / sentiment / helpful | 0.33 | 0.250 | 0.255 | 0.245 | 0.150 | 0.060 | 0.050 | 0.050 | 0.050 | 0.050 |
| bandit / sentiment / misleading | 0.60 | 0.490 | 0.570 | 0.235 | 0.150 | 0.150 | 0.185 | 0.150 | 0.150 | 0.150 |
| bandit / alphanumeric / none | 0.40 | 0.490 | 0.455 | 0.275 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| farm / ordinal / helpful | 0.00 | 0.000 | 0.050 | 0.280 | 0.150 | 0.050 | 0.050 | 0.050 | 0.050 | 0.050 |
| farm / ordinal / misleading | 1.00 | 0.960 | 1.000 | 0.520 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| farm / world / helpful | 0.01 | 0.050 | 0.050 | 0.385 | 0.150 | 0.050 | 0.050 | 0.050 | 0.050 | 0.050 |
| farm / world / misleading | 0.78 | 0.850 | 0.850 | 0.495 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| farm / sentiment / helpful | 0.20 | 0.050 | 0.200 | 0.350 | 0.150 | 0.055 | 0.050 | 0.050 | 0.050 | 0.050 |
| farm / sentiment / misleading | 0.70 | 0.610 | 0.600 | 0.300 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| farm / alphanumeric / none | 0.34 | 0.420 | 0.200 | 0.400 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| **Semantic mean** | 0.458 | **0.422** | **0.458** | **0.352** | **0.150** | **0.101** | **0.103** | **0.100** | **0.100** | **0.100** |

## Qwen3-8B, fresh (replicates 80-99)

| Cell | Paper | Base | Warning | Warning + explore | Label-ignoring planner | Planner imitation | This method s0 | This method s1 | Shuffled copies s1 | Agentic-DPO (round 2) |
|---|---|---|---|---|---|---|---|---|---|---|
| bandit / ordinal / helpful | 0.01 | 0.003 | 0.020 | 0.208 | 0.150 | 0.050 | 0.050 | 0.050 | 0.050 | 0.050 |
| bandit / ordinal / misleading | 0.95 | 0.910 | 0.890 | 0.362 | 0.150 | 0.150 | 0.175 | 0.150 | 0.150 | 0.150 |
| bandit / sentiment / helpful | 0.33 | 0.223 | 0.182 | 0.270 | 0.150 | 0.077 | 0.050 | 0.058 | 0.050 | 0.050 |
| bandit / sentiment / misleading | 0.60 | 0.728 | 0.810 | 0.280 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| bandit / alphanumeric / none | 0.40 | 0.335 | 0.325 | 0.302 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| farm / ordinal / helpful | 0.00 | 0.000 | 0.068 | 0.190 | 0.150 | 0.050 | 0.050 | 0.050 | 0.050 | 0.050 |
| farm / ordinal / misleading | 1.00 | 1.000 | 0.925 | 0.360 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| farm / world / helpful | 0.01 | 0.025 | 0.000 | 0.302 | 0.150 | 0.060 | 0.050 | 0.050 | 0.060 | 0.050 |
| farm / world / misleading | 0.78 | 0.825 | 0.850 | 0.220 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| farm / sentiment / helpful | 0.20 | 0.352 | 0.453 | 0.345 | 0.150 | 0.055 | 0.050 | 0.050 | 0.050 | 0.050 |
| farm / sentiment / misleading | 0.70 | 0.812 | 0.780 | 0.310 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| farm / alphanumeric / none | 0.34 | 0.438 | 0.450 | 0.355 | 0.150 | 0.143 | 0.147 | 0.150 | 0.150 | 0.150 |
| **Semantic mean** | 0.458 | **0.488** | **0.498** | **0.285** | **0.150** | **0.104** | **0.102** | **0.101** | **0.101** | **0.100** |

## OLMo-3-7B, dev (replicates 0-9)

| Cell | Base | Warning | Warning + explore | Label-ignoring planner | Planner imitation | This method s0 | This method s1 | Shuffled copies s1 | Agentic-DPO (round 5) |
|---|---|---|---|---|---|---|---|---|---|
| bandit / ordinal / helpful | 0.150 | 0.300 | 0.490 | 0.150 | 0.050 | 0.050 | 0.050 | 0.050 | 0.050 |
| bandit / ordinal / misleading | 0.750 | 0.750 | 0.465 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| bandit / sentiment / helpful | 0.200 | 0.260 | 0.250 | 0.150 | 0.060 | 0.050 | 0.050 | 0.050 | 0.050 |
| bandit / sentiment / misleading | 0.700 | 0.515 | 0.475 | 0.150 | 0.150 | 0.150 | 0.150 | 0.190 | 0.150 |
| bandit / alphanumeric / none | 0.500 | 0.600 | 0.360 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| farm / ordinal / helpful | 0.100 | 0.100 | 0.365 | 0.150 | 0.060 | 0.050 | 0.050 | 0.050 | 0.050 |
| farm / ordinal / misleading | 0.800 | 0.800 | 0.605 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| farm / world / helpful | 0.200 | 0.300 | 0.250 | 0.150 | 0.060 | 0.065 | 0.060 | 0.065 | 0.065 |
| farm / world / misleading | 0.550 | 0.510 | 0.395 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| farm / sentiment / helpful | 0.050 | 0.150 | 0.125 | 0.150 | 0.060 | 0.050 | 0.050 | 0.050 | 0.050 |
| farm / sentiment / misleading | 0.995 | 0.950 | 0.785 | 0.145 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| farm / alphanumeric / none | 0.570 | 0.655 | 0.570 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| **Semantic mean** | **0.450** | **0.463** | **0.420** | **0.149** | **0.104** | **0.101** | **0.101** | **0.105** | **0.101** |

## OLMo-3-7B, fresh (replicates 80-99)

| Cell | Base | Warning | Warning + explore | Label-ignoring planner | Planner imitation | This method s0 | This method s1 | Shuffled copies s1 | Agentic-DPO (round 5) |
|---|---|---|---|---|---|---|---|---|---|
| bandit / ordinal / helpful | 0.230 | 0.340 | 0.485 | 0.150 | 0.050 | 0.050 | 0.050 | 0.050 | 0.050 |
| bandit / ordinal / misleading | 0.750 | 0.728 | 0.503 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| bandit / sentiment / helpful | 0.200 | 0.300 | 0.200 | 0.150 | 0.058 | 0.050 | 0.050 | 0.058 | 0.050 |
| bandit / sentiment / misleading | 0.758 | 0.775 | 0.600 | 0.143 | 0.150 | 0.150 | 0.150 | 0.147 | 0.150 |
| bandit / alphanumeric / none | 0.323 | 0.385 | 0.395 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| farm / ordinal / helpful | 0.150 | 0.175 | 0.420 | 0.150 | 0.055 | 0.050 | 0.050 | 0.050 | 0.050 |
| farm / ordinal / misleading | 0.950 | 0.925 | 0.635 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| farm / world / helpful | 0.250 | 0.350 | 0.297 | 0.150 | 0.075 | 0.080 | 0.077 | 0.080 | 0.068 |
| farm / world / misleading | 0.700 | 0.625 | 0.547 | 0.150 | 0.173 | 0.158 | 0.150 | 0.250 | 0.150 |
| farm / sentiment / helpful | 0.128 | 0.177 | 0.185 | 0.147 | 0.050 | 0.050 | 0.053 | 0.053 | 0.050 |
| farm / sentiment / misleading | 0.915 | 0.850 | 0.812 | 0.150 | 0.150 | 0.150 | 0.150 | 0.147 | 0.150 |
| farm / alphanumeric / none | 0.425 | 0.450 | 0.335 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 | 0.150 |
| **Semantic mean** | **0.503** | **0.524** | **0.469** | **0.149** | **0.106** | **0.104** | **0.103** | **0.114** | **0.102** |

## Qwen3-14B, dev (replicates 0-9)

| Cell | Paper | Base | Warning | Warning + explore | Label-ignoring planner | This method |
|---|---|---|---|---|---|---|
| bandit / ordinal / helpful | 0.07 | 0.225 | 0.250 | 0.275 | 0.150 | 0.050 |
| bandit / ordinal / misleading | 0.49 | 0.805 | 0.760 | 0.345 | 0.150 | 0.150 |
| bandit / sentiment / helpful | 0.13 | 0.300 | 0.235 | 0.235 | 0.150 | 0.050 |
| bandit / sentiment / misleading | 0.24 | 0.485 | 0.455 | 0.260 | 0.150 | 0.150 |
| bandit / alphanumeric / none | 0.17 | 0.365 | 0.440 | 0.265 | 0.150 | 0.165 |
| farm / ordinal / helpful | 0.04 | 0.040 | 0.100 | 0.275 | 0.150 | 0.050 |
| farm / ordinal / misleading | 0.71 | 0.955 | 0.950 | 0.420 | 0.150 | 0.150 |
| farm / world / helpful | 0.10 | 0.000 | 0.150 | 0.175 | 0.150 | 0.060 |
| farm / world / misleading | 0.32 | 0.580 | 0.750 | 0.235 | 0.150 | 0.150 |
| farm / sentiment / helpful | 0.12 | 0.305 | 0.260 | 0.255 | 0.150 | 0.050 |
| farm / sentiment / misleading | 0.27 | 0.575 | 0.625 | 0.365 | 0.150 | 0.150 |
| farm / alphanumeric / none | 0.15 | 0.390 | 0.355 | 0.250 | 0.150 | 0.250 |
| **Semantic mean** | 0.249 | **0.427** | **0.454** | **0.284** | **0.150** | **0.101** |

## Qwen3-14B, fresh (replicates 80-99)

| Cell | Paper | Base | Warning | Warning + explore | Label-ignoring planner | This method |
|---|---|---|---|---|---|---|
| bandit / ordinal / helpful | 0.07 | 0.198 | 0.300 | 0.255 | 0.150 | 0.050 |
| bandit / ordinal / misleading | 0.49 | 0.805 | 0.738 | 0.280 | 0.150 | 0.150 |
| bandit / sentiment / helpful | 0.13 | 0.270 | 0.310 | 0.225 | 0.150 | 0.050 |
| bandit / sentiment / misleading | 0.24 | 0.738 | 0.758 | 0.260 | 0.150 | 0.152 |
| bandit / alphanumeric / none | 0.17 | 0.365 | 0.320 | 0.257 | 0.150 | 0.185 |
| farm / ordinal / helpful | 0.04 | 0.077 | 0.200 | 0.233 | 0.150 | 0.050 |
| farm / ordinal / misleading | 0.71 | 0.972 | 0.850 | 0.302 | 0.150 | 0.150 |
| farm / world / helpful | 0.10 | 0.160 | 0.275 | 0.160 | 0.150 | 0.053 |
| farm / world / misleading | 0.32 | 0.805 | 0.800 | 0.225 | 0.150 | 0.150 |
| farm / sentiment / helpful | 0.12 | 0.318 | 0.325 | 0.220 | 0.150 | 0.050 |
| farm / sentiment / misleading | 0.27 | 0.758 | 0.785 | 0.265 | 0.150 | 0.150 |
| farm / alphanumeric / none | 0.15 | 0.330 | 0.438 | 0.255 | 0.150 | 0.200 |
| **Semantic mean** | 0.249 | **0.510** | **0.534** | **0.242** | **0.150** | **0.100** |

## Agentic-DPO by round (seed 1), semantic mean

| Model | Replicates | Round 0 | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | dev (replicates 0-9) | 0.154 | 0.153 | 0.100 | 0.129 | 0.100 | 0.100 |
| Qwen3-8B | fresh (replicates 80-99) | 0.124 | 0.150 | 0.100 | 0.120 | 0.100 | 0.100 |
| OLMo-3-7B | dev (replicates 0-9) | 0.105 | 0.106 | 0.117 | 0.105 | 0.116 | 0.101 |
| OLMo-3-7B | fresh (replicates 80-99) | 0.107 | 0.107 | 0.120 | 0.105 | 0.116 | 0.102 |

## Reward scale and noise (Qwen3-8B, planner imitation adapter, semantic mean)

These runs use the plain-imitation adapter (v7 data, seed 0), not the final recipe.

| Condition | Replicates | Base | Warning | Warning + explore | Label-ignoring planner | Planner imitation |
|---|---|---|---|---|---|---|
| low reward scale (L+), no noise | dev (replicates 0-9) | 0.411 | 0.454 | 0.333 | 0.150 | 0.106 |
| low reward scale (L+), no noise | fresh (replicates 80-99) | 0.467 | 0.486 | 0.277 | 0.150 | 0.107 |
| high scale, negative values (H-), no noise | dev (replicates 0-9) | 0.249 | 0.265 | 0.364 | 0.150 | 0.101 |
| high scale, negative values (H-), no noise | fresh (replicates 80-99) | 0.208 | 0.201 | 0.314 | 0.150 | 0.103 |
| high scale (H+), sigma 6.25 | dev (replicates 0-9) | 0.421 | 0.460 | 0.347 | 0.150 | 0.108 |
| high scale (H+), sigma 6.25 | fresh (replicates 80-99) | 0.487 | 0.505 | 0.270 | 0.150 | 0.107 |
| high scale (H+), sigma 12.5 | dev (replicates 0-9) | 0.413 | 0.442 | 0.327 | 0.158 | 0.125 |
| high scale (H+), sigma 12.5 | fresh (replicates 80-99) | 0.475 | 0.487 | 0.271 | 0.160 | 0.147 |

## Files

- `per_cell_regret.csv`: every arm, condition, replicate split and cell.
- `per_replicate_regret.csv`: the regret of every single episode, from which all tables here are computed.
- `judge_*.json`: the pass/fail checks of each trained adapter (see the main README).
