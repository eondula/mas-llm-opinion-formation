# Float32 generalization across initialization pairs

Runs final prompt token replacements after blocks 12–15, in both directions, using Qwen2.5-1.5B-Instruct and scored choice. No individual head searches, generated responses, or conversation continuations.

Six distinct pairs cover Sensor and Storage with no persona, S1 priority, and S2 priority. The Sensor/no-persona pair is the original discovery pair. The two Storage priority pairs change saved choice from S2 to S1. The other three additional pairs retain their saved choices. These labels concern saved native scores; float32 baselines are recomputed and may differ. Sensor/S1 priority was also used as a control in previous pilots; these are additional to the discovery pair, not all previously unexamined cases.

Selection matches scenario, ordered checkpoints, schedule, agent, priorities, natural initialization and scored procedure. Deduplication uses the pair of exact saved rendered decision prompts, excluding 42 redundant matched pairs across schedules and assignments. Selected sources are pinned in config.json. Before inference, the runner reconstructs histories using project utilities, verifies pair hashes and uniqueness, and checks conditions match apart from presentation. Source hashes are retained in inputs.json. All selected representatives are synchronous homogeneous Qwen pairs. Exact prompt equivalence, rather than schedule labels, defines distinct initialization inputs.

## Remote run

```bash
python results/presentation_reversal_generalization/run.py --device cuda
```

`--device auto` selects CUDA then MPS; CPU is not selected. Uses float32 model execution, logits, log-softmax and accumulation, with CUDA TF32 disabled. Weights are converted checkpoint weights. Every candidate uses a fresh forward call without a cache. Hooks are removed after each call. Donors and baselines are recomputed; observation and candidate-suffix invariance checks must be exact. Each self replacement must reproduce both scores exactly. This makes 96 replacement rows: six pairs × two directions × four blocks × donor/self. The original behavioral files remain unchanged.

```bash
python results/presentation_reversal_generalization/run.py --prepare-only
python results/presentation_reversal_generalization/run.py --plot-only
python results/presentation_reversal_generalization/test_pilot.py
```

Preparation verifies source reconstruction without loading a model. Plot-only rebuilds CSV figure data, summary and PNG from a complete saved run. No PDF files are generated.

## Outputs

- `matched_cases.csv`: source conditions and saved baseline scores/choices, discovery/additional and saved changed/unchanged labels.
- `float32_baselines.csv`: recomputed baseline endorsement scores, margins, strategies and semantic S1/S2 labels.
- `patches.csv`: both endorsement scores, recipient baseline scores, margin changes, resulting and original strategies, semantic alternative names, and self controls.
- `figure_data.csv` and `case_effects.png`: each case and both directions, shared margin-change scale and resulting strategies. All plotted values come from CSV.
- `results_summary.md`: per-case signed effects and semantic choices. Positive changes favor Sensor precision or Storage fidelity; negative changes favor Sensor coverage or Storage reduction. Same-choice pairs do not demonstrate strategy transfer merely by retaining the donor's choice.
- `observation_scores.csv`, activation tensors, `inputs.json`, `configuration_used.json`, `environment.json`, `status.json`: verification and provenance.

From your Mac (replace GPU_HOST):

```bash
rsync -avP win.ondulae@GPU_HOST:/home/win.ondulae/mas-llm-opinion-formation/results/presentation_reversal_generalization/ /Users/ondulae/Documents/Research/LLM/mas-llm-opinion-formation/results/presentation_reversal_generalization/
```

This is a small selected set of distinct prompts, not independent random replications. Joint reversal of display and candidate blocks does not separate the two manipulations. Compare changes within each recomputed baseline and retain the discovery label in analysis.
