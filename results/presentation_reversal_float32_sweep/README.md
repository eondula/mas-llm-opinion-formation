# Float32 residual sweep

Changed pair only, both directions, all 28 blocks. No head combinations.
Run on the GPU machine from the repository root:

```bash
python results/presentation_reversal_float32_sweep/run.py --device cuda
```

Auto prefers CUDA then MPS with no CPU fallback. Model weights are converted to
float32, TF32 is disabled on CUDA, and scoring uses float32 log-softmax and sum.
Donors and both baselines are recomputed. Rendered prompts must match the saved
initialization exactly; numerical agreement with bfloat16 is not a replay gate.
No explanations or conversations are generated. The saved historical strategy
is kept separate from the recomputed diagnostic scored choice.

Each block replaces only the final prompt token after the block. Independent
endorsement calls use no shared caches and remove hooks each time. All 28
observations are checked, including independence from candidate suffixes.
Self replacements must reproduce both scores exactly. Total: 56 donor
replacements and 56 self controls. No original run records are modified.

Requires saved behavioral/reversed sources and
`results/presentation_reversal_patching/patches.csv`. Shared helpers are imported
from the tracked first pilot. Outputs include source hashes, float32 donor vectors,
float32_baselines.csv, observation_scores.csv, patches.csv, precision_comparison.csv,
precision_sweep.png and results_summary.md. The comparison contains each layer's
margin change, strategy and absolute-effect rank within precision and direction.
The report highlights block 15, strongest layers and transfers. Signed effects
in both directions determine whether the previously observed asymmetry persists.
Rank is descriptive, not a significance test. The native reference includes
bfloat16 scoring as well as forward execution; configurations are labeled.

Regenerate the PNG and comparison from saved sweeps without inference:

```bash
python results/presentation_reversal_float32_sweep/run.py --plot-only
```

Transfer from the Mac (replace GPU_HOST):

```bash
rsync -avP \
  win.ondulae@GPU_HOST:/home/win.ondulae/mas-llm-opinion-formation/results/presentation_reversal_float32_sweep/ \
  /Users/ondulae/Documents/Research/LLM/mas-llm-opinion-formation/results/presentation_reversal_float32_sweep/
```
