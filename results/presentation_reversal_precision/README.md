# Scoring precision audit

Changed case only, reversed donor → forward recipient, block 15, heads 2 and 9.
No new model experiment was executed during implementation.

The existing Runtime scores logits without casting: log-softmax and gathered
scores retain the logits dtype, followed by `.tolist()` and Python `sum`.
Python accumulation uses binary64 floats; it does not restore precision already
lost in log-softmax. The runtime dtype must be measured, not inferred from the
parameter dtype.

Run from the repository root on the remote GPU:

```bash
python results/presentation_reversal_precision/run.py --device cuda
```

CUDA or MPS is required; auto prefers CUDA. CUDA TF32 is disabled for the audit.
The model is loaded once in checkpoint-native precision; each condition's logits
are processed by BOTH original scoring and float32 log-softmax plus float32 sum.
There is only one forward pass per condition/candidate for the two reductions.
Then the model is converted to float32, donor activations and baselines recaptured,
and all conditions repeated, again with both reductions. This is float32 execution
of the checkpoint weights, not recovery of weights before checkpoint quantization.
No reduced-precision donors are used in float32 execution. Full float32 requires
more GPU memory and can be slower. No automatic precision fallback.

Four primary conditions per forward precision: baseline, head 2, head 9, joint.
Each patched condition also has self replacement; observation is checked before
patching. Caches are independent and hooks removed. Only the final prompt token
at the input to o_proj is replaced. Source is the saved confirmation inputs.json.
The model loader and tokenization reuse Runtime; the scoring calculation mirrors
Runtime while exposing both reductions on the same logits.

Outputs: precision_scores.csv (all values and dtypes), token_scores.csv,
joint_departures.csv, self_controls.csv, environment.json, donor baselines and
activations, saved_baseline_comparison.json, status.json and results_summary.md.
Joint departure is R(joint)-R(head2)-R(head9)+R(baseline), separately for full
margin and first-difference readout, within each forward/scoring configuration.
Original saved strategy is never overwritten; diagnostic choices are separate.
No generated responses or conversations are run.

```bash
python results/presentation_reversal_precision/test_precision.py
```

Transfer from local Mac (replace GPU_HOST):

```bash
rsync -avP \
  win.ondulae@GPU_HOST:/home/win.ondulae/mas-llm-opinion-formation/results/presentation_reversal_precision/ \
  /Users/ondulae/Documents/Research/LLM/mas-llm-opinion-formation/results/presentation_reversal_precision/
```
