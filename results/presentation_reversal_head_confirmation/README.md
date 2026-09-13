# Heads 2 and 9: confirmation, token scores and replacement strength

Standalone follow-up at block 15, using the same scored natural initialization
pairs and both donor → recipient directions. No model run is launched by setup.

## Run remotely

From the repository root with the project Python environment active:

```bash
git pull origin main
python results/presentation_reversal_head_confirmation/run.py --device cuda
```

`--device auto` prefers CUDA then MPS and never falls back to CPU. Explicit
`--device mps` is available; CUDA is preferable for matching the source runs.
Existing behavioral and reversed study files plus
`results/presentation_reversal_heads/patches.csv` must be present. Previous pilot
code supplies hook and device utilities. Checkpoint weights are loaded only by
normal execution, not by `--plot-only` or `--prepare-only`.

## Protocol

The adapter uses the same final prompt token and pre-o_proj query-head slices
as the validated head pilot. Heads 2 and 9 are zero indexed, at block 15.
For each head, alpha is independently 0, 0.5 or 1. Interior values use
`(1-alpha)*recipient + alpha*donor` in float32, then cast to the existing model
dtype. Endpoints use exact unchanged/assignment operations. No other positions
or head slices are replaced. No precision or backend fallback is performed.

Each of the nine conditions is run three times in each of four case/direction
contexts, with donor and self interpolation: **216 scoring conditions**, each
with two independent complete endorsement likelihood calls. Hooks are removed
after every call; caches are not shared. Self interpolation tests all strengths.
The three repeats check numerical consistency, not independent replications.

Before the grid, exact saved prompts/responses and decision/post scores must
replay. Observation must preserve both scores and candidate suffixes must not
change the captured prompt state. The 0/0 baseline and full 1/0, 0/1 and 1/1
conditions must match the previous head pilot within absolute tolerance 0.0001.
Self/zero conditions must retain baseline scores; repeated per-token scores must
agree within tolerance. Any failed check stops interpretation and writes status.

For every candidate, save token IDs, token strings, log-likelihoods, baseline
log-likelihoods, deltas and signed contributions to the S1-minus-S2 margin.
The shared prefix ends at the first differing token ID. The remaining suffix
includes all subsequent tokens, not just attribute words. Positions need not
align across alternative suffixes: their contributions are summed separately.
Token changes identify changes in output scoring, not linguistic head roles.

## Artifacts

- `config.json`, `inputs.json`, `matched_cases.csv`: design and source provenance.
- `environment.json`, `head_layout.json`, `reference_provenance.json`: execution,
  geometry and prior result hash.
- `replay.csv`, `observation_controls.csv`, `confirmation_controls.csv`, `status.json`:
  baseline, capture and prior endpoint checks.
- `activations_*.pt`: captured pre-projection donor vectors.
- `baseline_tokens.csv`: unpatched candidate token identities and scores.
- `grid.csv`: all conditions, repeats, scores, margins, choices and self/repeat errors.
- `token_scores.csv`: complete token-level measurements for every condition.
- `token_contributions.csv`: shared-prefix versus remaining-suffix margin changes,
  verified to sum to the measured margin change.
- `joint_margin_grid.png`: directly measured 3×3 margins, separate case/direction
  panels. Each panel uses a symmetric color scale around zero and labeled values.
- `token_effects.png`: per-candidate token score changes for head 2, head 9 and joint
  full replacement, with the first differing token marked.
- `interpretation.md`: checks and interpretation limits.

Both figures read saved CSVs directly and show repeat 0 after checking repeat
agreement. No PDFs are generated. To regenerate figures without model inference:

```bash
python results/presentation_reversal_head_confirmation/run.py --plot-only
```

Synthetic tests (no pretrained model inference):

```bash
python results/presentation_reversal_head_confirmation/test_pilot.py
```

## Transfer to the Mac

Run locally, replacing `GPU_HOST` with the SSH host or alias:

```bash
rsync -avP \
  win.ondulae@GPU_HOST:/home/win.ondulae/mas-llm-opinion-formation/results/presentation_reversal_head_confirmation/ \
  /Users/ondulae/Documents/Research/LLM/mas-llm-opinion-formation/results/presentation_reversal_head_confirmation/
```
