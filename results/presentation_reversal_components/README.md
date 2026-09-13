# Presentation reversal: attention and MLP outputs

This standalone follow-up investigates blocks **12–15 (zero indexed)** in the
same two Qwen2.5-1.5B scored-choice initialization pairs as the residual pilot.
One pair changes S2 to S1 under reversal; one retains S1. Both members of each
pair match scenario, checkpoint assignment, priorities, initialization, choice
procedure, and schedule. The two cases differ in persona, so they do not isolate
persona effects. Sources remain `full_behavioral_v2` and `full_reversed_v3`.

No dashboard changes, generated-choice interventions, prediction, new full
conversations, or automatic experiment execution are included.

## Run on the remote GPU machine

From the repository root, after pulling the commit and activating the project
environment containing PyTorch and Transformers:

```bash
python results/presentation_reversal_components/run.py --device cuda
```

For Apple GPUs use `--device mps`. The default `--device auto` prefers CUDA then
MPS, with no silent CPU fallback. `--device cpu` is explicit diagnostics only.
CUDA is preferable for matching the original CUDA runs. Memory or device errors
stop execution instead of retrying on CPU. Device placement is checked and saved.

The saved study folders must already exist on the machine. The implementation
imports tested hook and device helpers from `results/presentation_reversal_patching/run.py`,
and reuses project `Runtime`, prompt construction, scoring and generation.
No prior activation files are required. Activations are recaptured after replay.

## Copy results to the local Mac

Run locally; replace `GPU_HOST` with your SSH hostname or alias:

```bash
rsync -avP \
  win.ondulae@GPU_HOST:/home/win.ondulae/mas-llm-opinion-formation/results/presentation_reversal_components/ \
  /Users/ondulae/Documents/Research/LLM/mas-llm-opinion-formation/results/presentation_reversal_components/
```

## Intervention definition

All interventions replace only the final prompt token in one module at one block:

- **Attention:** `block.self_attn` output after `o_proj`, before residual addition.
  The returned attention tensor is replaced at the selected position; other tuple
  fields remain untouched. This is the full attention output, not one head.
- **MLP:** `block.mlp` output after `down_proj`, before residual addition.
- **Residual reference:** whole block output, reproducing the original target
  at the same four blocks in the current environment.

Each replacement is independent. Attention output can affect the subsequent MLP
within its block, so attention and MLP effects must not be added or interpreted
as disjoint mediation fractions. A residual reference also has a different
intervention location from either component output.

Both directions are run for both cases, with self replacement for every
component and layer: **48 donor replacements and 48 self controls**. Each
endorsement is scored independently with a fresh hook and no shared cache.
Hooks are removed after every call, including errors. Teacher forcing processes
candidate suffixes causally; only the prompt position is replaced. No post-patch
explanation is generated: the target is the pair of complete endorsement scores.

## Gates and artifacts

Before intervention, histories, scoring/generation prompts, full baseline
responses, decision scores and post-response scores must replay. Scores use an
absolute tolerance of 0.0001; responses and prompts must match exactly. A mismatch
is recorded and stops patching. Capture hooks must preserve scores, and prompt
activations must agree for both candidate suffixes. Self controls must preserve
scores and strategy. Failure stops interpretation.

Outputs:

- `matched_cases.csv`, `inputs.json`: source cases, histories, baseline outputs,
  configurations and source conversation hashes.
- `environment.json`, `replay.csv`, `status.json`: device, versions and replay audit.
- `observation_controls.csv`, `activations_*.pt`: 48 module capture checks and
  component vectors, positions, layer indices and component names.
- `patches.csv`: all 96 replacements with both likelihoods, raw margins, margin
  differences, baseline margins and strategies, choice changes, donor match,
  transfer indicators and self errors.
- `component_summary.csv`: effects separated by case, direction and component.
- `component_effects_data.csv`: one row per plotted donor replacement, including
  layer, component, case, direction, both scores, margins, strategy, transfer and
  the choice-boundary value. Both initial plotting and regeneration read this CSV.
- `component_effects.png`: margin changes and resulting strategies;
  the dotted margin line marks the change needed to reach the choice boundary.
- `interpretation.md`: descriptive summary and limits, written only after success.

For the unchanged case, donor agreement is not transfer. The `transfer` field
requires different donor/recipient baseline choices. S1 is selected on a score tie.
Blocks were chosen from the first pilot, so this is exploratory, not an independent
confirmation. No statistical significance or unique mechanism is inferred.

Regenerate plots without inference after a successful run:

```bash
python results/presentation_reversal_components/run.py --plot-only
```

Prepare source tables without loading a model using `--prepare-only`. Synthetic
module tests (no checkpoint downloads or model experiments):

```bash
python results/presentation_reversal_components/test_pilot.py
```
