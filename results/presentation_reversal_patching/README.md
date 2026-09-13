# Presentation reversal patching pilot

This standalone pilot uses saved natural initialization records from
`results/studies/full_behavioral_v2` and `results/studies/full_reversed_v3`.
`full_fixed_v2` contains continuations and is not used as an independent
initialization replicate. No new conversations or dashboard features are added.

## Selected cases

Both pairs use Qwen/Qwen2.5-1.5B-Instruct, Sensor, homogeneous agents,
Agent A, natural initialization, synchronous delivery, greedy decoding,
and scored choice with combined reasoning. The changed pair has no persona
and changes S2 to S1. The unchanged pair enables persona (Agent A prioritizes
precision) and retains S1. Forward and reversed members of each pair match
all these conditions; only alternative display and candidate block order differ.
The two pairs differ in persona and are illustrative controls, not an isolated
estimate of the effect of persona. Initialization has no peer messages.

## Run

From the repository root, using the project environment:

```sh
python results/presentation_reversal_patching/run.py --device cuda
```

The default `--device auto` prefers CUDA, then Apple MPS, and never falls
back to CPU. Use `--device cuda` or `--device mps` to require a specific backend.
If the requested GPU is unavailable, the pilot stops before loading the model.
Use `--device cpu` only for explicit CPU diagnostics; replay may differ from
the saved CUDA runs. MPS can also differ numerically from CUDA; replay verification
remains required on either backend. The pilot prints the GPU name, checks model placement,
and records GPU and CUDA details in `environment.json`. GPU memory errors
remain errors; the pilot does not retry on CPU.
Run `--prepare-only` to regenerate the matched-case table and reconstructed
inputs without model inference. Test the hook lifecycle with:

```sh
python results/presentation_reversal_patching/test_pilot.py
```

## Verification and intervention

The script reuses `Runtime`, `initial`, and `for_condition`, with the saved
prompt templates and configuration. It checks reconstructed decision histories,
rendered scoring/generation prompts, full responses, both decision endorsement
scores, and post-response scores. Responses must match exactly and likelihoods
must agree within the declared absolute tolerance (default 0.0001).
Any discrepancy is written to `replay.csv`; failed replay prevents patching.
The current device, package versions, dtype and checkpoint revision are saved.
Differences from the original CUDA environment may prevent verified replay.

After verified replay, observation hooks capture the hidden state after every
transformer block at the final prompt token. Observation checks both endorsement
scores and that candidate suffixes do not change captured prompt states.
Activations are saved on CPU in `.pt` files, with zero-indexed blocks and positions.

This is explicitly a **scored-choice pilot**. At each block and direction, the
recipient final prompt state is replaced with the donor state. Each endorsement
gets a fresh hook; the runtime scores the complete endorsement with caching
disabled, and the hook is always removed, including after exceptions. The
teacher-forced suffix follows the prompt causally; no suffix position is patched.
No cache is reused between candidates or interventions. Self replacement is
checked at every block in both directions. Its failure stops interpretation.

The selected strategy is the argmax of the two full endorsement likelihoods
(S1 on a tie). Explanations are regenerated only for baseline replay, not after
scoring interventions. No generated-choice intervention is implemented or claimed.
A future generated-choice pilot would need a separate generation-prefill target
and validity outcomes; it cannot be inferred from these margin interventions.

## Artifacts

- `config.json`: selected cases and intervention target.
- `matched_cases.csv`: condition table, saved baseline choices, scores, responses.
- `inputs.json`: saved source rows, reconstructed messages, conditions and hashes.
- `environment.json`, `replay.csv`, `status.json`: runtime provenance and replay gate.
- `observation_controls.csv`, `activations_*.pt`: capture controls and block states.
- `patches.csv`: both scores, margin, margin change, selected strategy, donor match,
  actual choice change, and self replacement errors by layer and direction.
- `layer_transfer.png` and `.pdf`: resulting strategies and margin changes.
- `interpretation.md`: brief finding and follow-up scope.

Artifacts after the replay gate are only produced if replay succeeds. The
unchanged case's donor match is not strategy transfer: its donor and recipient
already agree. Interpret transfer only when their baseline choices differ.
All plot measurements are in CSV. After a completed sweep, regenerate the layer figure without inference using `python results/presentation_reversal_patching/run.py --plot-only`. These selected cases do not estimate prevalence.
