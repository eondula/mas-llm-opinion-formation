# Block 15: individual attention output heads

Standalone scored-choice follow-up to the component pilot. The two natural
initialization cases and both donor → recipient directions are retained.
No dashboard changes, generated-choice patches, or full conversations are added.

## Run on the remote GPU

From the repository root with the project environment active:

```bash
git pull origin main
python results/presentation_reversal_heads/run.py --device cuda
```

`--device auto` prefers CUDA then MPS, never CPU. `--device mps` is supported;
CPU requires explicit `--device cpu` for diagnostics. The original runs used CUDA,
so CUDA is preferable for exact replay. No silent backend or precision fallback.

Required local source files: the existing behavioral/reversed study runs and
`results/presentation_reversal_components/patches.csv`. The earlier pilot code
must also be present (it is tracked); its hook and device utilities are reused.
Model loading, prompting, generation and full endorsement scoring use the project
utilities. No model experiment is run by installing or importing this code.

## Intervention location and verification

A forward pre-hook on block 15's `self_attn.o_proj` receives concatenated
**query-head weighted value outputs**, after aggregation and before projection.
Head h occupies `[h * head_dim : (h + 1) * head_dim]`. Only those coordinates
at the final prompt token are replaced. Every other recipient coordinate and
token is retained. No slices of projected outputs are labeled as heads.

The dimensions are taken from the loaded Qwen config and projection and checked.
Grouped-query attention shares keys/values, but the intervention indexes query
output heads, not key/value heads. Dimensions and head counts are saved in
`head_layout.json`. This adapter targets the Qwen2.5 architecture in these cases.

Exact rendered prompts and responses plus decision/post scores must replay within
absolute score tolerance 0.0001. Capture must preserve both scores, with identical
prompt activations under both candidate suffixes. Hooks are removed after each
call including exceptions; independent teacher-forced endorsement calls use no
shared cache. No candidate suffix position is patched.

Before ANY individual-head test, replacing all heads together must reproduce
both endorsement scores from (a) a fresh whole-attention output replacement and
(b) the saved component pilot's attention replacement, within the same tolerance.
This gate is applied to donor and self replacements in both cases and directions.
`location_controls.csv` records all three score pairs and discrepancies. Failure
stops the sweep rather than loosening tolerance or treating a margin match alone
as equivalence. A numerical failure warrants inspection of environment and
projection numerics before interpreting any individual heads.

Every individual head has a self-replacement control. S1 wins ties. The complete
scores, margin, chosen strategy and transfer indicator are recorded separately.
Donor agreement in the unchanged pair is not transfer.

## Conditional small combination

The prespecified exploratory rule selects the top **two** heads with positive
margin deltas at least **0.02** in the changed case, reversed → forward. If fewer
than two qualify, the joint test is skipped. This absolute threshold is a
screening convention, not a significance threshold. It is configured before the
run in `config.json`. Tie breaks use head index.

The selected pair is tested jointly in all four case/direction contexts, with
self controls. Joint scores are measured directly, never replaced by the sum of
single-head effects. Selection and its reason are saved even when skipped.
Opposing single-head effects remain visible in the plots and CSV.

## Outputs

- `config.json`, `inputs.json`, `matched_cases.csv`: cases, source hashes,
  reconstructed prompts, baseline scores and responses.
- `environment.json`, `replay.csv`, `status.json`: GPU and replay audit.
- `head_layout.json`, `reference_provenance.json`: head geometry and reference hash.
- `activations_*.pt`: pre-projection head vectors and attention outputs.
- `observation_controls.csv`, `location_controls.csv`: capture and equivalence gates.
- `patches.csv`: complete all-head, individual-head and optional joint records,
  including self controls, both scores, margin, strategy and donor/recipient baselines.
- `combination_selection.json`: chosen pair or explicit skip reason.
- `head_effects_data.csv`: every plotted donor measurement.
- `head_effects.png`: signed margin change by head, separate panels for both cases
  and directions; all-head and optional measured joint effects as reference lines.
- `interpretation.md`: descriptive signed effects and transfer heads, no causal
  necessity or statistical significance claims.

No PDF figures. Reproduce the PNG from CSV after a successful run:

```bash
python results/presentation_reversal_heads/run.py --plot-only
```

Synthetic tests only (no model downloads or inference):

```bash
python results/presentation_reversal_heads/test_pilot.py
```

## Transfer to the local Mac

Replace `GPU_HOST` with your SSH hostname or alias. Run locally:

```bash
rsync -avP \
  win.ondulae@GPU_HOST:/home/win.ondulae/mas-llm-opinion-formation/results/presentation_reversal_heads/ \
  /Users/ondulae/Documents/Research/LLM/mas-llm-opinion-formation/results/presentation_reversal_heads/
```
