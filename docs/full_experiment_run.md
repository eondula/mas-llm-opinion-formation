# Full experiment: run commands

Run commands from the repository root on a CUDA machine or Apple Silicon Mac with the project environment installed. Presets use `device: auto` (CUDA, then MPS, then CPU). Add `--device mps` or `--device auto` to a study command to select explicitly. No GPU experiment was launched during implementation.

## Test first

```sh
.venv/bin/python -m unittest discover -s tests/refactor -p 'test_full_study.py'
.venv/bin/python -m llm_mas study --config studies/full_behavioral_smoke.json --dry-run
.venv/bin/python -m llm_mas study --config studies/full_behavioral_smoke.json --run-id full_smoke_v1
```

The smoke run contains 12 conversations: both scenarios, a mixed pair, both choice procedures, and synchronous/reverse/no-peer schedules, capped at two updates. Inspect `tables/outcomes.csv` for invalid generated responses and `figures/index.html` for plots before launching the full jobs. Generated JSON compliance is a real experimental outcome; it is not guaranteed by the prompt.

## Full launch

Run these sequentially on one GPU:

```sh
.venv/bin/python -m llm_mas study --config studies/full_behavioral.json --run-id full_behavioral_v1
.venv/bin/python -m llm_mas study --config studies/full_behavioral_fixed.json --run-id full_fixed_v1
.venv/bin/python -m llm_mas study --config studies/full_behavioral_reversed.json --run-id full_reversed_v1
```

- **Primary: 480 conversations.** Sensor/Storage × five assignments × two personas × three starts × four schedules × two choice procedures. Includes 120 no-peer conversations.
- **Fixed horizon: 120 conversations.** Natural starts under the three interactive schedules; same other factors. Records stopping triggers but continues to round 15, exposing later revisions.
- **Reversed presentation: 120 conversations.** Matches the natural interactive subset while reversing initial scenario presentation and complete candidate blocks. This is a combined presentation treatment, not an isolated label-order effect.

Five assignments: Qwen1.5/Qwen1.5, Qwen3/Qwen3, Smol/Smol, Qwen1.5/Smol, Smol/Qwen1.5. All jobs use 15 updates maximum, greedy generation, a 256-new-token response budget and the supplied argument pools. The larger token budget applies to both choice procedures and differs from D's 128-token explanation budget.

Adaptive stopping requires unchanged public strategy, unchanged post-margin sign and adjacent full-message cosine >= 0.8 for every agent. There is no minimum-round gate. Invalid generated choices cannot satisfy strategy stability. Scored mode selects and prefixes the candidate; generated mode returns a JSON strategy and explanation with no selected prefix. Forced round 0 remains assigned in both modes. Raw generated messages are retained and delivered without repair. Choice and output-format effects are therefore part of this policy comparison.

Each job automatically produces numerical tables, per-conversation four-panel plots (stance, sentiment, public choice, raw margin), grouped outcome plots, length/stopping summaries, observed-conversation counts, switches and explanation-similarity summaries. No stopped trajectory is padded. Similarity analysis uses a cached CPU encoder; sentiment is descriptive and no sentiment validation is run.

## Separate controls and matched comparisons

After the primary job completes:

```sh
.venv/bin/python -m llm_mas.full_study_controls --source results/studies/full_behavioral_v1 --output results/controls/full_behavioral_v1 --device auto --dry-run
.venv/bin/python -m llm_mas.full_study_controls --source results/studies/full_behavioral_v1 --output results/controls/full_behavioral_v1 --device auto
```

Selects both agents at available rounds 1 and 2 of natural synchronous conversations. Entire conversations containing invalid generated choices are excluded; this limits the control population. Each selected context has eight branches: full/removed prior assistant history × full message/endorsement/explanation/no current message. These jobs measure likelihood margins, not new conversational trajectories. They preserve scenario/persona; natural starts avoid residual forced-assessment instructions. Explanation text can itself contain an endorsement. Histories, scores, paired deltas and model-separated control plots are saved. Up to 160 contexts / 1,280 branches; actual coverage is printed by dry-run. Controls resume by branch without modifying source conversations.

After all three conversation jobs complete:

```sh
.venv/bin/python -m llm_mas.full_study_compare --primary results/studies/full_behavioral_v1 --fixed results/studies/full_fixed_v1 --reversed results/studies/full_reversed_v1 --output results/comparisons/full_behavioral_v1
```

Produces matched endpoint comparisons and stopping savings plots. Fixed-horizon tables report whether any later public strategy differs from that at the first stopping trigger. Token savings count generated response tokens, not scoring/embedding overhead or wall-clock savings.

## Outputs and resume

Runs live under `results/studies/<run-id>/`. Open `figures/index.html`; numerical sources are in `tables/` and `paper_outputs.zip` packages figures/tables. Raw prompts and histories remain in conversation records. `round_summary.csv` reports how many conversations remain observed at each round. `outcomes.csv` distinguishes invalid endpoints, agreement, disagreement and stopping reasons.

```sh
.venv/bin/python -m llm_mas resume --run-id full_behavioral_v1
.venv/bin/python -m llm_mas figures --run-id full_behavioral_v1
```

Resume requires unchanged source/configuration and restarts unfinished conversations. Re-exporting figures uses saved data only. Keep an unchanged checkout while jobs are running. Scoring validation remains separate and is not part of these launches. Thresholds and single-transition stopping do not guarantee future stability; the fixed-horizon comparison evaluates that limitation.
