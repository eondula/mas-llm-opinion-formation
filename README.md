# Opinion Lab

Study context-conditioned stance in language-model agents with fixed model weights. The terminal runner and local dashboard share the `llm_mas` implementation. Each agent maintains a separate history and evaluates two candidate strategies using full-string conditional log-likelihoods.

## Setup

Run commands from the repository root. Use Python 3.11 or newer. Create an environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Model checkpoints and the sentiment classifier are downloaded on first use; subsequent runs can use the local Hugging Face cache. CUDA, Apple MPS and CPU are supported through PyTorch. Install a PyTorch build appropriate to your hardware. The current package does not require `src/`, legacy scripts, previous results or historical reports.

## Quick preview

```bash
python -m llm_mas preview --scenario storage --rounds 3 --run-id preview_check
```

Preview runs two agents sharing one checkpoint on one scenario. Initialization is natural by default. Use `--initialization 2,1` for forced starts, `--argument-mode generated` for generated arguments, `--device cuda` to select CUDA, or `--no-sentiment` to omit sentiment. Round zero records initialization; `--rounds` specifies subsequent updates. Omit `--run-id` to use the system date and time. Existing IDs cannot be overwritten.

## Dashboard

```bash
python -m llm_mas dashboard
```

Open http://127.0.0.1:8766. The dashboard supports configuration, live responses, curved delivery arrows, saved-run inspection, and completed figure/table downloads. Its Run modes, Experiment setup and Measurements links describe the current controls. Closing the browser does not stop the worker; use Stop to interrupt it.

## Studies

Inspect the full preset without loading models:

```bash
python -m llm_mas study --config studies/paper.json --dry-run
```

Run it with:

```bash
python -m llm_mas study --config studies/paper.json --run-id paper_study
```

The current terminal preset contains 3,168 main conversations: Sensor and Storage, 20 rounds, natural and all forced starts, four schedules, two argument modes and two presentation orders. The checkpoints are Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct and HuggingFaceTB/SmolLM2-1.7B-Instruct. Two-agent runs include all nine ordered pairs (three shared and six mixed). Three-agent runs use all three different checkpoints, with all six assignments to Agent A/B/C to counterbalance position. This gives 1,440 two-agent and 1,728 three-agent conversations. Research modules add measurements and intervened continuation runs. This can take substantial compute time; benchmark on your hardware before launching the preset.

The dashboard paper suite expands conditions for the selected agents and checkpoint assignment. It does not automatically apply the terminal preset's agent-count and checkpoint sweep.

For a smaller study, create a JSON configuration such as:

```json
{
  "mode": "study",
  "models": ["Qwen/Qwen2.5-1.5B-Instruct"],
  "agents": 3,
  "scenarios": ["storage"],
  "rounds": 3,
  "initialization": [2, 1, 1],
  "schedule": "synchronous",
  "argument_mode": "predefined",
  "research": ["validation", "history"]
}
```

A single model entry shares a checkpoint; otherwise supply one entry per agent. Schedules are `synchronous`, `sequential`, `reverse_sequential`, and `no_peer`. Scenario JSON files define two candidates, properties and nonempty argument pools. Predefined arguments cycle through each strategy's pool.

## Outputs and resume

Each run creates `results/studies/<run_id>/` with configuration, prompts, scenarios, task metadata, conversations, status and events. Requested research modules save measurements and analyses. Export creates figures, transcripts, CSV data, selected LaTeX tables, an HTML figure index and `paper_outputs.zip`. The ZIP contains figures and tables including copied transcripts; full raw records remain in the run directory. Outputs and downloaded weights are excluded from Git.

```bash
python -m llm_mas figures --run-id paper_study
python -m llm_mas resume --run-id paper_study
```

Resume checks source and input identity, reuses completed conversations/modules, and restarts interrupted units. Source changes require a new run ID. ETA is phase-specific. Multiple checkpoints occupy CPU memory; the active checkpoint is placed on the accelerator.

## Tests

Install Node.js to run the dashboard configuration regression test. The remaining tests use Python and the listed dependencies.

```bash
python -m unittest discover -s tests/refactor -v
node --check dashboard/study.js
```

Tests use simulated responses and a tiny local model to check scheduling, pipeline exports and interventions without downloading pretrained checkpoints. They do not establish scientific validity or full-suite GPU compatibility.

## Measurement and current limitations

The raw margin is candidate 1 log-likelihood minus candidate 2 log-likelihood. The plotted score is `tanh(margin / 2)`. Decision-stage scores select strategies; post-response scores reflect the appended response. Sentiment measures public wording and is scored after conversations and research analyses.

Group trajectory prediction currently covers two agents only. No dedicated no-history conversation control is implemented. Validation focuses on initial contexts; label, formatting and attribution robustness are incomplete. Internal analyses use selected events and supported model components; sufficiency matching can find no eligible pair. Export coverage does not guarantee complete research coverage. A full real-checkpoint study remains to be validated.

The dashboard license is in `dashboard/LICENSE`; no project-wide license has been selected.
