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

Preview runs one scenario with two or more agents and shared or per-agent checkpoints. Use --agents 3 and --models followed by three checkpoint IDs for a mixed group. Initialization is natural by default. Use `--initialization 2,1` for forced starts, `--argument-mode generated` for generated arguments, `--device cuda` to select CUDA, or `--no-sentiment` to omit sentiment. Round zero records initialization; `--rounds` specifies subsequent updates. Omit `--run-id` to use the system date and time. Existing IDs cannot be overwritten.

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

### Analyze saved runs

Open **Analyze saved runs** in the dashboard navigation (`/analyze.html`). Select a saved conversation, optionally select a second run/conversation with the same scenario, then choose **Create analysis**. With no additional analyses checked, this exports the study's conversation figures and tables without loading models. It also reuses available saved measurements for that conversation. Each figure has PNG/PDF and plotted-data CSV downloads; tables have full CSV downloads and a 20-row preview.

For new replay or mechanistic measurements, select an agent/round with peer delivery. The dashboard automatically sweeps all layers. Patches stay within the receiver checkpoint. Continuation repeats the intervention separately at each layer, replays the conversation, and checks the history before intervention. These options require fresh model inference and can be slow. They do not rerun the study grid. New outputs are stored under a separate run ID with source hashes and provenance; original runs are unchanged.

Comparisons align agent labels and shared rounds, export both trajectories and B-minus-A margins, and include checkpoint/condition metadata. They are descriptive comparisons, not controlled causal estimates. A quick-preview analysis does not supply missing validation or prediction experiments automatically.

The same workflow can run from the terminal:

```sh
.venv/bin/python -m llm_mas analyze --config analysis_request.json --run-id preview_analysis
```

Example `analysis_request.json` (replace IDs and task with saved values):

```json
{
  "source_run": "Test_5",
  "task": "000_000",
  "modules": ["mechanistic"],
  "agent": "Agent A",
  "round": 1,
  "device": "auto"
}
```

Set `modules` to `[]` for exports only. Add `compare_run` and `compare_task` for a two-conversation comparison. Available modules are `replay`, `mechanistic`, `heads`, `source_spans`, and `continuation`.

### Custom Quick preview scenarios

In the dashboard, click **Create scenario**. Enter an ID and title, then each strategy's name, two properties, candidate statement, and arguments (one per line). **Fill from selected scenario** provides an editable example. **Save and select scenario** validates the form, saves it under `scenarios/custom/`, and selects it for the next run. Existing files are never overwritten.

From the terminal, create an editable template, edit its JSON fields, validate, and run:

```sh
.venv/bin/python -m llm_mas scenarios create --template storage --output scenarios/custom/my_scenario.json
# Edit scenarios/custom/my_scenario.json with your scenario and both strategies.
.venv/bin/python -m llm_mas scenarios validate --file scenarios/custom/my_scenario.json
.venv/bin/python -m llm_mas preview --scenario scenarios/custom/my_scenario.json --rounds 5
```

The filename supplies the default scenario ID; `--id` overrides it. IDs allow letters, digits, underscores, and hyphens. Each strategy needs nonempty `name`, `property_1`, `property_2`, `candidate`, and an `arguments` list with at least one nonempty string. The two candidate statements must differ.

### Agent-selected arguments

Three argument modes are available in quick preview and study conditions:

- `predefined`: existing cyclic assignment from each strategy's ordered list.
- `agent_selected`: after choosing a strategy, the model scores the numbers of that strategy's listed arguments under a separate selection prompt and its current history. The highest-likelihood number determines which exact argument is communicated. The instruction requests a response to the latest peer reasoning from the fixed list, preferring relevant development over repetition. Without peer messages it requests an independent rationale. Text is used verbatim: no rewriting or combining. Repeated choices are allowed when relevant; ties use the first entry. List order/number labels may affect selection and should be counterbalanced when studying argument choice.
- `generated`: existing freely generated argument for the selected strategy.

The selection prompt/list is not appended to conversation history. Only the candidate plus selected argument is appended and delivered. Saved response rows contain a zero-based `argument_index` and, for agent selection, `argument_selection` with the prompt, rendered prompt, labels, scores and chosen number. Stance scoring is unchanged. This deterministic constrained selection requires one extra likelihood pass per argument, so it is slower than cyclic assignment. Existing reports describe the original modes, not this new condition.

```sh
python -m llm_mas preview --scenario storage --argument-mode agent_selected --rounds 3 --run-id selected_preview
python -m llm_mas study --config studies/argument_modes.json --run-id argument_comparison
```

In the dashboard choose **Conversational (fixed list)** under **Arguments**. For studies, uncheck the paper suite to use this setting or specify `argument_mode` in the condition list. The original paper-suite preset remains unchanged for reproducibility; `studies/argument_modes.json` is a separate three-mode comparison.

### Conversational arguments

Select **Conversational (generated)** in the dashboard or use `--argument-mode conversational`.
The decision margin still selects the strategy (except assigned initialization).
One generation call then produces an argument responding to the latest delivered
peer messages. At initialization and in no-peer control, it instead requests an
independent rationale. The canonical candidate remains the public-message prefix.
Repetition avoidance and scenario consistency are instructions, not enforced guarantees.
The generation instruction is saved in each row's `generation.instruction`; it is
not appended to retained history. Normal schedule and history rules still apply.

```sh
python -m llm_mas preview --scenario storage --argument-mode conversational --rounds 3
```

Study JSON accepts `"argument_mode": "conversational"` at the top level, in explicit
conditions, or in the argument-mode grid. Use selected conditions with the dashboard
paper suite unchecked; existing study presets retain their original conditions.

### Combined prompt with fixed argument texts

`combined_fixed` / **Combined prompt (fixed list)** includes both candidate statements
and both argument pools in one decision instruction. The candidate-only likelihood
margin selects the strategy. Each actual argument of that strategy is then scored
as a continuation of the selected candidate plus a newline under the same prompt.
Only argument-token log-likelihoods enter this second ranking; scores are summed
without length normalization. Exact selected text is appended to history. No argument
numbers or second user instruction are used. Repetition is discouraged, not forbidden.

Both pools are visible in before/decision/post stance probes, including natural
initialization. This is a different measurement context from the other modes. Forced
initialization still assigns the strategy. Selection scores, token counts, prompt,
and continuation prefix are saved. List ordering, wording, and argument length may
influence selection. Saved-history replay/interventions use the condition's probe.

```sh
python -m llm_mas preview --scenario storage --argument-mode combined_fixed --rounds 3
```

Study configurations and condition grids accept `"argument_mode": "combined_fixed"`.
For dashboard studies, uncheck the original paper suite to use this mode. Existing
study presets and historical results are unchanged.

### Combined prompt with flexible reasoning

Use `combined_reasoning` / **Combined prompt (flexible reasoning)** to let agents
combine and rephrase any number of the provided reasons. Both lists appear in the
shared stance/response prompt. The stance margin selects the strategy; the model
then generates the explanation as a continuation of that candidate, with no second
instruction and no per-argument scoring. Forced initialization remains assigned.
The instruction permits acknowledging the other strategy's benefits while explaining
the selected choice. Grounding and avoiding repetition are encouraged, not enforced.
There is no argument-count limit, but `max_argument_tokens` still bounds generation.
The explanation is stored in `argument`, with generation metadata and the exact
instruction; `argument_index` and `argument_selection` are null.

```sh
python -m llm_mas preview --scenario storage --argument-mode combined_reasoning --rounds 3
```

Study JSON accepts `"argument_mode": "combined_reasoning"`, including condition grids.
Uncheck the dashboard paper suite to use selected settings. `combined_fixed` remains
the separate exact-single-argument condition, so earlier configurations are not relabeled.

### Experiment C

```sh
python -m llm_mas study --config studies/experiment_C.json --dry-run
python -m llm_mas study --config studies/experiment_C.json --run-id experiment_C
```

Defaults: Sensor and Storage, two agents, natural and forced A:S1/B:S2 starts, 20 update rounds,
combined flexible reasoning, synchronous and forward sequential only. Each of the
three checkpoints runs as a homogeneous pair. Exactly one heterogeneous assignment
runs: Agent A = Qwen2.5-1.5B-Instruct; Agent B = SmolLM2-1.7B-Instruct. No reversed
assignment or other mixed combinations are included. Total: 32 conversations.

Dashboard: use **Quick preview** for one selected scenario and configuration,
with either **No persona** or **Persona**. Use **Study** for sweeps. The terminal
`studies/experiment_C.json` preset retains the full original Experiment C design.

For each scenario, `figures/` contains eight PNGs (four per initialization): homogeneous synchronous,
homogeneous sequential, heterogeneous synchronous, heterogeneous sequential. Each
homogeneous figure has three model rows, with post-response stance on the left and
public-message sentiment on the right. Heterogeneous figures have one row showing
the fixed pair. No PDFs are generated. `tables/homogeneous_summary.csv` and
`tables/heterogeneous_summary.csv` contain per-conversation outcomes, agreement timing,
agent checkpoints, initial/final strategies, decision-switch counts, final decision
and post-response margins, final stance/sentiment and repeated-message counts.
Stable agreement is measured only through the observed final round. Raw histories
and generation metadata remain in conversation JSON. The dashboard's saved-figures
page and ZIP download include the PNGs and summary CSVs.

### Analyze a completed Experiment C run in place

After pulling this code on the machine holding the results, run:

```bash
# Uses saved measurements only; no model loading or new conversations.
.venv/bin/python -m llm_mas analyze-c --run-id experiment_C_v2

# Adds all-layer activation and intervention heatmaps; requires model inference.
.venv/bin/python -m llm_mas analyze-c --run-id experiment_C_v2 --stage all --device cuda
```

Outputs go directly into that run's existing `figures/` and `tables/` folders.
Original conversations and existing figures are retained. The new exports use
an `analysis_caseNN_` prefix; the existing paper ZIP is not updated.

Selection is deterministic, separately per scenario: longest consecutive public
disagreement, a scored strategy switch maintained for at least six observed
rounds including its switch round, repeated scored switches, and the largest
public-choice difference between matched synchronous/sequential conversations.
Absent patterns and duplicate selections are recorded in
`tables/analysis_selection.csv`. Forced initialization and its transition to
round 1 are excluded from scored-switch counts. A schedule comparison is one
case containing both conversations. No case is selected using patch outcomes.

For each selected case:

- `*_within_step.png`: separate agent panels with raw before, decision, and post
  margins; public strategies and scored switches are marked. Forced round zero
  has no decision point. The corresponding CSV also retains saved sentiment.
- `*_activation_change.png` (`--stage all`): relative residual-vector change
  after peer delivery, across every transformer block and update round.
- `*_intervention_effect.png` (`--stage all`): patched minus baseline full-string
  decision margin, with markers for hypothetical public-choice flips.
  `*_layers.csv` includes the plotted values and numerical controls.

For K selected cases, the behavioral stage adds K PNGs; the full stage produces
3K PNGs. The existing stance/sentiment figures remain available. No PDFs or new
conversation trajectories are generated.

The fixed site is the residual output of each block at the final prompt token.
Each intervention independently replaces the after-delivery activation with
that same agent/checkpoint's before-delivery activation. Token indices may differ
between contexts; both are recorded. Both candidate strings are rescored under
the same patch. This tests that site's effect on the saved decision context;
it does not isolate semantic persuasion from message length or position changes,
and it is not a layer contribution decomposition or a sentiment intervention.
Late-layer effects may be small because this site has limited access to later
candidate positions. Round zero has no peer-delivery heatmap cell.

Replay uses the saved populated decision probe and verifies the exact rendered
prompt, the unmodified margin, and a self-patch at every layer. A margin error
above `--tolerance` (default 0.01) stops the analysis. Use the original model
versions and environment where possible; model revision drift can prevent replay.
The analysis manifest records source conversation hashes and the protocol.
All-layer analysis can be expensive: each selected update requires independent
candidate scoring for every patch and self-patch. Progress prints per layer.
Rerunning recomputes analysis and replaces only its prefixed exports.

`analyze-c` prints a progress bar, elapsed time, and an approximate remaining
ETA after each layer. Each unique agent/round replay is one work unit, subdivided
by its layer count; each case's figure export is another unit. Reused replay rows
are not counted twice. ETA starts as `estimating` and extrapolates elapsed wall
time, so it can change substantially with longer histories, other checkpoints,
and model-loading delays. It is not a GPU benchmark or a guaranteed finish time.
The same values are saved to `tables/analysis_status.json`, including during
loading, completion, cancellation, and failure. Background logs contain one
readable line per update rather than terminal control characters.

Pulling updated code does not add progress reporting to an already-running
Python process. Existing jobs can finish normally; a newly started job uses the
new reporting. This analysis does not yet resume partially completed replay work.

### Persona quick preview

Choose **Quick preview**, then **No persona** or **Persona**. Both options use
exactly the agent count, model configuration, scenario, initialization, schedule,
argument mode, round count and token limit selected in the controls. Changing
this preset does not reset settings or impose Experiment C defaults.

With Persona, scenario roles alternate by agent: A/C/etc. receive the Strategy 1
role, B/D/etc. the Strategy 2 role. Roles persist in system instructions but do
not force public choices. Initialization remains an independent setting. Prompt
preview shows every agent's initial context. Scenarios must define both
`persona_roles`; missing roles produce a validation error. No persona adds no
role instruction. Quick preview continues to run one conversation.

### Experiment D: persona comparison, 10 update rounds

```bash
.venv/bin/python -m llm_mas study --config studies/experiment_D.json --run-id experiment_D_v1
```

The preset crosses Sensor/Storage, the four Experiment C checkpoint assignments,
no persona/persona, natural/forced A:S1 B:S2 starts, and synchronous/forward
sequential delivery. It produces 64 conversations and 1,408 responses (round 0
plus 10 updates per agent). Both persona conditions use the current
`combined_reasoning` paragraph prompt, greedy generation, at most 128 explanation
tokens, full retained history, and sentiment scoring. The research list is empty;
there is no internal replay, activation capture, or intervention analysis.

After scoring, exports in the same run's `figures/` folder contain 32 grouped
stance/sentiment PNGs and four public-choice overview PNGs. No PDFs are created.
`tables/` contains homogeneous/heterogeneous outcome summaries with persona
labels, per-response margins and changes, public/post-stance mismatches,
repetition and paragraph-format checks, stance/sentiment changes, and matched
persona and schedule comparisons. Format checks are heuristics, not guarantees
of semantic compliance. Forced initial transitions and scored switches are
separate columns. Agreement is measured only through the observed final round.
All figures and tables are included in `paper_outputs.zip`.

This differs from Experiment C v2 in prompt wording and duration as well as the
added persona factor. Use within-D matched conditions for persona comparisons.

### Quick preview explanation decoding

Quick preview offers Greedy or Sampling. Sampling exposes Temperature only,
Top-p only, or both controls, plus an editable base seed (default 42). Temperature
only fixes top-p at 1; Top-p only fixes temperature at 1. Top-k is disabled in all
sampling modes. The default input values are temperature 0.7 and top-p 0.9.
Each generated response uses `base_seed + round * agents + agent_index`.
Effective parameters are validated and saved in the configuration and generation
records. Reproducibility depends on hardware, checkpoint and software versions.
In scored-choice mode, sampling affects explanations while choices use candidate
likelihoods. Generated-choice mode samples the structured choice and explanation.
Use a generated argument mode; fixed-list modes reject sampling. Study runs in
the dashboard keep greedy decoding. Existing runs are unaffected by UI changes.

### Full behavioral study

See [the full experiment run guide](docs/full_experiment_run.md) for the smoke test,
full runs, separate controls and analysis commands. Presets select CUDA/MPS/CPU
automatically. Full runs use a maximum of 15 update rounds.
