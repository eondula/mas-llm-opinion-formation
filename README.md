# Opinion formation experiments

Run repeated StrategyQA experiments with language-model agents, then analyze opinion changes, correctness, disagreement, consensus and message cosine similarity. One configuration-driven engine supports independent reasoning and synchronous or sequential deliberation.

## Start the experiment page

From the repository root:

```bash
source .venv/bin/activate
python -m llm_mas dashboard
```

Open **http://localhost:8766** for the agent conversation dashboard, or **http://localhost:8766/experiment.html** for repeated experiments. For a fresh environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m llm_mas dataset download
```

The separate experiment page provides settings, progress, and one summary table after completion. The conversation dashboard retains agents, chat, playback, stance and sentiment plots. Choose models, agents, questions, number of runs, updates, schedule, and natural or forced initial Yes/No answers. Sampling uses saved per-agent seeds. Greedy repetitions may be identical.

**Number of runs** means repetitions of the same selected question set with fresh histories. Ten runs of five questions produce 50 conversations per condition. An optional matched no-peer control doubles that count. Initial judgments are independent; forced answers apply only to the first turn.

## Controlled pilot

On the Experiments page click **Load controlled pilot**, optionally enter an experiment ID, then **Run experiment**. The frozen preset runs 10 balanced training questions across QQ, SS, QS and SQ, with 3 repetitions and 5 updates, natural starts and synchronous/no-peer branches sharing initialization. It produces 120 pairs, 240 conversations, 2,880 logical response rows and 2,640 generated responses. The stance audit runs first on the selected device and stops collection if repeatability fails. All reports remain in the experiment directory.

Scientific settings are locked while this preset is active. **Custom experiment** restores editable controls. The manifest loads directly from tracked `studies/pilot_question_selection.json`; no `docs/` directory or setup copy is required. In the dashboard choose **Pilot compute device → CUDA** for NVIDIA servers, or **MPS** for Apple Silicon, then **Load controlled pilot**. Detection prefers an available CUDA GPU.

Freeze the new protocol once on the machine where collection will run:

```bash
# On an allocated GPU node, using your CUDA-enabled Python environment:
python -m llm_mas pilot-freeze --device cuda
python -m llm_mas dashboard --port 8766
```

For Apple Silicon use `pilot-freeze --device mps`. New locks capture source, prompts, dependencies, device and CUDA/GPU identity in local ignored files. The dashboard shows the required freeze command and disables pilot launch until it passes. Refresh the preset after freezing. A changed lock is never overwritten; start a separately versioned protocol for scientific changes.

The v6 CUDA preset uses float32 weights for compatibility with NVIDIA GPUs that lack bfloat16 support; MPS retains bfloat16. These are separate protocols and should remain separate analysis conditions. Actual checkpoint and tokenizer commits remain pinned. Unavailable devices fail explicitly rather than falling back to CPU. A CUDA build of PyTorch and a GPU allocation are required; login nodes may have no CUDA device even when the cluster has GPUs.

Equivalent CLI launch: `python -m llm_mas run --config studies/pilot_cuda_v6.json` or `studies/pilot_mps_v6.json`. The earlier v1/v2/v3/v4/v5 configurations and locks are retained for provenance; this release uses v6 to keep its changed execution source separate from existing batches. The updated runner rejects collection or resume under retired versions. Finish existing batches with their original checkout. Restart an old dashboard process after pulling so it loads the new manifest lookup.

The dashboard uses shared border-box sizing and flexible grids, with a single-column layout on smaller viewports. Wide summary tables scroll inside labeled, keyboard-focusable regions. Skip links, visible keyboard focus, labeled progress, chart descriptions, touch targets and reduced-motion preferences follow [W3C reflow](https://www.w3.org/WAI/WCAG22/Understanding/reflow.html), [focus](https://www.w3.org/WAI/WCAG22/Understanding/focus-visible.html) and [target-size](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html) guidance. Automated structural checks do not establish full WCAG conformance.

## Background execution and reconnecting

**Finish any existing foreground pilot before pulling this update.** Changing source files during a frozen collection invalidates that batch. New background behavior cannot be applied retroactively to an already-running foreground process.

After updating and freezing the v6 protocol on your GPU node, start the dashboard without tying it to an SSH terminal:

```bash
bash scripts/start_dashboard.sh 8766
```

The launcher uses `.venv/bin/python` by default. For another CUDA-enabled environment, set `OPINION_PYTHON` to its Python executable. The service log is `results/dashboard.log`. Launch the pilot through the Experiments page as usual. Its worker starts in a separate session with stdin disconnected and stdout/stderr redirected to `results/experiments/<id>/worker.log`; closing the browser, SSH connection or dashboard terminal does not interrupt that new worker.

After reconnecting, recreate your SSH port-forward and open `/experiment.html`. The page automatically follows the latest active experiment, otherwise the latest launched experiment. Use **Follow latest run** to return after inspecting an older saved summary. A failed saved run appears separately from new-launch errors. Click **New experiment** to clear the old view without deleting its results, then load the pilot or choose Custom experiment. The message next to Run experiment explains whether another worker is active or the pilot is not ready. If the dashboard service stopped, run the launcher again on the same node. It reconnects to saved worker identity and progress, and **Stop** still sends an interrupt to that verified worker. PID creation time, owner and command are checked before a reconnected process can be controlled. A reused PID or an unrelated process is never treated as the worker. Runs observed from another node show their last saved progress without claiming live control there.

`worker.json`, `status.json`, logs and results persist outside the dashboard process. Workers continue if only the dashboard is restarted; stopping a pilot still cancels it. This does not survive a server reboot or extend a GPU allocation. On a scheduler-managed cluster, use the site's batch-job mechanism for an allocation that outlives SSH; closing an interactive allocation can terminate its jobs regardless of backgrounding.

## Outputs

New experiments are saved under:

```text
results/experiments/<experiment-id>/
  config.json, manifest.json, environment.json
  status.json, worker.log              # worker.log for page-launched runs
  index.html, summary.json, RESULTS.md
  inputs/                             # dataset selection, prompts, evaluator labels
  conversations/                      # messages, raw measurements and delivery provenance
  analysis/behavioral.json
  analysis/evaluation.json, analysis/message_similarity.json
  tables/                             # summary, turns, trajectories, groups, controls
  audit/, initializations/            # audit data and shared initial-state snapshots
  figures/                            # per-question single-repetition PNGs only
```

Correctness labels are used only after simulation. The primary stance is now recorded from raw Yes/No token logits at the public-answer selection step: margin = log P(Yes) − log P(No), stance = tanh(margin / 2). Sampling is preserved; a selected answer need not match the raw score sign. The original separate pre/post-response probes remain saved as auxiliary measurements. Historical records keep their original post-response scores.

Saved HTML reports include metric definitions, denominators and detailed exports. Use `python -m llm_mas dataset info` to inspect dataset and split provenance.

## Terminal execution

```bash
python -m llm_mas run --config studies/experiment.json --dry-run
python -m llm_mas run --config studies/experiment.json --repetitions 10 --run-id experiment_001
```

The preset uses the new experiment directory. New StrategyQA runs retain CSV/JSON analysis tables and generate only per-question PNG figures with stance, sentiment and explicit-answer group states. Figure-side CSV/PDF exports and report ZIPs are not generated. Cross-experiment reports are saved in `results/experiments/comparisons/`. Change question count, models, rounds, conditions and decoding in the JSON configuration. Regenerate complete reports for saved StrategyQA runs without any model inference:

```bash
python -m llm_mas reports --run-id experiment_001
python -m llm_mas reports --all
```

## Preservation and validation

The agent conversation dashboard remains active alongside the new experiment page. Documentation, tests, archived implementations and saved research runs remain local and are excluded from new commits.

```bash
MPLCONFIGDIR=/tmp/opinion-lab-matplotlib python -m unittest discover -s tests/refactor
```

### Offline modeling of the saved pilot

Run from the project root; this reads the existing experiment without loading language models or changing its files:

```bash
.venv/bin/python scripts/model_saved_pilot.py \
  --source results/experiments/new_exp_1_a \
  --output results/modeling/new_exp_1_a
```

Choose a fresh output directory for another analysis. Outputs include modeling rows, held-out predictions, fold parameters, per-question errors, summary CSV/JSON, a prediction-error figure, and a source-hash manifest. All repetitions, compositions, agents, and matched branches for a question stay in the same held-out fold. The target is the original post-turn stance. One-step predictions use observed prior states; recursive predictions start from observed initialization and subsequently use predicted own and peer states. Recursive forecasts model stances, not future language.

Persistence, an affine drift model fitted only to training-question no-peer controls, constrained DeGroot averaging, and constrained Friedkin–Johnsen anchoring are compared separately for each checkpoint and condition. Errors are averaged equally across questions. `valid_only` is a sensitivity analysis using valid current, prior, initial, and delivered-peer responses; original invalid records remain untouched. Fold weights are descriptive and cross-model stance scales are uncalibrated. Ten question clusters support exploratory comparison, not causal identification. No-peer DeGroot equals persistence; no-peer FJ own/peer weights cannot be identified separately.

Evaluate those held-out forecasts as explicit-answer and consensus predictions without refitting or generating responses:

```bash
.venv/bin/python scripts/evaluate_saved_outcomes.py \
  --source results/experiments/new_exp_1_a \
  --modeling results/modeling/new_exp_1_a \
  --output results/modeling/new_exp_1_a_outcomes
```

The evaluator exports per-agent and per-group predictions, metrics for every update round and the final recorded round, and question-level group confusion counts as CSV/JSON. It compares each stance-sign forecast with both the saved stance sign and the agent's explicit answer, and evaluates correctness using evaluator-only labels. Fixed initial-explicit-answer, always-Yes, and always-No baselines provide context. Group states distinguish unanimous Yes, unanimous No, disagreement, and unknown outcomes due to invalid responses. Exact-zero stance forecasts are ties. Incomplete valid-only groups are excluded from group metrics and their coverage is reported. All rates average questions equally; precision and recall exclude question groups with zero relevant denominators. Invalid explicit answers count as unsuccessful in strict matching and correctness, with valid-only matching reported separately. Group correctness means correct unanimous consensus. The evaluator verifies both input directories remain unchanged and rejects an experiment that differs from the modeling manifest.

### Configuration-aware system identification

Fit and evaluate reduced models of output dynamics from the existing pilot, without language-model inference:

```bash
MPLCONFIGDIR=/tmp/opinion-lab-matplotlib .venv/bin/python scripts/identify_saved_system.py \
  --source results/experiments/new_exp_1_a \
  --output results/modeling/new_exp_1_a_system_identification
```

Open `index.html` in the output directory to view the figures and tables. Each question has a figure showing all four compositions, both conditions, both agents, observed repetition trajectories, and held-out recursive forecasts. Other figures compare errors, configuration differences, final outcomes, and unbounded recursive margin magnitudes. Existing experiment and earlier modeling outputs remain unchanged; choose a fresh output directory for another analysis.

The primary target is the raw post-turn Yes/No log-likelihood margin; stance predictions use the original `tanh(margin/2)` transform. Checkpoint-specific self dynamics regress on previous own margin, initial-minus-own margin, round, agent slot, and presence of peer delivery. Coupled dynamics add delivered-peer-minus-own margin. Configuration-aware dynamics add regularized composition-and-slot adjustments to the offset, own state, initial difference, round, and peer gap. Shared and configuration terms use ridge regularization, with an unpenalized global intercept. Redundant terms mean individual coefficients depend on the chosen parameterization; assess held-out predictions and coefficient stability rather than interpreting a coefficient as causal influence.

Outer leave-one-question-out folds keep all configurations, repetitions, agents, and branches together. Inner question-held-out folds select ridge strength from `0.01, 0.1, 1, 10, 100` using one-step margin MSE. Feature centering and scaling use training data only; training weights give each question equal total weight. Classical reference models are refitted on margins, so their numbers differ from the previous stance-target analysis. No-peer drift is trained exclusively on no-peer data; the learned self/coupled models use both conditions with a delivery-presence feature. Recursive predictions use actual initial states and delivery identities/timing, but no later observed states or future text. Raw predicted margins are not clipped, so instability remains visible.

CSV/JSON exports include configuration/question/round errors, final answer and consensus metrics, correctness and reversal events, observed same-question configuration comparisons, inner validation scores, fold models, parameter variability, and residual diagnostics. Invalid-response sensitivity is reported separately and natural/forced starts cannot be mixed. Ground-truth labels enter outcome evaluation only. With ten previously inspected questions, this is exploratory internal validation; the model does not generate arguments or establish a sufficient conversational state representation.

### Question-centered deliberation report

Create a separate descriptive report that keeps questions, compositions, matched controls, and transcript context central:

```bash
MPLCONFIGDIR=/tmp/opinion-lab-matplotlib .venv/bin/python scripts/report_question_deliberation.py \
  --source results/experiments/new_exp_1_a \
  --modeling results/modeling/new_exp_1_a_system_identification \
  --output results/deliberation/new_exp_1_a_question_report
```

Open the output `index.html`. The report contains configuration synthesis, same-question figures for QQ/SS/QS/SQ, matched-control comparisons, consensus and disagreement trajectories, answer reversals/ABA oscillation, persistence, wording repetition, invalid-response denominators, and secondary correctness analysis. Each question page includes all recorded public conversations, exact delivered-message context, source-message links, and raw generated output for invalid responses. Shared initialization records are checked for exact equality across all 120 matched pairs.

Tables are exported as CSV and JSON under `tables/`. Explicit coding criteria separate automatic surface cues and normalized text reuse from semantic interpretation. Thirteen assistant-inspected, purposively selected excerpts illustrate reference, counterclaims, reuse, attribution errors, and no observable textual uptake; they are not an independently annotated sample or population prevalence estimates. A semantic review template covers every delivered-message event. Absence of a reference cue never establishes that a message was ignored, and a wording change does not establish a substantive argument revision. Existing cosine scores are reused without loading encoders. A limited modeling section reuses the saved benchmarks and links their largest errors back to transcripts; no models are refitted. Source and modeling input hashes are verified unchanged, and existing analysis directories are preserved.

The question-centered report's initial/final correctness figure uses **correct answers divided by valid answers**, with separate panels showing valid-answer coverage and counts. Invalid responses are not classified as incorrect. The initial-to-final transition figure and `initial_final_transitions.csv` retain Correct, Incorrect, and Invalid as distinct endpoint states, including correct→incorrect and incorrect→correct changes. Cell percentages use all agent endpoint pairs, and question-specific transition counts are included on each question page. Endpoint comparisons do not imply that no intermediate reversals occurred; initial and final valid-only correctness may have different denominators.

Question-centered trajectory pages now offer a repetition selector. Each run-specific figure plots exact per-agent scores for one matched discussion/control pair per composition, with categorical group states from those same conversation IDs and round indices. No averaging or overlaid repetitions are used in these question plots. Run labels are 1–3, with saved repetition IDs 0–2 shown explicitly. Group cells spell out Yes, No, Split, and Invalid. `question_plot_index.csv` records the source conversation IDs for every figure/configuration.


### Decision-time measurement protocol (v5)

New StrategyQA configurations default to `response_protocol: decision_time_v2`, `question_figure_profile: question_png`, and a 512-token total response budget (one answer token plus up to 511 explanation tokens). The answer step is constrained to single-token Yes/No candidates and records raw float32 logits/log-probabilities before candidate masking and sampling transformations. The actual sampling probabilities, selected answer, seeds, and forced-start status are saved. Explanation generation continues the same response prefix as plain text, with a separate saved seed; the application serializes answer/explanation as valid JSON, including quotes and newlines. Empty explanations and budget exhaustion are flagged without silently repairing or regenerating the response. Unsupported answer-token boundaries fail explicitly.

`decision_time` is the primary measured output; `pre_response_probe`, `before`, and `post` preserve the separate probe measurements. Dashboard stance curves, behavioral analysis, and new question PNGs use the decision-time field when available. The same `tanh(margin/2)` transformation is retained. Sentiment refers to the completed public message, not the answer-token probabilities. The current v6 pilot must be frozen separately on the collection machine; v4 data and locks remain unchanged. Do not resume v4 collection under the changed source.

To render the saved pilot in the compact PNG-only layout, without changing its old measurement or source files:

```bash
python -m llm_mas question-figures --run-id new_exp_1_a \
  --output results/deliberation/new_exp_1_a_question_png
```

The output contains an HTML index, analysis provenance JSON, and PNGs only in `figures/`. It uses historical post-response scores when decision-time measurements were not recorded; those new scores cannot be reconstructed from saved transcripts alone. Existing experiment analysis, audit, conversations, initializations, inputs and tables are retained. CSV tables remain analysis exports under `tables/`, not figure exports.


Custom experiments and pilots share the same dashboard reservation and worker-launch path. New workers claim an explicit one-use launch token, verify the reserved config, and reject folders containing scientific data or an already claimed launch. Older dashboard reservations are compared after resolving omitted defaults, rather than comparing raw JSON to the worker's resolved config. Known atomic metadata temporaries and NFS unlink placeholders do not make a fresh reservation look like a completed experiment. After updating, restart the dashboard process so custom controls and defaults are also current. Failed IDs remain preserved; use New experiment and a fresh ID. The startup fix is versioned as pilot v6 to preserve existing v5 protocol locks; custom experiments do not require freezing.
