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

The v4 CUDA preset uses float32 weights for compatibility with NVIDIA GPUs that lack bfloat16 support; MPS retains bfloat16. These are separate protocols and should remain separate analysis conditions. Actual checkpoint and tokenizer commits remain pinned. Unavailable devices fail explicitly rather than falling back to CPU. A CUDA build of PyTorch and a GPU allocation are required; login nodes may have no CUDA device even when the cluster has GPUs.

Equivalent CLI launch: `python -m llm_mas run --config studies/pilot_cuda_v4.json` or `studies/pilot_mps_v4.json`. The earlier v1/v2/v3 configurations and locks are retained for provenance; this release uses v4 to keep its changed execution source separate from existing batches. The updated runner rejects collection or resume under retired versions. Finish existing batches with their original checkout. Restart an old dashboard process after pulling so it loads the new manifest lookup.

The dashboard uses shared border-box sizing and flexible grids, with a single-column layout on smaller viewports. Wide summary tables scroll inside labeled, keyboard-focusable regions. Skip links, visible keyboard focus, labeled progress, chart descriptions, touch targets and reduced-motion preferences follow [W3C reflow](https://www.w3.org/WAI/WCAG22/Understanding/reflow.html), [focus](https://www.w3.org/WAI/WCAG22/Understanding/focus-visible.html) and [target-size](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html) guidance. Automated structural checks do not establish full WCAG conformance.

## Background execution and reconnecting

**Finish any existing foreground pilot before pulling this update.** Changing source files during a frozen collection invalidates that batch. New background behavior cannot be applied retroactively to an already-running foreground process.

After updating and freezing the v4 protocol on your GPU node, start the dashboard without tying it to an SSH terminal:

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
  index.html, REPORT.md, summary.json, RESULTS.md
  inputs/                             # dataset selection, prompts, evaluator labels
  conversations/                      # messages, raw measurements and delivery provenance
  analysis/behavioral.json
  analysis/evaluation.json, analysis/message_similarity.json
  tables/                             # summary, turns, trajectories, groups, controls
  figures/                            # indexed PNG/PDF figures and plotted-data CSVs
    behavioral/                       # correctness, reversals, stability, groups, peers
```

Correctness labels are used only after simulation. The original summed candidate log-likelihood scorer is retained: margin = log P(Yes) − log P(No), stance = tanh(margin / 2). Generated answers and measured stance remain separate.

Saved HTML reports include metric definitions, denominators and detailed exports. Use `python -m llm_mas dataset info` to inspect dataset and split provenance.

## Terminal execution

```bash
python -m llm_mas run --config studies/experiment.json --dry-run
python -m llm_mas run --config studies/experiment.json --repetitions 10 --run-id experiment_001
```

The preset uses the new experiment directory. Every future StrategyQA run generates the full report, detailed figures, CSV/JSON tables and refreshed ZIP automatically. Cross-experiment reports are saved in `results/experiments/comparisons/`. Change question count, models, rounds, conditions and decoding in the JSON configuration. Regenerate complete reports for saved StrategyQA runs without any model inference:

```bash
python -m llm_mas reports --run-id experiment_001
python -m llm_mas reports --all
```

## Preservation and validation

The agent conversation dashboard remains active alongside the new experiment page. Documentation, tests, archived implementations and saved research runs remain local and are excluded from new commits.

```bash
MPLCONFIGDIR=/tmp/opinion-lab-matplotlib python -m unittest discover -s tests/refactor
```
