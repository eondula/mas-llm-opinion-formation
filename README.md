# Opinion formation experiments

Run repeated StrategyQA experiments with language-model agents, then analyze opinion changes, correctness, disagreement, consensus and message cosine similarity. One configuration-driven engine supports independent reasoning and synchronous or sequential deliberation.

## Start the experiment page

From the repository root:

```bash
source .venv/bin/activate
python scripts/prepare_pilot.py
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

On the Experiments page click **Load controlled pilot**, optionally enter an experiment ID, then **Run experiment**. The frozen preset runs 10 balanced training questions across QQ, SS, QS and SQ, with 3 repetitions and 5 updates, natural starts and synchronous/no-peer branches sharing initialization. It produces 120 pairs, 240 conversations, 2,880 logical response rows and 2,640 generated responses. The stance audit runs first and stops collection if repeatability fails. All reports remain in the experiment directory.

Scientific settings are locked while this preset is active. **Custom experiment** restores editable controls. The saved configuration is [studies/pilot_experiment.json](studies/pilot_experiment.json). Run `python scripts/prepare_pilot.py` after cloning to restore the local question manifest required by the frozen protocol. The equivalent CLI launch is `python -m llm_mas run --config studies/pilot_experiment.json`.

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
