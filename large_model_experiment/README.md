# Large model experiment: Claude vs OpenAI agents

Same design as `results/experiments/pilot_experiment_2`, with API models as the two agents:

- `reasoning_answer_v4` prompts (option A), loaded from `prompts/`
- 2 agents, up to 5 update rounds: from round 1 each response ends with `Next: continue` or `Next: finish`, and the conversation stops once both agents say finish. The no-peer control runs as many rounds as its matched peer conversation.
- natural starts
- compositions CC / OO / CO / OC (C = Claude, O = OpenAI)
- synchronous peer exchange vs a matched no-peer control that shares the same round-0 responses
- the same 4 StrategyQA questions

This folder is self-contained. It reads the project's prompts, dataset and analysis code and changes none of them.

## Setup (once)

```bash
.venv/bin/pip install -r large_model_experiment/requirements.txt
# then put your keys in large_model_experiment/.env (gitignored; template in .env.example)
```

## Run (from the project root)

```bash
.venv/bin/python large_model_experiment/run_experiment.py --dry-run                      # plan + first prompt, no calls
.venv/bin/python large_model_experiment/run_experiment.py --mock --run-id mock_check     # offline pipeline check
.venv/bin/python large_model_experiment/run_experiment.py --run-id claude_openai_1       # real run: 352 API calls
.venv/bin/python large_model_experiment/run_experiment.py --report-only --run-id claude_openai_1   # rebuild tables/figures
```

- **Resuming:** rerunning the same command with the same `--run-id` resumes. Finished conversations and round-0 snapshots are reused.
- **Parallelism:** `--workers N` (default 4) runs question/composition pairs in parallel.
- **Models:** set in `.env` (`CLAUDE_MODEL`, `OPENAI_MODEL`) or with `--claude-model` / `--openai-model`.

## Outputs: `large_model_experiment/results/<run-id>/`

The output layout matches the local runs:

- `config.json`, `conversations/`, `initializations/`, `inputs/`, `environment.json`
- `tables/`: the same 18 CSVs
- `analysis/`, `summary.json`, `RESULTS.md`, `index.html`
- `figures/question_*.png`, one per question. Each row is one composition, with four panels:
  - sentiment with peers
  - sentiment without peers
  - each agent's explicit answer per round
  - group answer state per round

## Differences from the local runs (forced by the APIs)

- **No stance:** the APIs return no answer-token logits, so the stance columns in the tables are empty.
- **Answer source:** the answer is the `Answer: Yes/No` line the model writes after its reasoning. A response without one is recorded as `no_answer_line`; a refusal as `refusal`. Both show as Invalid in the figures.
- **No copy blocking:** the APIs give no access to token scores during generation, so own-message copy blocking is not applied.
- **Decoding:**
  - Current Claude models reject sampling parameters, so Claude runs with adaptive thinking at `CLAUDE_EFFORT` (default `low`).
  - OpenAI gets `temperature=0` and `reasoning_effort=none`. If the model rejects either, it is dropped and recorded in `environment.json`.
  - Neither provider guarantees identical output on reruns.
- **Peer messages:** when a peer message and the turn instruction arrive together, Claude receives them as one user turn with two separate text blocks. The API requires alternating roles. OpenAI receives them as two user messages, as the local models did.

The v4 prompts require checking delivered objections before finishing. Only a terminal Next line after the answer controls API stopping; embedded Next controls are removed from the delivered reasoning and retained in raw output.
