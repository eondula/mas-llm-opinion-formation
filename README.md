# Opinion Lab

Run language model agent conversations through a local dashboard or the terminal.

## Hardware requirements

- Windows or Linux with Python 3.11 or newer and Git.
- An NVIDIA GPU with a compatible driver and CUDA-enabled PyTorch for accelerated runs. The mechanistic pilots were run on an RTX 3090 with 24 GB VRAM; this is a tested configuration, not a minimum requirement.
- Enough RAM and GPU memory for the selected checkpoints and conversation histories. Memory needs increase with model size, context length and float32 execution.
- Internet access and disk space for model downloads on first use. Models are cached locally.

CPU execution is available but slower. Full study commands below explicitly select CUDA.

## Install

Clone the repository and enter its directory:

```sh
git clone https://github.com/eondula/mas-llm-opinion-formation.git
cd mas-llm-opinion-formation
```

**Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows PowerShell:**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch build compatible with your GPU driver. Check GPU availability after installation:

```sh
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda)"
```

Run the remaining commands from the repository root with the environment activated. They work in both Linux terminals and Windows PowerShell.

## Run the dashboard

```sh
python -m llm_mas dashboard
```

Open **http://127.0.0.1:8766**. Select the models, scenario and conversation settings, then start a run. Use **Quick preview** for a single conversation and **Study** for a sweep. Select CUDA for GPU execution. Saved runs and figures are accessible from the dashboard.

For interactive testing, select **Discussion prompt → Discussion then decision**. After initialization, each agent discusses the available messages before reassessing its choice. Existing controls still apply. Scored choice uses endorsement scores after discussion; generated choice adds a separate JSON decision call. The dedicated prompt is `prompts/discussion_then_decision.txt`. Saved response records include the discussion generation, pre-discussion scores, decision scores and post-response scores. Original, Open discussion and Topic discussion remain available.

Use the terminal presets below to run the exact full behavioral, fixed and reversed study grids.

Select **Run mode → Joint decision task** to give agents a shared recommendation task. Edit the team brief and choose fixed or adaptive stopping; other controls remain available. The decision sheet shows each agent's choice and reason, plus shared choice or unresolved disagreement. It does not force consensus or add a vote. The task prompt is `prompts/joint_task.txt`; the final sheet is saved as `joint_decision.json` in the run folder.

## Run from the terminal

Start with a short conversation:

```sh
python -m llm_mas preview --scenario storage --rounds 3 --device cuda --run-id preview_check
```

Run a small study before launching the full presets:

```sh
python -m llm_mas study --config studies/full_behavioral_smoke.json --device cuda --run-id full_smoke
```

Use a new run ID for each new run. Add `--dry-run` to a study command to inspect its tasks without loading models.

## Run the three studies

The presets use Sensor and Storage, two agents, Qwen2.5-1.5B, Qwen2.5-3B and SmolLM2-1.7B assignments, and both scored and generated choice. Run jobs sequentially when sharing one GPU.

| Study | Conversations | Observation |
|---|---:|---|
| Behavioral | 480 | Adaptive stopping, up to 15 updates; natural and forced starts, interactive and no-peer schedules |
| Full fixed | 120 | Natural interactive conditions continued through all 15 updates |
| Reversal | 120 | Natural interactive conditions with reversed alternative display and candidate-block order; adaptive stopping |

**Behavioral:**

```sh
python -m llm_mas study --config studies/full_behavioral.json --device cuda --run-id full_behavioral_new
```

**Full fixed:**

```sh
python -m llm_mas study --config studies/full_behavioral_fixed.json --device cuda --run-id full_fixed_new
```

**Reversal:**

```sh
python -m llm_mas study --config studies/full_behavioral_reversed.json --device cuda --run-id full_reversed_new
```

Results are saved in `results/studies/<run-id>/`: conversation JSON, prompts and configuration, figures, and CSV measurements under `tables/`. Save new research figures as PNG with CSV data sufficient to reproduce them; do not create PDF copies.

Resume an interrupted run with unchanged code and configuration, or regenerate its figures from saved data:

```sh
python -m llm_mas resume --run-id full_behavioral_new
python -m llm_mas figures --run-id full_behavioral_new
```

For **Joint decision task**, select **Agent-negotiated closure** to stop through explicit proposal and unanimous confirmation. Closure uses separate greedy control calls and no similarity threshold. All agents review the same completed round; a veto or invalid signal continues discussion. Closure may preserve disagreement. The dashboard shows control reasons, and raw controls are saved in termination records. No peer disables this extra exchange. The maximum rounds remain a budget safeguard. Edit `prompts/closure.txt` to change the control instruction.
