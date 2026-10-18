"""Claude vs OpenAI agents on the reasoning_answer_v3 protocol: same prompts, questions and paired design.

Run from the project root:
    .venv/bin/python large_model_experiment/run_experiment.py --run-id claude_openai_1
    .venv/bin/python large_model_experiment/run_experiment.py --mock --run-id mock_check    # offline check, no API calls
    .venv/bin/python large_model_experiment/run_experiment.py --dry-run                     # print the plan and first prompt
    .venv/bin/python large_model_experiment/run_experiment.py --report-only --run-id claude_openai_1

Design (mirrors results/experiments/pilot_experiment_2): 2 agents, 5 update rounds, natural starts,
compositions CC / OO / CO / OC, synchronous peer exchange vs matched no-peer control sharing the
same round-0 responses, the 4 StrategyQA questions. Differences from the local pipeline, all
forced by the APIs: no answer-token logits (no stance), the answer is the "Answer: Yes/No" line the
model writes, no own-message copy blocking, and decoding is the provider's (temperature 0 where
the model accepts it; current Claude models do not accept sampling parameters).
"""
import argparse
import json
import os
import platform
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from llm_mas.config import digest  # noqa: E402
from llm_mas.datasets import select, labels as dataset_labels  # noqa: E402
from llm_mas.decision_generation import split_reasoning  # noqa: E402
from llm_mas.engine import names_for  # noqa: E402
from llm_mas.prompts import templates  # noqa: E402
from llm_mas.question_policy import QuestionResponsePolicy  # noqa: E402
from llm_mas.storage import write, read  # noqa: E402
from llm_mas.trajectories import SCHEMA_VERSION  # noqa: E402
import providers  # noqa: E402

QUESTION_IDS = ['7d310e9ebc2025febdd6', 'a905e30e6cfc76377348', 'd692a072dbe7acefbc6b', '315575d5cbdd1d154744']
DATASET = dict(name='strategyqa', path='data/strategyqa', partition='train', split='train', split_seed=42)
PROTOCOL = 'reasoning_answer_v3_api'


def load_env(path):
    """Minimal KEY=VALUE reader; values already set in the shell take precedence."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        value = value.strip().strip('"').strip("'")
        if value and key.strip() not in os.environ:
            os.environ[key.strip()] = value


def env(name, default=None):
    value = os.environ.get(name, '').strip()
    return value if value else default


def build_agents(args):
    if args.mock:
        return {'mock-claude': providers.MockAgent('mock-claude', 'Yes'), 'mock-openai': providers.MockAgent('mock-openai', 'No')}, \
            {'mock-claude': 'C', 'mock-openai': 'O'}
    claude_model = args.claude_model or env('CLAUDE_MODEL', 'claude-opus-5-5')
    openai_model = args.openai_model or env('OPENAI_MODEL', 'gpt-6-astra')
    max_tokens = int(env('MAX_OUTPUT_TOKENS', '4096'))
    temperature = env('OPENAI_TEMPERATURE', '0')
    if not args.dry_run:
        missing = [k for k in ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY') if not env(k)]
        if missing:
            raise SystemExit('Missing '+', '.join(missing)+f' - add them to {HERE/".env"} (see .env.example)')
    agents = {} if args.dry_run else {
        claude_model: providers.ClaudeAgent(claude_model, env('CLAUDE_EFFORT', 'low'), max_tokens),
        openai_model: providers.OpenAIAgent(openai_model, env('OPENAI_REASONING_EFFORT', 'none'),
                                            float(temperature) if temperature.lower() != 'none' else None, max_tokens)}
    return agents, {claude_model: 'C', openai_model: 'O'}


class Experiment:
    def __init__(self, path, config, tasks, agents, texts):
        self.path, self.config, self.tasks, self.agents, self.texts = path, config, tasks, agents, texts
        self.lock = threading.Lock()
        self.completed, self.total = 0, 0

    def status(self, phase, **extra):
        with self.lock:
            write(self.path/'status.json', dict(phase=phase, completed=self.completed, total=self.total, updated=datetime.now().isoformat(), **extra))

    def policy(self, task, models):
        cfg = dict(response_protocol='reasoning_answer_v3', rounds=self.config['rounds'], initialization='natural',
                   paired_control=True, agents=2, schedule='synchronous', decoding='greedy', copy_block_ngram=0)
        return QuestionResponsePolicy(cfg, task, self.texts, names_for(2), models)

    def respond(self, policy, history, t, i, deliveries, models, names):
        prompt = policy.response_prompt(t, i, deliveries)
        decision_history = deepcopy(history)
        agent = self.agents[models[i]]
        result = agent.generate(history[0]['content'], history[1:]+[dict(role='user', content=prompt)])
        parsed = split_reasoning(result['text'])
        answer = parsed['stated_answer']
        status = 'refusal' if result['refusal'] else 'valid' if answer else 'no_answer_line'
        reasoning = parsed['reasoning']
        message = (reasoning+'\n' if reasoning else '')+'Answer: '+answer if status == 'valid' else result['text']
        history.append(dict(role='assistant', content=message))
        with self.lock:
            self.completed += 1
        print(f'  {names[i]} {models[i]} r{t}: {answer or status}')
        empty = dict(stance=None, margin=None)
        return dict(response_id=f'r{t}_a{i}', question_id=policy.task['id'], round=t, agent=names[i], model=models[i],
                    answer=answer if status == 'valid' else None, strategy=1 if answer == 'Yes' else 2 if answer == 'No' else None,
                    choice_status=status, decision_time=None, stance_source='none', pre_response_probe=None, forced=False,
                    assigned_initial_answer=None, before=empty, decision=empty, post=empty, message=message, argument=reasoning,
                    generation=dict(result, instruction=prompt, raw_output=result['text'], answer_marker_found=parsed['marker_found'],
                                    answer_before_reasoning=parsed['answer_before_reasoning'], reasoning=reasoning,
                                    choice_parser_version='stated-answer-line-v1'),
                    deliveries=deliveries, decision_history=decision_history, history=deepcopy(history))

    def converse(self, task, models, schedule, rounds, initial_rows=None):
        names = names_for(2)
        policy = self.policy(task, models)
        histories = [policy.initial(i) for i in range(2)]
        rows, latest = [], [None, None]
        if initial_rows is not None:
            rows = deepcopy(initial_rows)
            histories = [deepcopy(r['history']) for r in rows]
            latest = list(rows)
        for t in range(1 if initial_rows is not None else 0, rounds+1):
            old_rows = list(latest)
            for i in range(2):
                deliveries = []
                if t and schedule != 'no_peer':
                    for j in range(2):
                        if j == i:
                            continue
                        source = old_rows[j]
                        deliveries.append(dict(speaker=names[j], receiver=names[i], message=source['message'], round=t,
                                               source_id=source['response_id'], source_round=source['round']))
                        histories[i].append(dict(role='user', content=policy.texts['peer_message'].format(
                            speaker=names[j], message=source['message'], source_round=source['round'])))
                row = self.respond(policy, histories[i], t, i, deliveries, models, names)
                latest[i] = row
                rows.append(row)
        return rows

    def pair(self, ci, models, qi, task, protocol_hash):
        coordinates = dict(question_id=task['id'], models=list(models), agents=2, repetition=0, initialization='natural', protocol_hash=protocol_hash)
        pair_id = digest(coordinates)[:24]
        snapshot_file = self.path/'initializations'/f'{pair_id}.json'
        targets = {s: self.path/'conversations'/f'{ci*2+si:03d}_{qi:03d}.json' for si, s in enumerate(('synchronous', 'no_peer'))}
        if all(t.exists() for t in targets.values()):
            return
        if snapshot_file.exists():
            snapshot = read(snapshot_file)
        else:
            print(f'[{task["id"]}] {"/".join(models)}: round 0')
            rows = self.converse(task, models, 'synchronous', 0)
            snapshot = dict(pair_id=pair_id, coordinates=coordinates, rows=rows, initialization_hash=digest(rows))
            write(snapshot_file, snapshot)
        for schedule, target in targets.items():
            if target.exists():
                continue
            print(f'[{task["id"]}] {"/".join(models)}: {schedule}')
            rows = self.converse(task, models, schedule, self.config['rounds'], initial_rows=snapshot['rows'])
            record = dict(schema_version=SCHEMA_VERSION, config=dict(self.config, models=list(models), schedule=schedule),
                          scenario=task, question_id=task['id'], repetition=0,
                          measurement_stages=dict(before='none', decision='stated_answer_line_after_reasoning', post='none', primary='explicit_answer'),
                          rows=rows, termination=dict(reason='max_rounds', round=self.config['rounds'], policy='fixed'),
                          pair_id=pair_id, initialization_hash=snapshot['initialization_hash'], control_design='shared_initialization',
                          protocol_hash=protocol_hash, task=task, dataset=self.config['dataset'],
                          research_split=self.config['research_splits'][task['id']])
            write(target, record)
        self.status('conversations')

    def run(self, workers):
        protocol_hash = digest(dict(protocol=PROTOCOL, prompts={k: v for k, v in self.texts.items() if k.startswith('qa_')},
                                    agents={m: a.settings() for m, a in self.agents.items()}, rounds=self.config['rounds']))
        compositions = self.config['model_configurations']
        jobs = [(ci, models, qi, task) for ci, models in enumerate(compositions) for qi, task in enumerate(self.tasks)]
        self.total = len(jobs)*(2+2*2*self.config['rounds'])
        self.status('conversations')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(self.pair, *job, protocol_hash) for job in jobs]:
                future.result()
        write(self.path/'environment.json', dict(python=platform.python_version(), platform=platform.platform(),
              packages=package_versions(), agents={m: a.settings() for m, a in self.agents.items()}, protocol=PROTOCOL, protocol_hash=protocol_hash,
              stance='not measured: API models expose no answer-token logits',
              copy_block='not applied: providers do not expose a logits processor'))


def package_versions():
    import importlib.metadata
    out = {}
    for name in ('anthropic', 'openai', 'torch', 'transformers', 'matplotlib'):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def postprocess(path, config, figure_labels):
    """Same post-run analysis as llm_mas/runner.py, with stance-free figures."""
    from llm_mas.evaluation import evaluate
    from llm_mas.sentiment import measure_sentiment
    from llm_mas.message_analysis import analyze as message_analysis
    from llm_mas.behavioral import analyze as behavioral
    from llm_mas import behavioral_reports, question_figures
    import plots
    evaluation_labels = dataset_labels(ROOT, config['dataset'])
    write(path/'inputs/evaluation_labels.json', {q: evaluation_labels[q] for q in config['question_ids']})
    evaluate(path, evaluation_labels)
    cache = {}
    for target in sorted((path/'conversations').glob('*.json')):
        record = json.loads(target.read_text())
        for row in record['rows']:
            if row['message'] not in cache:
                cache[row['message']] = measure_sentiment(row['message'])['sentiment_score']
            row['sentiment'] = cache[row['message']]
        write(target, record)
    message_analysis(path)
    behavioral(path)
    original = question_figures.report
    question_figures.report = lambda p, data=None, output=None: plots.report(p, figure_labels, data)
    try:
        behavioral_reports.generate(path, reanalyze=False)
    finally:
        question_figures.report = original


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-id', default=datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--question-ids', nargs='+', default=QUESTION_IDS)
    parser.add_argument('--claude-model'); parser.add_argument('--openai-model')
    parser.add_argument('--workers', type=int, default=4, help='question/composition pairs run in parallel')
    parser.add_argument('--mock', action='store_true', help='offline deterministic agents; no API calls')
    parser.add_argument('--dry-run', action='store_true', help='print the plan and first prompt, then exit')
    parser.add_argument('--report-only', action='store_true', help='rerun analysis and figures on saved conversations')
    args = parser.parse_args()
    load_env(HERE/'.env')
    path = HERE/'results'/args.run_id
    if args.report_only:
        config = read(path/'config.json')
        postprocess(path, config, read(path/'inputs/figure_labels.json'))
        print('Report rebuilt:', path/'index.html')
        return
    agents, figure_labels = build_agents(args)
    claude, openai_model = list(figure_labels)
    tasks, dataset, assignments = select(ROOT, dict(dataset=DATASET, question_ids=args.question_ids, question_balance='equal'))
    compositions = [[claude, claude], [openai_model, openai_model], [claude, openai_model], [openai_model, claude]]
    config = dict(mode='experiment', output_layout='experiment', models=compositions[0], agents=2, rounds=args.rounds,
                  initialization='natural', schedule='synchronous', decoding='provider_default_temperature_0_where_supported',
                  repetitions=1, question_count=len(tasks), question_ids=[t['id'] for t in tasks], question_balance='equal',
                  paired_control=True, conditions=[dict(schedule='synchronous'), dict(schedule='no_peer')],
                  model_configurations=compositions, agent_configuration='heterogeneous', response_protocol=PROTOCOL,
                  copy_block_ngram=0, max_argument_tokens=int(env('MAX_OUTPUT_TOKENS', '4096')), sentiment=True, message_similarity=True,
                  question_figure_profile='question_png', stopping='fixed', prompt_style='question', argument_mode='combined_reasoning',
                  decision_mode='generated', dataset=dataset, tasks=tasks, scenarios=tasks,
                  research_splits={t['id']: assignments[t['id']] for t in tasks}, mock=args.mock)
    texts = templates()
    print(f'Run {args.run_id}: {len(tasks)} questions x {len(compositions)} compositions x 2 branches, {args.rounds} rounds')
    print(f'Models: C = {claude}, O = {openai_model}; API calls: {len(tasks)*len(compositions)*(2+4*args.rounds)}')
    if args.dry_run:
        experiment = Experiment(path, config, tasks, {}, texts)
        policy = experiment.policy(tasks[0], compositions[0])
        history = policy.initial(0)
        print('\n--- system ---\n'+history[0]['content']+'\n--- user ---\n'+history[1]['content']+'\n--- user ---\n'+policy.response_prompt(0, 0, []))
        return
    if (path/'config.json').exists():
        saved = read(path/'config.json')
        if {k: v for k, v in saved.items()} != json.loads(json.dumps(config)):
            raise SystemExit(f'{path} exists with a different configuration; choose a new --run-id')
        print('Resuming saved run')
    for name, value in (('config.json', config), ('inputs/tasks.json', tasks), ('inputs/dataset.json', dataset),
                        ('inputs/prompts.json', {k: v for k, v in texts.items() if k.startswith('qa_')}), ('inputs/figure_labels.json', figure_labels)):
        write(path/name, value)
    experiment = Experiment(path, config, tasks, agents, texts)
    try:
        experiment.run(args.workers)
        experiment.status('analysis')
        postprocess(path, config, figure_labels)
        experiment.status('complete')
    except BaseException as error:
        experiment.status('failed', error=str(error))
        (path/'error.log').write_text(traceback.format_exc())
        print(f'Stopped: {error}. Rerun the same command to resume from saved conversations.')
        raise
    print('Done:', path/'index.html')


if __name__ == '__main__':
    main()
