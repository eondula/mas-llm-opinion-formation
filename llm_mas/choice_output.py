"""Explicit schema for autonomous public choices; no wording heuristics."""
import json

PARSER_VERSION = "4-unclosed-json-fence"

def _candidate(text):
    candidate = text.strip() if isinstance(text, str) else text
    open_fence = False
    if isinstance(candidate, str):
        lines = candidate.splitlines()
        if lines and lines[0].strip().lower() in ('```json', '```'):
            closed = len(lines) > 1 and lines[-1].strip() == '```'
            candidate = '\n'.join(lines[1:-1] if closed else lines[1:]).strip()
            open_fence = not closed
    return candidate, open_fence


def parse(text):
    candidate, _ = _candidate(text)
    try:
        value=json.loads(candidate)
    except (ValueError,TypeError):
        # Recover only a complete object body missing its final closing brace.
        # No strategy inference, truncated strings, prose removal, or score repair.
        if not isinstance(candidate, str) or not candidate.startswith('{') or not candidate.endswith('"'):
            return None, text, 'invalid_json'
        try:
            value = json.loads(candidate + '}')
        except ValueError:
            return None, text, 'invalid_json'
        if not isinstance(value, dict) or set(value) != {'strategy', 'explanation'}:
            return None, text, 'invalid_schema'
    if not isinstance(value,dict) or type(value.get('strategy')) is not int or value['strategy'] not in (1,2) or not isinstance(value.get('explanation'),str) or not value['explanation'].strip():
        return None, text, 'invalid_schema'
    return value['strategy'],value['explanation'],'valid'

def instruction(texts,scenario):
    q=texts['stance_probe'].format(candidate_1=scenario['strategy_1']['candidate'],candidate_2=scenario['strategy_2']['candidate'])
    q=q[:q.index('Return the exact candidate statement')]
    return q+'Return only a JSON object with "strategy" (integer 1 or 2) and "explanation" (a short paragraph). Do not use markdown fences.'


def display_message(row):
    """Format parseable generated text without changing the recorded decision."""
    raw = row.get('message', '')
    if 'answer' in row:
        if row.get('next_action') in ('continue','finish') and row.get('answer') in ('Yes','No'):
            from .decision_generation import strip_next_controls
            reasoning=strip_next_controls(row.get('argument') or '')
            return (reasoning+'\n' if reasoning else '')+'Answer: '+row['answer']+'\nNext: '+row['next_action']
        # Reasoning-first messages already contain the recorded answer and Next
        # signal. Keep their exact public order instead of rebuilding answer-first.
        if (row.get('decision_time') or {}).get('stage') == 'answer_token_after_reasoning' and raw:
            return raw
        return row['answer']+'\n'+row.get('argument', '') if row.get('answer') in ('Yes','No') else raw
    strategy, explanation, status = parse(raw)
    if status == 'valid':
        text = f'Strategy {strategy}\n{explanation}'
        if row.get('strategy') != strategy:
            text += '\n[Display only: saved choice is invalid or differs; record unchanged.]'
        return text
    if row.get('choice_status') == 'valid' and row.get('strategy') in (1, 2) and isinstance(row.get('argument'), str):
        return f"Strategy {row['strategy']}\n{row['argument']}"
    return raw


def recovery(text):
    """Audit whether a valid choice needed the single-brace recovery."""
    if parse(text)[2] != 'valid':
        return None
    candidate, open_fence = _candidate(text)
    missing_brace = False
    try:
        json.loads(candidate)
    except ValueError:
        missing_brace = True
    if open_fence and missing_brace:
        return 'missing_final_object_brace_and_markdown_fence'
    if open_fence:
        return 'missing_closing_markdown_fence'
    return 'missing_final_object_brace' if missing_brace else None
