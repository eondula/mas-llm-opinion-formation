"""Explicit schema for autonomous public choices; no wording heuristics."""
import json

PARSER_VERSION = "3-missing-final-object-brace"

def parse(text):
    candidate = text.strip() if isinstance(text,str) else text
    # Accept a single enclosing Markdown fence, not arbitrary prose or JSON repair.
    if isinstance(candidate,str):
        lines=candidate.splitlines()
        if len(lines)>=3 and lines[0].strip().lower() in ('```json','```') and lines[-1].strip()=='```':
            candidate='\n'.join(lines[1:-1])
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
    candidate = text.strip()
    lines = candidate.splitlines()
    if len(lines) >= 3 and lines[0].strip().lower() in ('```json', '```') and lines[-1].strip() == '```':
        candidate = '\n'.join(lines[1:-1])
    try:
        json.loads(candidate)
    except ValueError:
        return 'missing_final_object_brace'
    return None
