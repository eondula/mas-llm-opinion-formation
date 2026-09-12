"""Explicit schema for autonomous public choices; no wording heuristics."""
import json

def parse(text):
    try:
        value=json.loads(text)
    except (ValueError,TypeError):
        return None, text, 'invalid_json'
    if not isinstance(value,dict) or type(value.get('strategy')) is not int or value['strategy'] not in (1,2) or not isinstance(value.get('explanation'),str) or not value['explanation'].strip():
        return None, text, 'invalid_schema'
    return value['strategy'],value['explanation'],'valid'

def instruction(texts,scenario):
    q=texts['stance_probe'].format(candidate_1=scenario['strategy_1']['candidate'],candidate_2=scenario['strategy_2']['candidate'])
    q=q[:q.index('Return the exact candidate statement')]
    return q+'Return only a JSON object with "strategy" (integer 1 or 2) and "explanation" (a short paragraph). Do not use markdown fences.'
