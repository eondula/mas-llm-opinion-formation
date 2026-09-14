"""Decision sheet derived from recorded choices; no invented collective rationale."""
DEFAULT_BRIEF = (
    'Submit one team recommendation choosing one of the two options for the scenario. '
    'Explain why its benefits justify its main drawback. Address a specific concern '
    'raised by another agent when peer messages are available. If you cannot agree, '
    'record the competing recommendations and the unresolved tradeoff.')


def decision_sheet(rows, scenario, agents, schedule):
    last_round = max(row['round'] for row in rows)
    final = [row for row in rows if row['round'] == last_round]
    choices = [row['strategy'] for row in final]
    valid = len(final) == agents and all(choice in (1, 2) for choice in choices)
    agreed = valid and len(set(choices)) == 1
    selected = choices[0] if agreed else None
    return dict(round=last_round,
                status='shared_choice' if agreed else 'unresolved' if valid else 'incomplete_or_invalid',
                strategy=selected,
                recommendation=scenario[f'strategy_{selected}']['candidate'] if selected else None,
                peer_exchange=schedule != 'no_peer',
                note='Compiled from individual recorded choices and reasons; no separate collective ratification was performed.',
                positions=[dict(agent=r['agent'], strategy=r['strategy'],
                                recommendation=scenario[f"strategy_{r['strategy']}"]['candidate'] if r['strategy'] in (1, 2) else None,
                                reason=r['argument'], choice_status=r.get('choice_status')) for r in final])
