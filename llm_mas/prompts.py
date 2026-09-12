from .config import ROOT

def templates():
    names = ('system', 'scenario', 'starting_assessment', 'peer_message', 'stance_probe', 'generate_argument', 'select_argument', 'conversational_argument', 'combined_fixed', 'combined_reasoning')
    return {name: (ROOT / 'prompts' / f'{name}.txt').read_text() for name in names}

def initial(scenario, texts, order='strategy_1_first', agents=2, persona=None):
    a, b = scenario['strategy_1'], scenario['strategy_2']
    if order == 'strategy_2_first': a, b = b, a
    system = texts['system']
    if persona is not None:
        role = scenario['persona_roles'][f'strategy_{persona}']
        system += '\n\nYour role: ' + role['role_name'] + '.\n' + role['priority']
    if agents > 2:
        system = system.replace('another artificial agent', 'other artificial agents')
    return [{'role':'system', 'content':system}, {'role':'user', 'content':texts['scenario'].format(
        title=scenario['title'], a1=a['property_1'], a2=a['property_2'], b1=b['property_1'], b2=b['property_2'])}]


def for_condition(texts, config, scenario):
    """Resolve the scoring instruction for this experimental condition."""
    if config.get('argument_mode') not in ('combined_fixed', 'combined_reasoning'):
        return texts
    # Keep candidate placeholders for validation variants; escape literal braces in arguments.
    template = texts[config['argument_mode']]
    for k in (1, 2):
        pool = '\n'.join('- '+a for a in scenario[f'strategy_{k}']['arguments'])
        template = template.replace('{arguments_'+str(k)+'}', pool.replace('{','{{').replace('}','}}'))
    if config.get('reverse_candidate_blocks'):
        a=template.index('STRATEGY 1\n'); b=template.index('STRATEGY 2\n'); end=template.index('Return the exact candidate statement')
        template=template[:a]+template[b:end].rstrip()+'\n\n'+template[a:b].rstrip()+'\n\n'+template[end:]
    return dict(texts, stance_probe=template)
