from .config import ROOT

def templates():
    names = ('system', 'scenario', 'starting_assessment', 'peer_message', 'stance_probe', 'generate_argument', 'select_argument', 'conversational_argument', 'combined_fixed', 'combined_reasoning', 'open_discussion', 'topic_discussion', 'topic_system', 'topic_probe', 'discussion_then_decision', 'joint_task', 'closure')
    return {name: (ROOT / 'prompts' / f'{name}.txt').read_text() for name in names}

def initial(scenario, texts, order='strategy_1_first', agents=2, persona=None, agent_name=None, prompt_style="original", joint_task_brief=None):
    a, b = scenario['strategy_1'], scenario['strategy_2']
    if order == 'strategy_2_first': a, b = b, a
    system = texts['topic_system'] if prompt_style == 'topic_discussion' else texts['system']
    scenario_template = texts['scenario']
    if prompt_style in ('open_discussion', 'discussion_then_decision'):
        system = system.replace('Do not introduce additional assumptions.', 'You may discuss hypothetical priorities if explicitly identified as hypothetical; do not introduce new task facts.')
        scenario_template = scenario_template.replace('No other information should be considered.', 'Keep these task facts fixed. You may ask questions or discuss explicitly hypothetical priorities.')
    if joint_task_brief is not None:
        system += '\n\n'+texts['joint_task'].format(brief=joint_task_brief)
    if agent_name is not None:
        names = [f'Agent {chr(65+i)}' if i < 26 else f'Agent {i+1}' for i in range(agents)]
        system += '\nYour name is '+agent_name+'. Other participants: '+', '.join(name for name in names if name != agent_name)+'. Use these names when referring to participants.'
    if persona is not None:
        role = scenario['persona_roles'][f'strategy_{persona}']
        system += '\n\nYour role: ' + role['role_name'] + '.\n' + role['priority']
    if agents > 2:
        system = system.replace('another artificial agent', 'other artificial agents')
    if prompt_style == 'topic_discussion':
        topic = scenario.get('topic', scenario['title'])
        content = 'Topic: '+topic+'\n\nOption A: '+a['candidate']+'\nOption B: '+b['candidate']
        return [{'role':'system','content':system}, {'role':'user','content':content}]
    if scenario.get('context'):
        # Escape braces because the scenario template is formatted below.
        scenario_template = scenario['context'].replace('{', '{{').replace('}', '}}') + '\n\n' + scenario_template
    return [{'role':'system', 'content':system}, {'role':'user', 'content':scenario_template.format(
        title=scenario['title'], a1=a['property_1'], a2=a['property_2'], b1=b['property_1'], b2=b['property_2'])}]


def for_condition(texts, config, scenario):
    """Resolve the scoring instruction for this experimental condition."""
    if config.get('argument_mode') not in ('combined_fixed', 'combined_reasoning'):
        return texts
    # Keep candidate placeholders for validation variants; escape literal braces in arguments.
    template = texts['topic_probe'] if config.get('prompt_style') == 'topic_discussion' else texts[config['argument_mode']]
    for k in (1, 2):
        pool = '\n'.join('- '+a for a in scenario[f'strategy_{k}']['arguments'])
        template = template.replace('{arguments_'+str(k)+'}', pool.replace('{','{{').replace('}','}}'))
    if config.get('reverse_candidate_blocks'):
        a=template.index('STRATEGY 1\n'); b=template.index('STRATEGY 2\n'); end=template.index('Return the exact candidate statement')
        template=template[:a]+template[b:end].rstrip()+'\n\n'+template[a:b].rstrip()+'\n\n'+template[end:]
    return dict(texts, stance_probe=template)
