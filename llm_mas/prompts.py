from .config import ROOT

def templates():
    names = ('system', 'scenario', 'starting_assessment', 'peer_message', 'stance_probe', 'generate_argument')
    return {name: (ROOT / 'prompts' / f'{name}.txt').read_text() for name in names}

def initial(scenario, texts, order='strategy_1_first', agents=2):
    a, b = scenario['strategy_1'], scenario['strategy_2']
    if order == 'strategy_2_first': a, b = b, a
    system = texts['system']
    if agents > 2:
        system = system.replace('another artificial agent', 'other artificial agents')
    return [{'role':'system', 'content':system}, {'role':'user', 'content':texts['scenario'].format(
        title=scenario['title'], a1=a['property_1'], a2=a['property_2'], b1=b['property_1'], b2=b['property_2'])}]
