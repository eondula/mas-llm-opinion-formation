"""Expand explicit study factors without coupling physical positions to names."""
from itertools import product


def conditions(config):
    explicit = config.get('conditions')
    if explicit: return explicit
    grid = config.get('grid', {})
    if not grid: return [{}]
    allowed={'schedule','argument_mode','presentation_order','initialization'}
    if set(grid)-allowed: raise ValueError('Unknown grid factor')
    keys=list(grid)
    if any(not isinstance(grid[k],list) or not grid[k] for k in keys):raise ValueError('Grid factors require nonempty lists')
    return [dict(zip(keys,values)) for values in product(*(grid[k] for k in keys))]


def configurations(config):
    models=config.get('model_configurations') or [config['models']]
    groups=config.get('agent_counts') or [config['agents']]
    for n in groups:
        for assignment in models:
            if len(assignment) not in (1,n):continue
            for condition in conditions(config):
                starts=list(product((1,2),repeat=n)) if condition.get('initialization')=='all_forced' else [condition.get('initialization',config['initialization'])]
                for start in starts:
                    yield dict(config,**dict(condition,agents=n,models=assignment,initialization=list(start) if isinstance(start,tuple) else start),conditions=[],grid={},model_configurations=[],agent_counts=[])
