"""Version and environment records, independent of the experiment runner."""
import importlib.metadata
import platform
from .config import ROOT, digest

SCORER_VERSION = 'candidate-log-likelihood-v1'
EXPLORATION_SCORER_VERSION = 'exploration-float32-v1'


def source_hashes():
    return {str(p.relative_to(ROOT)): digest(p.read_text())
            for p in sorted((ROOT / 'llm_mas').glob('*.py'))}


def environment(config, runtimes):
    from .choice_output import PARSER_VERSION
    packages = {}
    for name in ('torch', 'transformers', 'numpy', 'matplotlib'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    models = {}
    for name, runtime in runtimes.items():
        model = getattr(runtime, 'model', None)
        parameters = getattr(model, 'parameters', None)
        parameter = next(parameters(), None) if parameters else None
        tokenizer = getattr(runtime, 'tokenizer', None)
        models[name] = dict(device=getattr(runtime, 'device', config['device']),
                           parameter_dtype=str(parameter.dtype) if parameter is not None else None,
                           checkpoint_revision=getattr(getattr(model, 'config', None), '_commit_hash', None),
                           tokenizer_revision=getattr(runtime,'tokenizer_revision',None) or getattr(tokenizer, 'init_kwargs', {}).get('_commit_hash'))
    exploration = config['mode'] == 'scenario_exploration'
    return dict(schema_version=2, python=platform.python_version(), platform=platform.platform(),
                packages=packages, models=models,
                parser='binary-answer-json-v1' if config.get('dataset') else 'strict-json-1' if exploration else PARSER_VERSION,
                scorer=EXPLORATION_SCORER_VERSION if exploration else SCORER_VERSION,
                measurement_probe='question-yes-no-v1' if config.get('dataset') else 'historical-configured-probe',
                decoding=config['decoding'], sampling=config.get('sampling'),
                seed=config.get('seed', 42), repetitions=config.get('repetitions', 1))
