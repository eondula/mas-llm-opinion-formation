"""Chat adaptation and scoring inputs for the additional dashboard models."""
import unittest
from llm_mas.runtime import Runtime
from llm_mas.config import resolve

MODELS = ('TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'allenai/OLMo-2-0425-1B-Instruct')

class CheckpointTests(unittest.TestCase):
    def test_rendered_tokens_do_not_duplicate_special_tokens(self):
        for name in MODELS:
            runtime = Runtime.__new__(Runtime)
            runtime.name = name
            calls = []
            runtime.tokenizer = lambda text, **kwargs: calls.append((text,kwargs))
            runtime.encode_rendered('prompt')
            runtime.encode_rendered('promptcandidate', return_tensors='pt')
            self.assertEqual(calls[0][1], dict(add_special_tokens=False))
            self.assertEqual(calls[1][1], dict(add_special_tokens=False,return_tensors='pt'))

    def test_existing_checkpoint_inputs_unchanged(self):
        runtime = Runtime.__new__(Runtime)
        runtime.name = 'Qwen/Qwen2.5-1.5B-Instruct'
        runtime.tokenizer = lambda text, **kwargs: kwargs
        self.assertEqual(runtime.encode_rendered('text'), {})

    def test_new_model_assignments_resolve(self):
        for names in ([MODELS[0]], [MODELS[1]], list(MODELS)):
            config = resolve(dict(agents=2,models=names))
            self.assertEqual(config['models'], names)

    def test_smollm3_non_thinking_preserves_history(self):
        from copy import deepcopy
        from types import SimpleNamespace
        runtime = Runtime.__new__(Runtime)
        runtime.name = 'HuggingFaceTB/SmolLM3-3B'
        runtime.tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **options: (messages, options))
        for history in ([dict(role='user',content='Task')],
                        [dict(role='system',content='Persona'),dict(role='user',content='Task')]):
            original = deepcopy(history)
            messages, options = runtime.render_chat(history)
            self.assertEqual(history, original)
            self.assertEqual(messages[0]['role'], 'system')
            self.assertTrue(messages[0]['content'].endswith('/no_think'))
            self.assertFalse(options['enable_thinking'])
        runtime.tokenizer = lambda text, **kwargs: kwargs
        self.assertFalse(runtime.encode_rendered('rendered')['add_special_tokens'])
        self.assertEqual(resolve(dict(models=[runtime.name]))['models'], [runtime.name])
