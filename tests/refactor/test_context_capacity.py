import unittest
from types import SimpleNamespace
from llm_mas.runtime import Runtime

class ContextCapacityTests(unittest.TestCase):
    def runtime(self):
        r = Runtime.__new__(Runtime)
        r.name = 'test-model'
        r.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=100))
        r.interaction_context = dict(agent='Agent B', round=3, stage='decision')
        return r

    def test_boundary_includes_output_reservation(self):
        r = self.runtime()
        r.check_context(80,20)
        with self.assertRaisesRegex(ValueError, 'input=80 tokens, reserved output=21, limit=100'):
            r.check_context(80,21)

    def test_error_identifies_failing_agent_and_model(self):
        with self.assertRaisesRegex(ValueError, 'test-model.*agent=Agent B, round=3, stage=decision'):
            self.runtime().check_context(101)

    def test_unknown_limit_is_allowed(self):
        r = self.runtime()
        r.model.config = SimpleNamespace()
        r.check_context(1000)
