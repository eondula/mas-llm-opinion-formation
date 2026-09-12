import unittest
from types import SimpleNamespace
from llm_mas.runtime import select_device

class DeviceTests(unittest.TestCase):
    def test_device_availability(self):
        for cuda,mps,expected in [(True,True,'cuda'),(False,True,'mps'),(False,False,'cpu')]:
            torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:cuda),backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:mps)))
            self.assertEqual(select_device(torch,'auto'),expected)
            self.assertEqual(select_device(torch,'cpu'),'cpu')
            if not cuda:
                with self.assertRaisesRegex(ValueError,'CUDA is unavailable'):select_device(torch,'cuda')
