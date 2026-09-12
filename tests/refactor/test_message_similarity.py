import unittest
from types import SimpleNamespace
import torch
from llm_mas.message_similarity import MessageSimilarity

class Tokenizer:
    cls_token_id=101
    sep_token_id=102
    # Deliberately no prepare_for_model, matching Transformers 5.
    def __call__(self, text, add_special_tokens=False):
        return {'input_ids':list(range(200,200+int(text)))}

class Model:
    def __init__(self): self.calls=[]
    def __call__(self, **inputs):
        self.calls.append(inputs)
        ids=inputs['input_ids'].float()
        return SimpleNamespace(last_hidden_state=torch.stack([ids,torch.ones_like(ids)],dim=-1))

class SimilarityTests(unittest.TestCase):
    def test_chunks_special_tokens_masks_and_cache(self):
        encoder=object.__new__(MessageSimilarity)
        encoder.torch=torch
        encoder.tokenizer=Tokenizer()
        encoder.model=Model()
        encoder.cache={}
        scores=encoder.compare(['510','0'],['510','0'])
        self.assertTrue(all(abs(s-1)<1e-6 for s in scores))
        self.assertEqual([c['input_ids'].shape[1] for c in encoder.model.calls],[256,256,4,2])
        for c in encoder.model.calls:
            self.assertEqual(c['input_ids'][0,0].item(),101)
            self.assertEqual(c['input_ids'][0,-1].item(),102)
            self.assertTrue(torch.all(c['attention_mask']==1))
            self.assertTrue(torch.all(c['token_type_ids']==0))
        encoder.compare(['510'],['0'])
        self.assertEqual(len(encoder.model.calls),4)
