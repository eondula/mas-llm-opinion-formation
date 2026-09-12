"""Cached full-message embeddings for adjacent-round stopping checks."""
import math

class MessageSimilarity:
    encoder = 'sentence-transformers/all-MiniLM-L6-v2'
    revision = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'

    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder, revision=self.revision)
        # Small CPU encoder avoids competing with the active conversation model's VRAM.
        self.model = AutoModel.from_pretrained(self.encoder, revision=self.revision).eval()
        self.cache = {}

    def compare(self, current, previous):
        missing = list(dict.fromkeys(t for t in current + previous if t not in self.cache))
        for text in missing:
            ids = self.tokenizer(text, add_special_tokens=False)['input_ids']
            # Encode all text in chunks, avoiding silent loss of long-message endings.
            chunks = [ids[i:i+254] for i in range(0, len(ids), 254)] or [[]]
            vectors, weights = [], []
            for chunk in chunks:
                # This pinned encoder is BERT: each chunk is [CLS] tokens [SEP].
                # Transformers 5 tokenizers no longer expose prepare_for_model.
                cls, sep = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
                if cls is None or sep is None:
                    raise ValueError('Message encoder requires BERT CLS and SEP tokens')
                ids_tensor = self.torch.tensor([[cls, *chunk, sep]], dtype=self.torch.long)
                inputs = dict(input_ids=ids_tensor,
                              attention_mask=self.torch.ones_like(ids_tensor),
                              token_type_ids=self.torch.zeros_like(ids_tensor))
                with self.torch.inference_mode():
                    hidden = self.model(**inputs).last_hidden_state
                    mask = inputs['attention_mask'].unsqueeze(-1)
                    vectors.append((hidden*mask).sum(1)/mask.sum(1).clamp(min=1))
                    weights.append(max(1,len(chunk)))
            pooled = sum(v*w for v,w in zip(vectors,weights))/sum(weights)
            self.cache[text] = self.torch.nn.functional.normalize(pooled, dim=1)[0].tolist()
        scores = []
        for a,b in zip(current, previous):
            value = sum(x*y for x,y in zip(self.cache[a],self.cache[b]))
            if not math.isfinite(value): raise ValueError('Non-finite message similarity')
            scores.append(max(-1., min(1.,value)))
        return scores
