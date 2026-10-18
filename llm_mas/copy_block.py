"""Deterministic blocking of n-grams copied from an agent's own earlier messages.

Only n-grams that occur in the agent's previous public messages are banned, so
agents can still quote the question, the instructions and peer messages. N-grams
that also occur in the shared task context are exempt.
"""
from collections import defaultdict


def ngrams(sequence, n):
    return {tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)}


class OwnMessageNgramBlock:
    def __init__(self, own_messages, exempt, n, prompt_length):
        if n < 2:
            raise ValueError('Copy blocking requires n >= 2')
        self.n, self.prompt_length = n, prompt_length
        allowed = set().union(*(ngrams(s, n) for s in exempt)) if exempt else set()
        self.banned = defaultdict(set)
        for sequence in own_messages:
            for gram in ngrams(sequence, n) - allowed:
                self.banned[gram[:-1]].add(gram[-1])
        self.blocked_steps = 0
        self.banned_ngrams = sum(len(v) for v in self.banned.values())

    def __call__(self, input_ids, scores):
        generated = input_ids[0, self.prompt_length:].tolist()
        if len(generated) < self.n-1:
            return scores
        ban = self.banned.get(tuple(generated[len(generated)-self.n+1:]))
        if not ban:
            return scores
        ids = list(ban)
        if int(scores[0].argmax()) in ban:
            self.blocked_steps += 1
        scores = scores.clone()
        scores[:, ids] = -float('inf')
        return scores
