"""
Character-level tokenizer: builds a fixed vocabulary from the training text
and provides encode()/decode() between strings and integer token ids.
"""


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[ch] for ch in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])