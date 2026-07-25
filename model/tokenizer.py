"""
Two tokenizers:

CharTokenizer -- simple character-level (Phase 1), one token per character.

BPETokenizer -- Byte-Pair Encoding, built from scratch. Starts from a
character-level base vocabulary, then learns `num_merges` merge rules by
repeatedly combining the most frequent adjacent pair of tokens across the
training corpus -- the same algorithm used by GPT-2/3. Text is first split
into rough "words" (keeping leading whitespace attached, GPT-2 style) so
merges never cross word boundaries.
"""

import re
from collections import Counter


class CharTokenizer:
    def __init__(self, text=None):
        if text is not None:
            chars = sorted(set(text))
            self.vocab_size = len(chars)
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[ch] for ch in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def to_dict(self):
        return {
            "type": "char",
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, d):
        tok = cls()
        tok.stoi = d["stoi"]
        tok.itos = {int(k): v for k, v in d["itos"].items()}
        tok.vocab_size = d["vocab_size"]
        return tok


SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+"""
)


class BPETokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.merges = []
        self.merge_rank = {}
        self.vocab_size = 0

    def train(self, text, num_merges, verbose=False):
        base_chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(base_chars)}
        self.itos = {i: ch for i, ch in enumerate(base_chars)}
        next_id = len(base_chars)

        words = SPLIT_PATTERN.findall(text)
        word_counts = Counter(words)
        word_ids = {w: [self.stoi[ch] for ch in w] for w in word_counts}

        for merge_i in range(num_merges):
            pair_counts = Counter()
            for w, count in word_counts.items():
                ids = word_ids[w]
                for a, b in zip(ids, ids[1:]):
                    pair_counts[(a, b)] += count

            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            new_id = next_id
            next_id += 1

            self.itos[new_id] = self.itos[best_pair[0]] + self.itos[best_pair[1]]
            self.merges.append((best_pair, new_id))
            self.merge_rank[best_pair] = merge_i

            for w in word_counts:
                word_ids[w] = self._apply_merge(word_ids[w], best_pair, new_id)

            if verbose and merge_i % 200 == 0:
                print(f"merge {merge_i}: {best_pair} -> {new_id} "
                      f"({self.itos[new_id]!r}, count={best_count})")

        self.vocab_size = next_id

    @staticmethod
    def _apply_merge(ids, pair, new_id):
        out = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def encode(self, text):
        ids = []
        for word in SPLIT_PATTERN.findall(text):
            word_ids = [self.stoi[ch] for ch in word if ch in self.stoi]
            while len(word_ids) >= 2:
                pairs = list(zip(word_ids, word_ids[1:]))
                ranked = [(self.merge_rank[p], p) for p in pairs if p in self.merge_rank]
                if not ranked:
                    break
                _, pair = min(ranked, key=lambda x: x[0])
                new_id = next(nid for (p, nid) in self.merges if p == pair)
                word_ids = self._apply_merge(word_ids, pair, new_id)
            ids.extend(word_ids)
        return ids

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def to_dict(self):
        return {
            "type": "bpe",
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "merges": [[list(pair), new_id] for pair, new_id in self.merges],
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, d):
        tok = cls()
        tok.stoi = d["stoi"]
        tok.itos = {int(k): v for k, v in d["itos"].items()}
        tok.merges = [(tuple(pair), new_id) for pair, new_id in d["merges"]]
        tok.merge_rank = {pair: i for i, (pair, _) in enumerate(tok.merges)}
        tok.vocab_size = d["vocab_size"]
        return tok