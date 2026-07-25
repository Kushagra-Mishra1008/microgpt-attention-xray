"""
The full GPT model: token + position embeddings, a stack of transformer
Blocks, a final layer norm, and a linear head projecting back to vocab_size.

forward() can optionally return attention weights from every layer, shaped
(n_layer, B, n_head, T, T) -- this is the data the attention X-ray visualizes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.block import Block


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.0):
        super().__init__()
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, return_attention=False):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                             # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb                                                  # (B, T, n_embd)

        all_attn = [] if return_attention else None
        for block in self.blocks:
            x, attn_weights = block(x, return_attention=return_attention)
            if return_attention:
                all_attn.append(attn_weights)   # each: (B, n_head, T, T)

        x = self.ln_f(x)                                                       # (B, T, n_embd)
        logits = self.lm_head(x)                                               # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B_, T_, C = logits.shape
            loss = F.cross_entropy(logits.view(B_ * T_, C), targets.view(B_ * T_))

        attentions = torch.stack(all_attn, dim=0) if return_attention else None  # (n_layer, B, n_head, T, T)
        return logits, loss, attentions

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]        # crop to last block_size tokens
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]                     # only the last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx