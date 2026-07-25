"""
Self-attention: single Head and MultiHeadAttention.

Built from scratch -- no HuggingFace, no einops. Every operation here is a
plain PyTorch primitive (Linear, matmul, softmax) so every line is auditable.

Both modules can optionally return their raw attention weights alongside the
output, which is what powers the attention X-ray visualization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """One self-attention head: causal, scaled dot-product attention."""

    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        B, T, C = x.shape
        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)    # (B, T, head_size)
        v = self.value(x)    # (B, T, head_size)

        wei = q @ k.transpose(-2, -1)               # (B, T, T) raw attention scores
        wei = wei * (k.shape[-1] ** -0.5)            # scale by 1/sqrt(head_size)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # causal mask
        wei = F.softmax(wei, dim=-1)                 # turn scores into probabilities
        attn_weights = wei                           # keep a reference before dropout

        wei = self.dropout(wei)
        out = wei @ v                                # (B, T, head_size) weighted sum of values

        if return_attention:
            return out, attn_weights
        return out, None


class MultiHeadAttention(nn.Module):
    """Multiple attention heads running in parallel, outputs concatenated and projected."""

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        head_outputs = []
        head_attns = []
        for h in self.heads:
            out, attn = h(x, return_attention=return_attention)
            head_outputs.append(out)
            if return_attention:
                head_attns.append(attn)  # each: (B, T, T)

        out = torch.cat(head_outputs, dim=-1)
        out = self.dropout(self.proj(out))

        if return_attention:
            # stack into (B, num_heads, T, T)
            attn_stack = torch.stack(head_attns, dim=1)
            return out, attn_stack
        return out, None