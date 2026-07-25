"""
FeedForward and Block: the two pieces that combine with attention to form
one transformer layer, including residual connections and layer norm.
"""

import torch.nn as nn

from model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Per-token MLP: expand, non-linearity, project back down."""

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: self-attention + feedforward, each with a
    pre-norm residual connection."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, return_attention=False):
        attn_out, attn_weights = self.sa(self.ln1(x), return_attention=return_attention)
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x, attn_weights