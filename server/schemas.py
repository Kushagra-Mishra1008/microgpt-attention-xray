"""Request/response shapes for the inference API."""

from pydantic import BaseModel
from typing import List, Optional


class TokenizeRequest(BaseModel):
    text: str


class TokenizeResponse(BaseModel):
    tokens: List[str]
    token_ids: List[int]


class AttentionRequest(BaseModel):
    text: str
    model: str = "small"   # which checkpoint to use: micro / small / medium


class AttentionResponse(BaseModel):
    tokens: List[str]
    # attention[layer][head][from_token][to_token] -> weight
    attention: List[List[List[List[float]]]]
    n_layer: int
    n_head: int


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = None
    model: str = "small"


class ModelInfo(BaseModel):
    name: str
    n_embd: int
    n_head: int
    n_layer: int
    block_size: int
    val_loss: Optional[float] = None