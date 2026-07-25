"""
FastAPI inference server for MicroGPT.

Loads whichever trained checkpoints are available in checkpoints/ at startup,
and serves tokenization, attention-weight extraction, and text generation.

Run with:
    uvicorn server.main:app --reload
"""

import os
import json
import asyncio

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from model.gpt import GPTLanguageModel
from model.tokenizer import CharTokenizer
from server.schemas import (
    TokenizeRequest, TokenizeResponse,
    AttentionRequest, AttentionResponse,
    GenerateRequest, ModelInfo,
)

CHECKPOINT_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="MicroGPT Attention X-Ray API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this before deploying publicly
    allow_methods=["*"],
    allow_headers=["*"],
)

# name -> {"model": GPTLanguageModel, "tokenizer": CharTokenizer, "config": dict, "val_loss": float}
LOADED_MODELS = {}


def build_fallback_tokenizer():
    """Rebuild the tokenizer from the raw data files. Used when a checkpoint
    was saved before stoi/itos were included in the saved dict."""
    with open("data/train.txt", "r", encoding="utf-8") as f:
        train_text = f.read()
    with open("data/val.txt", "r", encoding="utf-8") as f:
        val_text = f.read()
    return CharTokenizer(train_text + val_text)


def load_checkpoint(name, filename):
    path = os.path.join(CHECKPOINT_DIR, filename)
    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]

    model = GPTLanguageModel(
        vocab_size=config["vocab_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        block_size=config["block_size"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    if "stoi" in checkpoint and "itos" in checkpoint:
        tokenizer = CharTokenizer.__new__(CharTokenizer)  # bypass __init__, restore directly
        tokenizer.stoi = checkpoint["stoi"]
        tokenizer.itos = {int(k): v for k, v in checkpoint["itos"].items()}
        tokenizer.vocab_size = config["vocab_size"]
    else:
        print(f"  '{name}' checkpoint has no saved tokenizer -- rebuilding from data files")
        tokenizer = build_fallback_tokenizer()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "val_loss": float(checkpoint.get("val_loss", -1)),
    }


@app.on_event("startup")
def startup():
    candidates = {
        "micro": "checkpoint_micro.pt",
        "small": "checkpoint_small.pt",
        "medium": "checkpoint_medium.pt",
    }
    for name, filename in candidates.items():
        loaded = load_checkpoint(name, filename)
        if loaded is not None:
            LOADED_MODELS[name] = loaded
            print(f"loaded '{name}' (val_loss={loaded['val_loss']:.4f})")
    if not LOADED_MODELS:
        print(f"WARNING: no checkpoints found in {CHECKPOINT_DIR}/")


def get_model_or_404(name):
    if name not in LOADED_MODELS:
        available = list(LOADED_MODELS.keys())
        raise HTTPException(
            status_code=404,
            detail=f"model '{name}' not loaded. available: {available}",
        )
    return LOADED_MODELS[name]


@app.get("/models", response_model=list[ModelInfo])
def list_models():
    return [
        ModelInfo(
            name=name,
            n_embd=entry["config"]["n_embd"],
            n_head=entry["config"]["n_head"],
            n_layer=entry["config"]["n_layer"],
            block_size=entry["config"]["block_size"],
            val_loss=entry["val_loss"],
        )
        for name, entry in LOADED_MODELS.items()
    ]


@app.post("/tokenize", response_model=TokenizeResponse)
def tokenize(req: TokenizeRequest):
    # tokenizer is shared vocab across checkpoints trained on the same data,
    # so any loaded model's tokenizer works here
    if not LOADED_MODELS:
        raise HTTPException(status_code=503, detail="no models loaded")
    tokenizer = next(iter(LOADED_MODELS.values()))["tokenizer"]

    token_ids = tokenizer.encode(req.text)
    tokens = [req.text[i] for i in range(len(req.text))]
    return TokenizeResponse(tokens=tokens, token_ids=token_ids)


@app.post("/attention", response_model=AttentionResponse)
def get_attention(req: AttentionRequest):
    entry = get_model_or_404(req.model)
    model, tokenizer, config = entry["model"], entry["tokenizer"], entry["config"]

    text = req.text[-config["block_size"]:]   # truncate to what the model can see
    token_ids = tokenizer.encode(text)
    idx = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        _, _, attentions = model(idx, return_attention=True)
        # attentions: (n_layer, B=1, n_head, T, T) -> drop batch dim, move to cpu list
        attentions = attentions[:, 0, :, :, :].cpu().tolist()

    tokens = list(text)
    return AttentionResponse(
        tokens=tokens,
        attention=attentions,
        n_layer=config["n_layer"],
        n_head=config["n_head"],
    )


def sample_next_token(logits, temperature, top_k):
    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@app.post("/generate")
async def generate(req: GenerateRequest):
    entry = get_model_or_404(req.model)
    model, tokenizer, config = entry["model"], entry["tokenizer"], entry["config"]
    block_size = config["block_size"]

    idx = torch.tensor([tokenizer.encode(req.prompt)], dtype=torch.long, device=DEVICE)

    async def event_stream():
        nonlocal idx
        for _ in range(req.max_new_tokens):
            idx_cond = idx[:, -block_size:]
            with torch.no_grad():
                logits, _, attentions = model(idx_cond, return_attention=True)
            logits = logits[:, -1, :]
            idx_next = sample_next_token(logits, req.temperature, req.top_k)
            idx = torch.cat((idx, idx_next), dim=1)

            token_id = idx_next.item()
            char = tokenizer.itos[token_id]
            last_layer_attn = attentions[-1, 0, :, -1, :].cpu().tolist()  # last layer, last token's attention row, all heads

            payload = {"token": char, "token_id": token_id, "attention_row": last_layer_attn}
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0)  # yield control so the stream actually flushes

    return StreamingResponse(event_stream(), media_type="text/event-stream")