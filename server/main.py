"""
FastAPI inference server for MicroGPT.

On startup: downloads any missing checkpoint files from Hugging Face Hub,
downloads the training corpus if missing, then loads whatever checkpoints
are available. Supports both char-level and BPE (word-level) checkpoints --
tokenizer type is auto-detected from what's saved inside each checkpoint.

Run with:
    python -m uvicorn server.main:app --reload --port 8080
"""

import os
import json
import asyncio
import urllib.request

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from model.gpt import GPTLanguageModel
from model.tokenizer import BPETokenizer, CharTokenizer
from server.schemas import (
    TokenizeRequest, TokenizeResponse,
    AttentionRequest, AttentionResponse,
    GenerateRequest, ModelInfo,
)

CHECKPOINT_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HF_REPO = "Kushagra-Mishra1008/microgpt-attention-xray-checkpoints"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

app = FastAPI(title="MicroGPT Attention X-Ray API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOADED_MODELS = {}
CORPUS_TEXT = ""


def ensure_checkpoint_downloaded(filename):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(path):
        return path

    url = f"{HF_BASE_URL}/{filename}"
    print(f"  downloading {filename} from Hugging Face Hub...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"  done: {filename}")
        return path
    except Exception as e:
        print(f"  failed to download {filename}: {e}")
        if os.path.exists(path):
            os.remove(path)
        return None


def ensure_data_downloaded():
    os.makedirs("data", exist_ok=True)
    train_path = os.path.join("data", "train.txt")
    val_path = os.path.join("data", "val.txt")
    if os.path.exists(train_path) and os.path.exists(val_path):
        return

    print("  data/train.txt or data/val.txt missing -- downloading Tiny Shakespeare...")
    raw_path = os.path.join("data", "shakespeare.txt")
    urllib.request.urlretrieve(SHAKESPEARE_URL, raw_path)
    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read()

    split_idx = int(len(text) * 0.9)
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(text[:split_idx])
    with open(val_path, "w", encoding="utf-8") as f:
        f.write(text[split_idx:])
    print("  data ready.")


def build_tokenizer_from_checkpoint(tok_dict):
    tok_type = tok_dict.get("type")
    if tok_type == "bpe":
        return BPETokenizer.from_dict(tok_dict)
    if tok_type == "char":
        return CharTokenizer.from_dict(tok_dict)
    if "merges" in tok_dict:
        return BPETokenizer.from_dict(tok_dict)
    return CharTokenizer.from_dict(tok_dict)


def load_checkpoint(name, filename):
    path = ensure_checkpoint_downloaded(filename)
    if path is None or not os.path.exists(path):
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

    tok_dict = checkpoint.get("tokenizer")
    if tok_dict is not None:
        tokenizer = build_tokenizer_from_checkpoint(tok_dict)
    elif "stoi" in checkpoint and "itos" in checkpoint:
        tokenizer = CharTokenizer.from_dict({
            "stoi": checkpoint["stoi"],
            "itos": {str(k): v for k, v in checkpoint["itos"].items()},
            "vocab_size": config["vocab_size"],
        })
    else:
        print(f"  '{name}' checkpoint has no saved tokenizer -- rebuilding from data files")
        ensure_data_downloaded()
        with open("data/train.txt", "r", encoding="utf-8") as f:
            train_text = f.read()
        with open("data/val.txt", "r", encoding="utf-8") as f:
            val_text = f.read()
        tokenizer = CharTokenizer(train_text + val_text)

    tokenizer_type = "bpe" if isinstance(tokenizer, BPETokenizer) else "char"

    return {
        "model": model,
        "tokenizer": tokenizer,
        "tokenizer_type": tokenizer_type,
        "config": config,
        "val_loss": float(checkpoint.get("val_loss", -1)),
    }


@app.on_event("startup")
def startup():
    global CORPUS_TEXT

    ensure_data_downloaded()

    candidates = {
        "small-char": "checkpoint_small_char.pt",
        "medium-char": "checkpoint_medium_char.pt",
        "small-word": "checkpoint_small_word.pt",
        "medium-word": "checkpoint_medium_word.pt",
    }
    for name, filename in candidates.items():
        try:
            loaded = load_checkpoint(name, filename)
        except Exception as e:
            print(f"  failed to load '{name}' from {filename}: {e}")
            continue
        if loaded is not None:
            LOADED_MODELS[name] = loaded
            print(f"loaded '{name}' ({loaded['tokenizer_type']}, "
                  f"val_loss={loaded['val_loss']:.4f}, vocab_size={loaded['config']['vocab_size']})")
    if not LOADED_MODELS:
        print(f"WARNING: no checkpoints could be loaded")

    try:
        with open("data/train.txt", "r", encoding="utf-8") as f:
            train_text = f.read()
        with open("data/val.txt", "r", encoding="utf-8") as f:
            val_text = f.read()
        CORPUS_TEXT = train_text + val_text
        print(f"loaded corpus text ({len(CORPUS_TEXT):,} characters)")
    except FileNotFoundError:
        print("WARNING: could not load corpus text -- /corpus will be empty")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(LOADED_MODELS.keys())}


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
            tokenizer_type=entry["tokenizer_type"],
        )
        for name, entry in LOADED_MODELS.items()
    ]


@app.get("/corpus")
def get_corpus():
    return {
        "text": CORPUS_TEXT,
        "length": len(CORPUS_TEXT),
        "source": "Tiny Shakespeare",
    }


@app.post("/tokenize", response_model=TokenizeResponse)
def tokenize(req: TokenizeRequest):
    if not LOADED_MODELS:
        raise HTTPException(status_code=503, detail="no models loaded")
    tokenizer = next(iter(LOADED_MODELS.values()))["tokenizer"]

    token_ids = tokenizer.encode(req.text)
    tokens = [tokenizer.itos[i] for i in token_ids]
    return TokenizeResponse(tokens=tokens, token_ids=token_ids)


@app.post("/attention", response_model=AttentionResponse)
def get_attention(req: AttentionRequest):
    entry = get_model_or_404(req.model)
    model, tokenizer, config = entry["model"], entry["tokenizer"], entry["config"]

    token_ids = tokenizer.encode(req.text)[-config["block_size"]:]
    idx = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        _, _, attentions = model(idx, return_attention=True)
        attentions = attentions[:, 0, :, :, :].cpu().tolist()

    tokens = [tokenizer.itos[i] for i in token_ids]
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
            piece = tokenizer.itos[token_id]
            last_layer_attn = attentions[-1, 0, :, -1, :].cpu().tolist()

            payload = {"token": piece, "token_id": token_id, "attention_row": last_layer_attn}
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")