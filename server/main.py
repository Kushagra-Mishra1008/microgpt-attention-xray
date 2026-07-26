"""
FastAPI inference server for MicroGPT.

At startup: pre-downloads all checkpoint FILES to disk in a background
thread (so a live request never has to wait on a slow network download
inside its own timeout window), but doesn't load any of them into memory
yet. Models are lazy-loaded into RAM one at a time on first actual use --
this keeps RAM usage bounded for free-tier hosting. /models reports whether
each checkpoint file is actually present on disk yet, so the frontend can
show a "still preparing" state instead of a raw error.

Run with:
    python -m uvicorn server.main:app --reload --port 8080
"""

import os
import gc
import json
import asyncio
import threading
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

MODEL_REGISTRY = {
    "small-char":  {"filename": "checkpoint_small_char.pt",  "tokenizer_type": "char", "n_embd": 384, "n_head": 6, "n_layer": 6, "block_size": 256, "vocab_size": 65,   "val_loss": 1.5414},
    "medium-char": {"filename": "checkpoint_medium_char.pt", "tokenizer_type": "char", "n_embd": 512, "n_head": 8, "n_layer": 8, "block_size": 256, "vocab_size": 65,   "val_loss": 1.4852},
    "small-word":  {"filename": "checkpoint_small_word.pt",  "tokenizer_type": "bpe",  "n_embd": 384, "n_head": 6, "n_layer": 6, "block_size": 256, "vocab_size": 3065, "val_loss": 4.5400},
    "medium-word": {"filename": "checkpoint_medium_word.pt", "tokenizer_type": "bpe",  "n_embd": 512, "n_head": 8, "n_layer": 8, "block_size": 256, "vocab_size": 3065, "val_loss": 4.5671},
}

app = FastAPI(title="MicroGPT Attention X-Ray API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CURRENT = None
CORPUS_TEXT = ""


def checkpoint_path(filename):
    return os.path.join(CHECKPOINT_DIR, filename)


def is_downloaded(filename):
    return os.path.exists(checkpoint_path(filename))


def ensure_checkpoint_downloaded(filename):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = checkpoint_path(filename)
    if os.path.exists(path):
        return path

    url = f"{HF_BASE_URL}/{filename}"
    print(f"  downloading {filename} from Hugging Face Hub...")
    tmp_path = path + ".partial"
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, path)
    print(f"  done: {filename}")
    return path


def ensure_data_downloaded():
    os.makedirs("data", exist_ok=True)
    train_path = os.path.join("data", "train.txt")
    val_path = os.path.join("data", "val.txt")
    if os.path.exists(train_path) and os.path.exists(val_path):
        return

    print("  downloading Tiny Shakespeare...")
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


def get_model(name):
    global CURRENT

    if name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"unknown model '{name}'. available: {list(MODEL_REGISTRY.keys())}")

    if CURRENT is not None and CURRENT["name"] == name:
        return CURRENT

    if CURRENT is not None:
        print(f"  evicting '{CURRENT['name']}' to load '{name}'")
        del CURRENT["model"]
        CURRENT = None
        gc.collect()

    reg = MODEL_REGISTRY[name]
    path = ensure_checkpoint_downloaded(reg["filename"])
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
        ensure_data_downloaded()
        with open("data/train.txt", "r", encoding="utf-8") as f:
            train_text = f.read()
        with open("data/val.txt", "r", encoding="utf-8") as f:
            val_text = f.read()
        tokenizer = CharTokenizer(train_text + val_text)

    print(f"  loaded '{name}'")
    CURRENT = {"name": name, "model": model, "tokenizer": tokenizer, "config": config}
    return CURRENT


@app.on_event("startup")
def startup():
    global CORPUS_TEXT
    ensure_data_downloaded()
    try:
        with open("data/train.txt", "r", encoding="utf-8") as f:
            train_text = f.read()
        with open("data/val.txt", "r", encoding="utf-8") as f:
            val_text = f.read()
        CORPUS_TEXT = train_text + val_text
        print(f"loaded corpus text ({len(CORPUS_TEXT):,} characters)")
    except FileNotFoundError:
        print("WARNING: could not load corpus text -- /corpus will be empty")

    def predownload_all():
        for name, reg in MODEL_REGISTRY.items():
            try:
                ensure_checkpoint_downloaded(reg["filename"])
            except Exception as e:
                print(f"  pre-download failed for '{name}': {e}")
        print("  all checkpoints pre-downloaded to disk.")

    threading.Thread(target=predownload_all, daemon=True).start()


@app.get("/health")
def health():
    return {"status": "ok", "current_model": CURRENT["name"] if CURRENT else None}


@app.get("/models", response_model=list[ModelInfo])
def list_models():
    return [
        ModelInfo(
            name=name,
            n_embd=reg["n_embd"],
            n_head=reg["n_head"],
            n_layer=reg["n_layer"],
            block_size=reg["block_size"],
            val_loss=reg["val_loss"],
            tokenizer_type=reg["tokenizer_type"],
            ready=is_downloaded(reg["filename"]),
        )
        for name, reg in MODEL_REGISTRY.items()
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
    entry = get_model(req.model)
    tokenizer = entry["tokenizer"]

    token_ids = tokenizer.encode(req.text)
    tokens = [tokenizer.itos[i] for i in token_ids]
    return TokenizeResponse(tokens=tokens, token_ids=token_ids)


@app.post("/attention", response_model=AttentionResponse)
def get_attention(req: AttentionRequest):
    entry = get_model(req.model)
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
    entry = get_model(req.model)
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