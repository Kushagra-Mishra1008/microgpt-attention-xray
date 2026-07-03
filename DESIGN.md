# Project: MicroGPT + Attention X-Ray

A GPT-style language model implemented from scratch in PyTorch, paired with a browser-based
"attention X-ray" — an interactive UI where you type text and watch attention weights light up
across every head and layer in real time.

**Author:** Kushagra (CS student, Java background, Python + ML basics, currently studying
Karpathy / 3Blue1Brown transformer material)

**Purpose:** Resume/portfolio project demonstrating deep understanding of transformer internals,
not just API orchestration.

---

## 1. Instructions for the AI assistant (READ FIRST)

This project has a strict division of labor. The primary goal is that Kushagra **understands
every line of the core model**. An impressive repo he cannot explain in an interview is a
failed project.

### Tutor mode — do NOT write this code for him
For the files listed below, act as a tutor and reviewer only:
- Explain concepts, answer questions, quiz him ("why do we scale by sqrt(d_k)?")
- Review code he has written; point out bugs and non-idiomatic patterns
- If he is stuck, give hints escalating in specificity — never paste the full solution
- After each component works, ask him 2–3 conceptual questions to verify understanding

**Tutor-mode files:** everything in `model/` (`tokenizer.py`, `attention.py`, `block.py`,
`gpt.py`) and the core training loop logic in `train.py`.

### Contractor mode — build this freely
Write, refactor, and polish without restriction:
- Data downloading/preprocessing scripts
- Training dashboard, logging, plotting, checkpointing utilities
- The entire FastAPI inference server (`server/`)
- The entire React frontend (`web/`)
- Tests, README, CI, deployment configs

### General behavior
- Work phase by phase (Section 5). Do not jump ahead.
- At the end of each phase, verify the acceptance criteria before moving on.
- Keep the core model dependency-free beyond PyTorch — no HuggingFace transformers,
  no einops. That constraint is the point.

---

## 2. Goals and non-goals

### Goals
1. From-scratch GPT: tokenizer, embeddings, multi-head causal self-attention, transformer
   blocks, training loop — all hand-written in plain PyTorch.
2. Attention X-ray web app: type a prompt, see per-head/per-layer attention as animated
   arcs/heatmaps; generate text token by token with temperature control.
3. Small scaling study: train 3 model sizes, compare loss curves and sample quality.
4. Clean, documented, deployable repo with a live demo link.

### Non-goals
- SOTA quality. Coherent-ish text on a small corpus is success.
- Large-scale training. Everything must train on a free Colab T4 / consumer GPU in hours.
- Multi-GPU, mixed precision wizardry, or serving at scale.

---

## 3. System architecture

Three components, cleanly separated:

```
[ Training (Python/PyTorch, offline) ]
        |  produces checkpoints (.pt) + tokenizer file
        v
[ Inference server (FastAPI) ]
        |  REST/JSON: tokenize, forward-with-attentions, generate (SSE stream)
        v
[ Attention X-ray frontend (React + SVG/Canvas) ]
```

### 3.1 Model (`model/`)
Decoder-only transformer, nanoGPT-style but hand-written:

| Config    | layers | heads | d_model | params | purpose            |
|-----------|--------|-------|---------|--------|--------------------|
| micro     | 4      | 4     | 128     | ~0.8M  | fast iteration     |
| small     | 6      | 6     | 384     | ~10M   | main demo model    |
| medium    | 8      | 8     | 512     | ~25M   | scaling study top  |

- Tokenizer: start character-level (simple, debuggable). Phase 6 stretch: hand-written BPE.
- Context length: 256 tokens (keeps attention matrices small enough to visualize).
- `forward(idx, return_attentions=True)` must return attention tensors of shape
  `[n_layers, n_heads, T, T]` — this is the contract the whole visualizer depends on.

### 3.2 Dataset
Primary: Tiny Shakespeare (~1MB, classic, instantly recognizable outputs).
Alternate for personality: chess games in PGN, or song lyrics-free corpus of public-domain text.
Decision: start with Shakespeare; swapping corpora later is trivial.

### 3.3 Inference server (`server/`)
FastAPI, loads a checkpoint at startup.

Endpoints:
- `POST /tokenize` → `{tokens: [...], token_ids: [...]}`
- `POST /attention` → body `{text}` → full attention tensor (float16, possibly top-k sparsified
  per row to cap payload size) + tokens
- `POST /generate` → body `{prompt, max_new_tokens, temperature, top_k}` → Server-Sent Events
  stream; each event carries the new token **and** its attention rows so the UI can animate
  generation live
- `GET /models` → available checkpoints (micro/small/medium) for the comparison view

### 3.4 Frontend (`web/`)
React + Vite. Visualization in SVG (arcs) and Canvas (heatmaps) — no heavy chart libs.

Views:
1. **X-ray view** — input box; tokens rendered as pills; attention drawn as arcs between
   tokens, thickness/opacity = weight; selectors for layer and head; hover a token to isolate
   its outgoing attention; small per-head heatmap grid.
2. **Generate view** — streaming generation with temperature/top-k sliders; each new token
   animates in with its attention fan.
3. **Compare view** (stretch) — same prompt through micro/small/medium side by side.

---

## 4. Repo layout

```
microgpt/
  DESIGN.md            <- this file
  CLAUDE.md            <- points the assistant at DESIGN.md section 1
  model/
    tokenizer.py       [tutor mode]
    attention.py       [tutor mode]  single head + multi-head, causal mask
    block.py           [tutor mode]  LN -> attn -> residual -> LN -> MLP -> residual
    gpt.py             [tutor mode]  embeddings, block stack, LM head, generate()
  train.py             [tutor mode: loop logic] [contractor: logging/checkpoint plumbing]
  data/
    prepare.py         [contractor]
  server/
    main.py            [contractor]
    schemas.py         [contractor]
  web/                 [contractor]
    src/...
  experiments/
    scaling.md         loss curves + samples for 3 sizes
  tests/               [contractor, but Kushagra reads them]
  README.md            [contractor draft, Kushagra edits]
```

---

## 5. Phase plan

### Phase 0 — Setup (half a day)
Repo, venv, PyTorch install, download Shakespeare, GPU sanity check (Colab or local).
**Done when:** `python -c "import torch; print(torch.cuda.is_available())"` behaves as expected
and data is downloaded.

### Phase 1 — Tokenizer + bigram baseline (weekend 1, part 1)
Char-level tokenizer; embedding-only bigram model; basic training loop; sample garbage text.
**Done when:** loss decreases from ~4.5 toward ~2.5 on Shakespeare and sampling produces
letter-frequency-plausible gibberish. Kushagra can explain cross-entropy and why the bigram
model's ceiling is low.

### Phase 2 — Self-attention (weekend 1, part 2 — the heart of the project)
Single head with causal masking, then multi-head. Verify attention rows sum to 1 and the
mask blocks future tokens (write these as actual asserts/tests).
**Done when:** shapes documented by hand for every tensor in the forward pass; Kushagra can
derive scaled dot-product attention on paper.

### Phase 3 — Full GPT + real training (weekend 2)
Blocks (residuals, LayerNorm, MLP), positional embeddings, weight init, LR schedule.
Train `small` to convergence (~15–30 min on T4).
**Done when:** val loss < 1.6 on Shakespeare char-level and samples look like drunk
Shakespeare. Checkpoints saved with config embedded.

### Phase 4 — Inference server (weekend 3, part 1) [contractor-heavy]
FastAPI endpoints per 3.3, including SSE streaming generation with attention payloads.
**Done when:** `curl` demos for all endpoints work; attention payload for a 50-token prompt
is < 2MB.

### Phase 5 — Attention X-ray UI (weekends 3–4) [contractor-heavy, Kushagra directs UX]
X-ray view then Generate view.
**Done when:** typing "the cat sat on the" and switching heads visibly shows different heads
attending to different structure; a 30-second screen recording is demo-ready.

### Phase 6 — Scaling study + polish (weekend 5)
Train micro/medium, write `experiments/scaling.md` with loss curves and sample comparisons.
Stretch: hand-written BPE tokenizer [tutor mode]; Compare view; deploy (server on a small
VM / HF Space, frontend on Vercel/GitHub Pages).
**Done when:** README has GIFs, live link, and a "what I learned" section.

---

## 6. Key technical decisions (locked)

1. **Plain PyTorch only for the model.** No HF, no einops. (Learning constraint.)
2. **Char-level first, BPE as stretch.** Debuggability beats vocabulary efficiency here.
3. **Context 256.** Attention matrices stay visualizable and payloads small.
4. **Attention extraction via `return_attentions=True`**, not forward hooks. Explicit > magic
   for a learning project.
5. **SSE for generation streaming** (simpler than WebSockets, one-directional fits the need).
6. **SVG arcs for attention, Canvas for heatmaps.** No charting libraries.
7. **float16 + optional top-k sparsification** for attention payloads to keep the UI snappy.

## 7. Risks

- **Training divergence / NaNs** → follow known-good nanoGPT hyperparameters; add grad clipping.
- **Attention payload too big for long prompts** → cap prompt length in UI; sparsify.
- **Scope creep on the frontend** → X-ray view is the product; everything else is stretch.
- **Assistant writes too much core code** → Section 1 rules; Kushagra should call this out
  if it happens.
