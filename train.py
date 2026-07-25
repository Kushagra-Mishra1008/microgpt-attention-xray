"""
Training script for MicroGPT, using a trained BPE tokenizer.

Usage:
    python train.py
"""
    import os
    import torch

    from model.tokenizer import BPETokenizer
    from model.gpt import GPTLanguageModel

    # ---- config ----
    n_embd = 512
    n_head = 8
    n_layer = 8
    block_size = 256
    batch_size = 64
    learning_rate = 3e-4
    max_iters = 5000
    eval_interval = 500
    eval_iters = 100
    dropout = 0.2
    num_merges = 3000   # BPE vocabulary size = num unique chars + num_merges

    CHECKPOINT_NAME = "checkpoint_medium.pt"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"using device: {device}")


    def load_split(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()


    def get_batch(data, batch_size, block_size):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x, y


    def init_weights(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    @torch.no_grad()
    def estimate_loss(model, train_data, val_data):
        out = {}
        model.eval()
        for split, split_data in [('train', train_data), ('val', val_data)]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                xb, yb = get_batch(split_data, batch_size, block_size)
                xb, yb = xb.to(device), yb.to(device)
                _, loss, _ = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out


    def main():
        train_text = load_split('data/train.txt')
        val_text = load_split('data/val.txt')

        print(f"training BPE tokenizer ({num_merges} merges)...")
        tokenizer = BPETokenizer()
        tokenizer.train(train_text + val_text, num_merges=num_merges, verbose=True)
        vocab_size = tokenizer.vocab_size
        print(f"vocab size: {vocab_size}")

        train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
        val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
        print(f"train tokens: {len(train_data)}, val tokens: {len(val_data)}")

        model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
        model.apply(init_weights)
        model = model.to(device)

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{n_params:.3f}M parameters")

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

        best_val_loss = float('inf')

        for iter in range(max_iters):
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss(model, train_data, val_data)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
                    f"lr {scheduler.get_last_lr()[0]:.6f}")

                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': {
                            'vocab_size': vocab_size, 'n_embd': n_embd, 'n_head': n_head,
                            'n_layer': n_layer, 'block_size': block_size,
                        },
                        'tokenizer': tokenizer.to_dict(),
                        'val_loss': best_val_loss,
                    }, os.path.join('checkpoints', CHECKPOINT_NAME)
                    print(f"  saved checkpoint (val loss {best_val_loss:.4f})")

            xb, yb = get_batch(train_data, batch_size, block_size)
            xb, yb = xb.to(device), yb.to(device)

            _, loss, _ = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"final train loss: {loss.item():.4f}")
        print(f"best val loss: {best_val_loss:.4f}")


    if __name__ == '__main__':
        main()