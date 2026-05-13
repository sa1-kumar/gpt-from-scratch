"""
train.py — Train a character-level GPT on Tiny Shakespeare
Usage: python3 train.py
"""

import torch
import torch.nn as nn
import time
import math
from gpt import GPT

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = '../input.txt'

# Model
VOCAB_SIZE  = None   # set after tokenization
D_MODEL     = 128
N_HEADS     = 4
N_LAYERS    = 3
MAX_SEQ_LEN = 64
DROPOUT     = 0.1

# Training
BATCH_SIZE  = 32
MAX_ITERS   = 3000
EVAL_EVERY  = 200
LR          = 3e-4

# Device
DEVICE = 'mps' if torch.backends.mps.is_available() else \
         'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE}")

# ── Data ──────────────────────────────────────────────────────────────────────
with open(DATA_PATH, 'r') as f:
    text = f.read()

# Character-level tokenization
chars     = sorted(set(text))
VOCAB_SIZE = len(chars)
stoi      = {c: i for i, c in enumerate(chars)}
itos      = {i: c for i, c in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"Vocab size: {VOCAB_SIZE} chars")
print(f"Dataset size: {len(text):,} characters")

# Train/val split (90/10)
data  = torch.tensor(encode(text), dtype=torch.long)
n     = int(0.9 * len(data))
train = data[:n]
val   = data[n:]

print(f"Train: {len(train):,} | Val: {len(val):,}")

# ── Batch sampler ─────────────────────────────────────────────────────────────
def get_batch(split):
    d   = train if split == 'train' else val
    ix  = torch.randint(len(d) - MAX_SEQ_LEN, (BATCH_SIZE,))
    x   = torch.stack([d[i:i+MAX_SEQ_LEN] for i in ix])
    y   = torch.stack([d[i+1:i+MAX_SEQ_LEN+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ── Model ─────────────────────────────────────────────────────────────────────
model = GPT(
    vocab_size  = VOCAB_SIZE,
    d_model     = D_MODEL,
    n_heads     = N_HEADS,
    n_layers    = N_LAYERS,
    max_seq_len = MAX_SEQ_LEN,
    dropout     = DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ── Eval ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss(eval_iters=50):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# ── Training loop ─────────────────────────────────────────────────────────────
print("\nStarting training...\n")
train_losses, val_losses, iters_log = [], [], []
t0 = time.time()

for step in range(MAX_ITERS):
    # Eval checkpoint
    if step % EVAL_EVERY == 0:
        losses = estimate_loss()
        elapsed = time.time() - t0
        print(f"Step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | {elapsed:.1f}s")
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        iters_log.append(step)

        # Generate sample
        if step > 0:
            prompt    = torch.zeros((1, 1), dtype=torch.long).to(DEVICE)
            generated = model.generate(prompt, max_new_tokens=100, temperature=0.8, top_k=20)
            print(f"\nSample:\n{decode(generated[0].tolist())}\n")

    # Forward + backward
    x, y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ── Final generation ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Training complete. Final generation:")
print("="*60)
prompt    = torch.zeros((1, 1), dtype=torch.long).to(DEVICE)
generated = model.generate(prompt, max_new_tokens=300, temperature=0.8, top_k=20)
print(decode(generated[0].tolist()))

# ── Save model ────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), '../gpt_shakespeare.pt')
print("\nModel saved to gpt_shakespeare.pt")
