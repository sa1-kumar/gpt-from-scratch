import math
import torch
from attention import (
    SingleHeadAttention,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    SinusoidalPositionalEncoding,
)
from gpt import GPT


def section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def ok(label: str):
    print(f"  [PASS] {label}")


# ── 1. SingleHeadAttention ────────────────────────────────
section("1. SingleHeadAttention")

B, T, d_model, d_k = 2, 8, 32, 16
sha = SingleHeadAttention(d_model, d_k)
x = torch.randn(B, T, d_model)
out, w = sha(x, causal=True)

assert out.shape == (B, T, d_k),   f"expected {(B,T,d_k)}, got {out.shape}"
assert w.shape   == (B, T, T),     f"expected {(B,T,T)},   got {w.shape}"

# causal: upper triangle of weights must be ~0
triu = torch.triu(w, diagonal=1)
assert triu.abs().max().item() < 1e-6, "causal mask leaking"

ok("output shape")
ok("weight shape")
ok("causal mask — upper triangle zeroed")

# non-causal
out_nc, w_nc = sha(x, causal=False)
assert out_nc.shape == (B, T, d_k)
ok("non-causal forward pass")


# ── 2. MultiHeadAttention ────────────────────────────────
section("2. MultiHeadAttention")

n_heads = 4
mha = MultiHeadAttention(d_model, n_heads)
out, w = mha(x, causal=True)

assert out.shape == (B, T, d_model),          f"got {out.shape}"
assert w.shape   == (B, n_heads, T, T),       f"got {w.shape}"

triu = torch.triu(w, diagonal=1)
assert triu.abs().max().item() < 1e-6, "causal mask leaking in MHA"

ok("output shape [B, T, d_model]")
ok("weight shape [B, n_heads, T, T]")
ok("causal mask across all heads")


# ── 3. FeedForward ───────────────────────────────────────
section("3. FeedForward")

ffn = FeedForward(d_model)
out = ffn(x)
assert out.shape == (B, T, d_model), f"got {out.shape}"
ok("output shape preserved")


# ── 4. TransformerBlock ──────────────────────────────────
section("4. TransformerBlock")

block = TransformerBlock(d_model, n_heads)
out, w = block(x, causal=True)
assert out.shape == (B, T, d_model), f"got {out.shape}"
assert w.shape   == (B, n_heads, T, T)
ok("output shape [B, T, d_model]")
ok("weight shape [B, n_heads, T, T]")

# residual: output should differ from input (not identity)
assert not torch.allclose(out, x), "block output == input — residual broken?"
ok("residual connection changes activations")


# ── 5. SinusoidalPositionalEncoding ─────────────────────
section("5. SinusoidalPositionalEncoding")

spe = SinusoidalPositionalEncoding(d_model, max_seq_len=512)
x_emb = torch.randn(B, T, d_model)
out = spe(x_emb)

assert out.shape == (B, T, d_model), f"got {out.shape}"
ok("output shape preserved")

# no learnable parameters
n_params = sum(p.numel() for p in spe.parameters())
assert n_params == 0, f"expected 0 params, got {n_params}"
ok("zero learnable parameters")

# buffer is registered
assert hasattr(spe, 'pe'), "pe buffer missing"
assert spe.pe.shape == (1, 512, d_model)
ok("pe buffer shape [1, max_seq_len, d_model]")

# shorter sequence works
out_short = spe(torch.randn(B, 3, d_model))
assert out_short.shape == (B, 3, d_model)
ok("handles T < max_seq_len")

# encoding is deterministic (fixed)
out_a = spe(x_emb)
out_b = spe(x_emb)
assert torch.allclose(out_a, out_b)
ok("deterministic — same input gives same output")

# even dims are sin, odd dims are cos (spot-check pos=0 vs pos=1)
pe = spe.pe[0]           # [T, d_model]
div0 = math.exp(0 * -(math.log(10000.0) / d_model))  # dim 0 divisor = 1
assert abs(pe[0, 0].item() - math.sin(0 * div0)) < 1e-5,  "even dim != sin"
assert abs(pe[0, 1].item() - math.cos(0 * div0)) < 1e-5,  "odd dim != cos"
ok("even dims = sin, odd dims = cos")


# ── 6. GPT forward pass ──────────────────────────────────
section("6. GPT — end-to-end forward pass")

gpt = GPT(vocab_size=65, d_model=64, n_heads=4, n_layers=2, max_seq_len=64)
idx = torch.randint(0, 65, (B, T))

logits, loss = gpt(idx)
assert logits.shape == (B, T, 65), f"got {logits.shape}"
assert loss is None
ok("logits shape [B, T, vocab_size]")

targets = torch.randint(0, 65, (B, T))
logits2, loss = gpt(idx, targets)
assert loss is not None
assert loss.item() > 0
ok(f"cross-entropy loss = {loss.item():.4f}")


# ── 7. GPT generate ──────────────────────────────────────
section("7. GPT — generate()")

gpt.eval()
prompt = torch.randint(0, 65, (1, 4))
generated = gpt.generate(prompt, max_new_tokens=10)
assert generated.shape == (1, 14), f"got {generated.shape}"
ok(f"generated {10} new tokens, total length = {generated.shape[1]}")


print(f"\n{'='*55}")
print("  All tests passed.")
print(f"{'='*55}\n")
