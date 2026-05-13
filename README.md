# GPT From Scratch

A clean, well-documented implementation of GPT built from first principles in PyTorch.
Built as part of a 90-day applied ML research sprint.

## What's Implemented

- **Single-Head Attention** — scaled dot-product attention with causal masking
- **Multi-Head Attention** — h parallel heads, output projection, shape-preserving
- **Transformer Block** — Pre-LN, MHA + FFN + residual connections
- **GPT** — full decoder-only transformer with weight tying and autoregressive generation

## Architecture

```
Input tokens [B, T]
     │
Token Embedding + Positional Embedding    [B, T, d_model]
     │
N × TransformerBlock
     ├── LayerNorm → MultiHeadAttention → + residual
     └── LayerNorm → FeedForward        → + residual
     │
LayerNorm → LM Head → Logits            [B, T, vocab_size]
```

## Quick Start

### Install dependencies

```bash
pip install torch numpy matplotlib wandb
```

### Run sanity check

```bash
cd src
python gpt.py
```

Expected output:
```
Parameters: ~1.5M
Logits: torch.Size([2, 32, 65])
Loss:   ~4.17        # ln(65) — random init baseline
Generated: torch.Size([1, 21])
```

### In Colab

```python
!git clone https://github.com/sa1-kumar/gpt-from-scratch.git
%cd gpt-from-scratch
!pip install torch numpy matplotlib wandb

import sys
sys.path.append('src')
from attention import SingleHeadAttention, MultiHeadAttention, TransformerBlock
from gpt import GPT
```

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Layer norm position | Pre-LN | Stable gradients at init, no warmup needed |
| Activation | GELU | Smoother than ReLU, standard in GPT-2+ |
| Weight tying | Yes | Fewer params, better performance |
| Positional encoding | Learned | Simpler than sinusoidal, works as well in practice |
| Bias in QKV | No | Empirically unnecessary, saves params |

## Repo Structure

```
gpt-from-scratch/
├── src/
│   ├── attention.py   # SingleHeadAttention, MultiHeadAttention, TransformerBlock
│   ├── gpt.py         # GPT model + generation
│   └── __init__.py
├── notebooks/
│   └── 01-exploration.ipynb   # experiments and visualizations
├── README.md
└── .gitignore
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
