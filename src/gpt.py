import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import TransformerBlock


class GPT(nn.Module):
    """
    GPT-style decoder-only transformer.

    Architecture:
        token embedding + positional embedding
        → dropout
        → N x TransformerBlock (Pre-LN, causal MHA + FFN)
        → LayerNorm
        → linear head → logits over vocab

    Args:
        vocab_size:   number of tokens in vocabulary
        d_model:      embedding dimension
        n_heads:      number of attention heads
        n_layers:     number of transformer blocks
        max_seq_len:  maximum sequence length (context window)
        dropout:      dropout probability
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.dropout   = nn.Dropout(dropout)

        self.blocks   = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: token embedding and lm_head share weights
        # Reduces parameters; empirically improves performance
        self.lm_head.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            idx:     [B, T] token indices
            targets: [B, T] target indices (optional, for loss computation)
        Returns:
            logits: [B, T, vocab_size]
            loss:   scalar cross-entropy loss (None if targets not provided)
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        # Embeddings
        tok = self.token_emb(idx)                                # [B, T, d_model]
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # [T, d_model]
        x = self.dropout(tok + pos)                              # [B, T, d_model]

        # Transformer blocks
        for block in self.blocks:
            x, _ = block(x)

        x = self.ln_final(x)                                     # [B, T, d_model]
        logits = self.lm_head(x)                                 # [B, T, vocab_size]

        # Loss
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            idx:            [B, T] context token indices
            max_new_tokens: number of tokens to generate
            temperature:    >1 = more random, <1 = more deterministic
            top_k:          if set, sample only from top k logits
        Returns:
            [B, T + max_new_tokens] token indices
        """
        for _ in range(max_new_tokens):
            # Crop context to max_seq_len
            idx_cond = idx[:, -self.max_seq_len:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # [B, vocab_size] — last token only

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat([idx, next_tok], dim=1)

        return idx


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = GPT(
        vocab_size=65,
        d_model=128,
        n_heads=4,
        n_layers=3,
        max_seq_len=64,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    idx     = torch.randint(0, 65, (2, 32))
    targets = torch.randint(0, 65, (2, 32))
    logits, loss = model(idx, targets)
    print(f"Logits: {logits.shape}")          # [2, 32, 65]
    print(f"Loss:   {loss.item():.4f}")       # ~4.17 = ln(65)

    prompt    = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=10)
    print(f"Generated: {generated.shape}")    # [1, 21]
