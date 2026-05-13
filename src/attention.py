import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SingleHeadAttention(nn.Module):
    """
    Scaled dot-product attention with optional causal masking.

    Args:
        d_model: input embedding dimension
        d_k: dimension of query/key/value projections
    """
    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True):
        """
        Args:
            x: [B, T, d_model]
            causal: if True, mask future positions (decoder/GPT style)
        Returns:
            output: [B, T, d_k]
            weights: [B, T, T]
        """
        B, T, _ = x.shape

        Q = self.W_q(x)   # [B, T, d_k]
        K = self.W_k(x)   # [B, T, d_k]
        V = self.W_v(x)   # [B, T, d_k]

        scores = Q @ K.transpose(-2, -1)        # [B, T, T]
        scores = scores / math.sqrt(self.d_k)   # scale to prevent softmax saturation

        if causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)     # [B, T, T]
        output = weights @ V                    # [B, T, d_k]
        return output, weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention: h heads run in parallel, each with d_k = d_model // h.
    Output shape matches input shape — blocks can be stacked.

    Args:
        d_model: input/output embedding dimension
        n_heads: number of attention heads
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # One large projection per Q/K/V — more efficient than h separate ones
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True):
        """
        Args:
            x: [B, T, d_model]
            causal: if True, mask future positions
        Returns:
            output: [B, T, d_model]
            weights: [B, n_heads, T, T]
        """
        B, T, d_model = x.shape

        Q = self.W_q(x)   # [B, T, d_model]
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads: [B, T, d_model] → [B, n_heads, T, d_k]
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # [B, n_heads, T, T]

        if causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)                      # [B, n_heads, T, T]
        out = weights @ V                                        # [B, n_heads, T, d_k]

        # Merge heads: [B, n_heads, T, d_k] → [B, T, d_model]
        # .contiguous() required after transpose before view (memory layout fix)
        out = out.transpose(1, 2).contiguous().view(B, T, d_model)

        return self.W_o(out), weights


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Expands to 4*d_model then contracts back. Applied per token independently.

    Args:
        d_model: input/output dimension
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # [B, T, d_model] → [B, T, d_model]


class TransformerBlock(nn.Module):
    """
    Full transformer block: Pre-LN + MHA + residual, Pre-LN + FFN + residual.

    Pre-LN (vs Post-LN): LayerNorm runs before sublayer, keeping gradients
    well-behaved at initialization. Enables stable training of deep stacks
    without careful warmup.

    Args:
        d_model: embedding dimension
        n_heads: number of attention heads
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, causal: bool = True):
        """
        Args:
            x: [B, T, d_model]
        Returns:
            x: [B, T, d_model]  ← same shape, stackable
            weights: [B, n_heads, T, T]
        """
        # Attention sublayer: Pre-LN + residual
        attn_out, weights = self.attn(self.ln1(x), causal=causal)
        x = x + attn_out

        # FFN sublayer: Pre-LN + residual
        x = x + self.ffn(self.ln2(x))

        return x, weights
