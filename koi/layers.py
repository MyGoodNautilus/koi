"""Small shared blocks: norms, SwiGLU, short conv, rope. The goblin toolbox."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x.to(dtype) * self.weight


class GatedRMSNorm(nn.Module):
    """RMSNorm followed by a SiLU gate. Dessert the Gated DeltaNet layers get,
    the plain attention layers get none. Life is unfair, goblin eats anyway."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x.to(dtype) * self.weight
        return x * F.silu(gate)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class CausalShortConv(nn.Module):
    """Depthwise causal conv, one per channel.

    Gives q/k/v a tiny nibble of local context before the big mixing. Standard
    Gated DeltaNet snack. Supports full sequence mode and one token mode.
    """

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, groups=dim,
            padding=kernel_size - 1, bias=True,
        )

    def forward(self, x: torch.Tensor, conv_state: Optional[torch.Tensor] = None):
        # x: (B, L, D)
        if conv_state is None:
            # full sequence mode, slice off the padded tail
            y = self.conv(x.transpose(1, 2))[:, :, : x.shape[1]]
            return y.transpose(1, 2), None
        else:
            # one token mode, state holds the last kernel_size - 1 inputs
            B, D = x.shape[0], x.shape[2]
            xt = x[:, -1:, :]
            window = torch.cat([conv_state, xt], dim=1)
            w = self.conv.weight.squeeze(1)
            y = (window.transpose(1, 2) * w.unsqueeze(0)).sum(dim=-1) + self.conv.bias
            y = y.unsqueeze(1)
            new_state = window[:, 1:, :]
            return y, new_state


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float):
    """cos/sin tables for absolute positions, always fp32.

    Goblin no trust cheap math near trig. Only relative gaps matter later, so
    absolute positions are safe to rotate with.
    """
    half = head_dim // 2
    inv = 1.0 / (theta ** (torch.arange(half, device=positions.device, dtype=torch.float32) / half))
    ang = positions.float()[:, None] * inv[None, :]  # (L, half)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Rotate x (B, H, L, Dh) by absolute positions.

    q at pos a dot k at pos b only depends on a - b. That is why a rolling
    buffer of rotated keys still lines up with fresh queries. Free lunch,
    goblin approved.
    """
    half = x.shape[-1] // 2
    cos, sin = rope_cos_sin(positions, x.shape[-1], theta)  # (L, half)
    cos = cos.to(x.dtype)[None, None]
    sin = sin.to(x.dtype)[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
