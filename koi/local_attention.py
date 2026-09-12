"""Local dense attention mixer. The short range sniffer of the fish.

Every even layer of Koi is one of these: causal sliding window softmax
attention. Dense inside the window, blind past it. RoPE carries positions.
The long range memorizing is left to the Gated DeltaNet layers, this one just
does crisp cheap token mixing nearby.

Training runs blocked attention through scaled_dot_product_attention, each
query block only loads the keys it is allowed to see, so no L x L monster
table ever gets built. Inference keeps a rolling KV ring buffer, O(1) per
token with a fixed size pile of window keys.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KoiConfig
from .layers import apply_rope


def windowed_attention(q, k, v, window: int, block: int, scale: float) -> torch.Tensor:
    """Blocked causal sliding window attention.

    Query blocks of `block` tokens. Block at [qs, qe) reads keys
    [max(0, qe - window), qe). Causal AND inside the window, one bool mask.

    q, k, v: (B, H, L, Dh), returns (B, H, L, Dh)
    """
    B, H, L, Dh = q.shape
    blk = min(block, L)
    out = torch.empty_like(q)
    for qs in range(0, L, blk):
        qe = min(qs + blk, L)
        # the FIRST query in the block reaches back furthest, aim the window at it
        ks = max(0, qs - window + 1)
        qb = q[:, :, qs:qe]
        kb = k[:, :, ks:qe]
        vb = v[:, :, ks:qe]
        qpos = torch.arange(qs, qe, device=q.device)[:, None]
        kpos = torch.arange(ks, qe, device=q.device)[None, :]
        allow = (kpos <= qpos) & (qpos - kpos < window)
        o = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=allow, scale=scale)
        out[:, :, qs:qe] = o
    return out


class LocalDenseAttention(nn.Module):
    """Causal sliding window attention with RoPE. No gates, no state matrix,
    just a tidy little KV pile that rolls."""

    def __init__(self, cfg: KoiConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        self.H, self.Dh = cfg.n_heads, cfg.head_dim
        self.window = cfg.local_window
        self.theta = cfg.rope_theta
        self.scale = cfg.head_dim ** -0.5

        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)

    def _qkv(self, x: torch.Tensor):
        B, L, D = x.shape
        H, Dh = self.H, self.Dh
        q = self.q_proj(x).view(B, L, H, Dh).transpose(1, 2)  # (B, H, L, Dh)
        k = self.k_proj(x).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, Dh).transpose(1, 2)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self._qkv(x)
        pos = torch.arange(L, device=x.device)
        q = apply_rope(q, pos, self.theta)
        k = apply_rope(k, pos, self.theta)
        o = windowed_attention(q, k, v, self.window, self.cfg.chunk_size, self.scale)
        return self.o_proj(o.transpose(1, 2).reshape(B, L, D))

    def step(self, x: torch.Tensor, state):
        """One token in, one token out.

        Ring buffer holds the last `window` keys and values. New bone goes in
        the front slot, old bone falls out the back. Order inside the buffer
        does not matter, softmax is a set kind of goblin.
        """
        B, L, D = x.shape
        assert L == 1
        H, Dh, W = self.H, self.Dh, self.window
        if state is None:
            state = {
                "k": torch.zeros(B, H, W, Dh, device=x.device, dtype=x.dtype),
                "v": torch.zeros(B, H, W, Dh, device=x.device, dtype=x.dtype),
                "count": 0,  # keys in the pile so far, capped at W
                "pos": 0,    # absolute position of the incoming token
            }

        pos = state["pos"]
        q, k, v = self._qkv(x)  # (B, H, 1, Dh) each
        p = torch.tensor([pos], device=x.device, dtype=torch.long)
        q = apply_rope(q, p, self.theta)
        k = apply_rope(k, p, self.theta)

        slot = pos % W
        state["k"][:, :, slot] = k[:, :, 0]
        state["v"][:, :, slot] = v[:, :, 0]
        state["count"] = min(state["count"] + 1, W)
        state["pos"] = pos + 1

        n = state["count"]
        keys = state["k"][:, :, :n]  # view, no copy, goblin wastes nothing
        vals = state["v"][:, :, :n]

        with torch.autocast(device_type=x.device.type, enabled=False):
            # fp32 scores, softmax goblin demands elbow room
            scores = torch.einsum("bhid,bhjd->bhij", q.float(), keys.float()) * self.scale
            w = torch.softmax(scores, dim=-1)
            o = torch.einsum("bhij,bhjd->bhid", w, vals.float())

        o = o.to(x.dtype).transpose(1, 2).reshape(B, 1, D)
        return self.o_proj(o), state
