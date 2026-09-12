"""Koi, a hybrid language model. Named for the fish that swims through delta
water without losing a scale.

Every odd layer (1st, 3rd, ...) is a Gated DeltaNet implementation: linear
attention with a fixed size recurrent state, channel wise decay gates,
chunkwise parallel training, O(1) decode. Every even layer is local dense
attention: causal sliding window softmax attention with RoPE. Interleave the
two and you get long range memory at linear cost plus sharp short range
mixing. That is the whole goblin plot.

Decode keeps per layer state only: a fixed matrix hoard for the Gated DeltaNet
layers, a rolling window of keys for the local layers. No KV cache that grows
with the sequence, ever.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KoiConfig
from .gdn import GatedDeltaNet
from .layers import RMSNorm, SwiGLU
from .local_attention import LocalDenseAttention


class KoiBlock(nn.Module):
    """Pre-norm block: mixer, then SwiGLU. Mixer type is decided at birth."""

    def __init__(self, cfg: KoiConfig, layer_type: str):
        super().__init__()
        assert layer_type in ("gdn", "local"), f"unknown layer type {layer_type}"
        self.layer_type = layer_type
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.mixer = GatedDeltaNet(cfg) if layer_type == "gdn" else LocalDenseAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def step(self, x, state):
        m_state = None if state is None else state.get("mixer")
        h, m_state = self.mixer.step(self.norm1(x), m_state)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, {"mixer": m_state}


class KoiForCausalLM(nn.Module):
    """The full fish. Odd layers Gated DeltaNet, even layers local dense."""

    def __init__(self, cfg: KoiConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [KoiBlock(cfg, t) for t in cfg.layer_types]
        )
        self.norm_f = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)
        # the blanket init tramples the lazy gate init above, so re-stomp back.
        # trampled hoards make sad goblins and sadder loss curves
        for m in self.modules():
            if isinstance(m, GatedDeltaNet):
                m.reset_gates()

        self.gradient_checkpointing = False

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def layer_plan(self) -> str:
        """Short plan string like G,L,G,L. Goblin summary rock."""
        return ",".join({"gdn": "G", "local": "L"}[t] for t in self.cfg.layer_types)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                ignore_index: int = -100):
        x = self.embed(input_ids)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1),
                ignore_index=ignore_index,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 300,
                 temperature: float = 0.8, top_k: Optional[int] = 40,
                 top_p: Optional[float] = None, eot_id: Optional[int] = None):
        """Autoregressive generation, O(1) per token.

        Prefill walks the prompt through the recurrent path one token at a
        time. One code path for prefill and decode, so train and infer never
        drift apart. GDN layers hold a fixed state, local layers hold a window
        sized pile. Total cache: fixed. Goblin pockets stay light forever.
        """
        self.eval()
        B, L = input_ids.shape
        states: List[Optional[Dict]] = [None] * len(self.blocks)

        x_emb = self.embed(input_ids)
        last_logits = None
        for t in range(L):
            xt = x_emb[:, t:t + 1, :]
            for i, block in enumerate(self.blocks):
                xt, states[i] = block.step(xt, states[i])
            h = self.norm_f(xt)
            last_logits = self.lm_head(h)

        generated = input_ids
        for _ in range(max_new_tokens):
            logits = last_logits[:, -1, :].float() / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                remove = cum > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits[remove] = -float("inf")
                logits = torch.full_like(logits, -float("inf")).scatter(1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)

            if eot_id is not None and (next_id == eot_id).all():
                break

            xt = self.embed(next_id)
            for i, block in enumerate(self.blocks):
                xt, states[i] = block.step(xt, states[i])
            h = self.norm_f(xt)
            last_logits = self.lm_head(h)

        return generated

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
