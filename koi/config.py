"""Koi config. One dataclass to rule the fish."""

from dataclasses import dataclass
from typing import List


@dataclass
class KoiConfig:
    # goblin dials. turn them and the fish changes shape
    vocab_size: int = 256
    pad_id: int = 0
    eot_id: int = 1
    context_len: int = 16384

    d_model: int = 1024
    n_layers: int = 12
    n_heads: int = 8
    head_dim: int = 128            # d_model must equal n_heads * head_dim, goblin math
    d_ff: int = 2752               # SwiGLU hidden, ~8/3 * d_model rounded to 64
    conv_kernel: int = 4
    gate_low_rank: int = 64        # low rank bottleneck for the decay gate
    chunk_size: int = 128          # training chunk size for the delta rule
    local_window: int = 1024       # sliding window width for local dense layers
    rope_theta: float = 10000.0
    dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self):
        assert self.d_model == self.n_heads * self.head_dim, \
            "d_model must equal n_heads * head_dim"
        assert self.context_len % self.chunk_size == 0, \
            "context_len must be divisible by chunk_size"
        assert self.local_window >= 1, "local_window must be at least 1"

    @property
    def layer_types(self) -> List[str]:
        """The layer plan.

        Odd layers (1st, 3rd, ...) are Gated DeltaNet, even layers are local
        dense attention. Goblin counts from one, like a normal person, so the
        code uses i % 2 == 0 for the odd ones.
        """
        return ["gdn" if i % 2 == 0 else "local" for i in range(self.n_layers)]
