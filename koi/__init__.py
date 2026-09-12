"""Koi: hybrid LM. Odd layers Gated DeltaNet, even layers local dense."""

from .config import KoiConfig
from .gdn import GatedDeltaNet
from .local_attention import LocalDenseAttention
from .model import KoiBlock, KoiForCausalLM

__version__ = "0.2.0"

__all__ = [
    "KoiConfig",
    "KoiBlock",
    "KoiForCausalLM",
    "GatedDeltaNet",
    "LocalDenseAttention",
    "__version__",
]
