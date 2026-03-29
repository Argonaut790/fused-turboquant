"""
Simulation cache: compress -> decompress roundtrip on DynamicCache.

Stores the lossy fp16 result in the standard HuggingFace DynamicCache.
This does NOT save memory — it simulates the quality degradation from
TurboQuant compression so you can measure perplexity impact.

Usage:
    from fused_turboquant.hf import make_simulation_cache
    cache = make_simulation_cache(bits=4)
    outputs = model.generate(..., past_key_values=cache, use_cache=True)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import DynamicCache

from fused_turboquant.core.quantizer import TurboQuantMSE

logger = logging.getLogger(__name__)

_quantizer_pool: dict[tuple[int, int, int, str], TurboQuantMSE] = {}


def _get_quantizer(head_dim: int, bits: int, seed: int, device: str) -> TurboQuantMSE:
    key = (head_dim, bits, seed, device)
    if key not in _quantizer_pool:
        _quantizer_pool[key] = TurboQuantMSE(
            head_dim=head_dim, bits=bits, seed=seed, device=device,
        )
    return _quantizer_pool[key]


def _roundtrip(x: torch.Tensor, bits: int, layer_idx: int) -> torch.Tensor:
    """Compress then decompress a KV tensor, returning lossy fp16."""
    if x.numel() == 0:
        return x
    head_dim = x.shape[-1]
    if head_dim < 1 or (head_dim & (head_dim - 1)) != 0:
        return x
    orig_dtype = x.dtype
    device_str = str(x.device)
    tq = _get_quantizer(head_dim, bits, seed=42 + layer_idx, device=device_str)
    compressed = tq.encode(x.float())
    return tq.decode(compressed).to(orig_dtype)


class SimulationCache(DynamicCache):
    """DynamicCache that applies TurboQuant roundtrip on every update.

    Keys and optionally values are compressed then decompressed before
    being stored in the standard fp16 cache. This lets you measure
    quality degradation (perplexity, cosine sim) without changing the
    attention path.
    """

    def __init__(self, bits: int = 4, compress_values: bool = True):
        super().__init__()
        self.bits = bits
        self.compress_values = compress_values
        self._layers_seen = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_states = _roundtrip(key_states, self.bits, layer_idx * 2)
        if self.compress_values:
            value_states = _roundtrip(value_states, self.bits, layer_idx * 2 + 1)
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


def make_simulation_cache(bits: int = 4, compress_values: bool = True) -> SimulationCache:
    """Create a simulation cache for quality testing."""
    _quantizer_pool.clear()
    return SimulationCache(bits=bits, compress_values=compress_values)
