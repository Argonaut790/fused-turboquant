"""
TurboQuant KV Cache integration.

Patches DynamicCache.update() to quantize key/value states before storage.

Usage:
    from turboquant_kv_cache import TurboQuantWrapper
    wrapper = TurboQuantWrapper(model, bits=4)
    outputs = wrapper.generate(input_ids, max_new_tokens=200)
"""

import torch
from transformers import DynamicCache
from turboquant_core import TurboQuantMSE

_quantizer_cache: dict[tuple[int, int, str], TurboQuantMSE] = {}


def _get_quantizer(head_dim: int, bits: int, device: str) -> TurboQuantMSE:
    key = (head_dim, bits, device)
    if key not in _quantizer_cache:
        _quantizer_cache[key] = TurboQuantMSE(d=head_dim, bits=bits, device=device)
    return _quantizer_cache[key]


def _quantize_dequantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Roundtrip: quantize then dequantize. Simulates lossy storage."""
    if x.numel() == 0:
        return x
    tq = _get_quantizer(x.shape[-1], bits, str(x.device))
    orig_dtype = x.dtype
    q = tq.quantize(x.float())
    return tq.dequantize(q).to(orig_dtype)


def make_quantized_cache(bits: int = 4, quantize_values: bool = True) -> DynamicCache:
    cache = DynamicCache()
    _original_update = cache.update

    def _patched_update(key_states, value_states, layer_idx, cache_kwargs=None):
        key_states = _quantize_dequantize(key_states, bits)
        if quantize_values:
            value_states = _quantize_dequantize(value_states, bits)
        return _original_update(key_states, value_states, layer_idx, cache_kwargs)

    cache.update = _patched_update
    return cache


class TurboQuantWrapper:
    def __init__(self, model, bits: int = 4, quantize_values: bool = True):
        self.model = model
        self.bits = bits
        self.quantize_values = quantize_values

    def generate(self, *args, **kwargs):
        cache = make_quantized_cache(self.bits, self.quantize_values)
        kwargs["past_key_values"] = cache
        kwargs["use_cache"] = True
        return self.model.generate(*args, **kwargs)

    @property
    def device(self):
        return next(self.model.parameters()).device
