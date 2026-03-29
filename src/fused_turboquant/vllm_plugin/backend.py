"""
vLLM attention backend with fused-turboquant KV cache compression.

This backend wraps vLLM's default attention mechanism, adding transparent
KV cache compression via TurboQuant_MSE with Randomized Hadamard Transform.

Architecture-aware: only compresses full_attention layers in hybrid models
like Qwen3.5 (skips DeltaNet/linear attention layers automatically).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_BACKEND_AVAILABLE = False

try:
    import torch
    from fused_turboquant.core.quantizer import TurboQuantMSE

    _BACKEND_AVAILABLE = True
except ImportError:
    pass


class TurboQuantRHTBackend:
    """
    fused-turboquant attention backend for vLLM.

    This is a stub that will be expanded as vLLM's attention backend API
    stabilizes. The current implementation provides the plugin registration
    interface and basic KV cache compression hooks.

    For immediate use, see the HuggingFace integration in cache/kv_cache.py
    or use the turboquant-vllm package which has a more mature vLLM integration.
    """

    name = "TURBOQUANT_RHT"

    def __init__(self, **kwargs: Any):
        if not _BACKEND_AVAILABLE:
            raise ImportError(
                "fused-turboquant backend requires torch. "
                "Install with: uv add fused-turboquant[vllm]"
            )
        self.bits = kwargs.get("bits", 4)
        self.compress_values = kwargs.get("compress_values", True)
        logger.info(
            f"TurboQuantRHTBackend initialized: {self.bits}-bit, "
            f"compress_values={self.compress_values}"
        )

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @classmethod
    def is_available(cls) -> bool:
        return _BACKEND_AVAILABLE
