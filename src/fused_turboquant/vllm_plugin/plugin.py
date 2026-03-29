"""
vLLM plugin registration for fused-turboquant attention backend.

Registers via vLLM's entry point system — enable with:
    vllm serve <model> --attention-backend TURBOQUANT_RHT

This module provides the plugin hook. The actual attention backend
implementation is in backend.py.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register_backend() -> None:
    """
    vLLM plugin entry point. Called automatically when fused-turboquant is installed
    and vLLM loads plugins.

    Registers the fused-turboquant attention backend so it can be selected via
    --attention-backend TURBOQUANT_RHT.
    """
    try:
        from fused_turboquant.vllm_plugin.backend import TurboQuantRHTBackend

        try:
            from vllm.attention.backends.registry import AttentionBackendRegistry

            AttentionBackendRegistry.register("TURBOQUANT_RHT", TurboQuantRHTBackend)
            logger.info("fused-turboquant attention backend registered with vLLM")
        except ImportError:
            logger.warning(
                "vLLM attention backend registry not found. "
                "fused-turboquant plugin requires vLLM >= 0.8. "
                "The plugin will not be available."
            )
    except Exception as e:
        logger.warning(f"Failed to register fused-turboquant backend: {e}")
