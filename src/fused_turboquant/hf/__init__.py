"""HuggingFace transformers integration for fused-turboquant."""

from fused_turboquant.core.adaptive import calibrate_layer_bits
from fused_turboquant.hf.fused_cache import (
    CompressedKVCache,
    FusedTurboQuantRunner,
    check_model_compatibility,
    patch_model,
    unpatch_model,
)

__all__ = [
    "CompressedKVCache",
    "check_model_compatibility",
    "patch_model",
    "unpatch_model",
    "FusedTurboQuantRunner",
    "calibrate_layer_bits",
]
