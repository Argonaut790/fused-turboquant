"""
Automatic mixed-precision per-layer bit-rate assignment (AdaptiveBits).

Runs a small calibration pass to measure per-layer quantization error at
each bit-rate (2, 3, 4), then assigns rates to meet a target quality or
memory budget. Subsumes simple "boundary" strategies with a strictly more
powerful framework.

Usage:
    from fused_turboquant.core.adaptive import calibrate_layer_bits

    bit_map = calibrate_layer_bits(
        model, tokenizer,
        target_compression=5.0,
        calibration_text="The quick brown fox jumps over the lazy dog.",
    )
    # bit_map: {0: 4, 1: 3, 2: 3, ..., 31: 4}
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from fused_turboquant.core.quantizer import TurboQuantMSE

logger = logging.getLogger(__name__)

CANDIDATE_BITS = (2, 3, 4)


def _measure_layer_error(
    hidden: torch.Tensor,
    head_dim: int,
    bits: int,
    device: str,
) -> float:
    """Measure mean cosine distance for a single layer's KV at a given bit-rate.

    Args:
        hidden: KV-like tensor of shape [num_vectors, head_dim].
        head_dim: dimension per head.
        bits: quantization bit-width.
        device: torch device string.

    Returns:
        Mean cosine distance (1 - cosine_similarity), lower is better.
    """
    tq = TurboQuantMSE(head_dim=head_dim, bits=bits, device=device)
    with torch.no_grad():
        reconstructed = tq.roundtrip(hidden.float())
    cos_sim = torch.nn.functional.cosine_similarity(
        hidden.float(),
        reconstructed,
        dim=-1,
    )
    return (1.0 - cos_sim.mean()).item()


def _compute_bytes_per_elem(head_dim: int, bits: int) -> float:
    """Bytes per KV element (packed indices + fp16 norm) for a given bit-rate."""
    if bits == 4:
        packed = head_dim // 2
    elif bits == 3:
        packed = head_dim * 3 // 8
    elif bits == 2:
        packed = head_dim // 4
    else:
        packed = head_dim
    return packed + 2  # fp16 norm


def calibrate_layer_bits(
    model,
    tokenizer=None,
    head_dim: Optional[int] = None,
    calibration_text: str = "The quick brown fox jumps over the lazy dog. " * 8,
    calibration_ids: Optional[torch.Tensor] = None,
    target_compression: Optional[float] = None,
    quality_target: Optional[float] = 0.99,
    candidate_bits: tuple[int, ...] = CANDIDATE_BITS,
) -> dict[int, int]:
    """Calibrate per-layer bit-rates using a small forward pass.

    Runs the model on calibration text, captures KV activations per layer,
    measures quantization error at each candidate bit-rate, then assigns
    rates to optimize quality within a compression budget (or vice versa).

    Args:
        model: HuggingFace CausalLM model.
        tokenizer: tokenizer for calibration_text. Not needed if calibration_ids given.
        head_dim: override head dimension. Auto-detected if None.
        calibration_text: text for calibration forward pass.
        calibration_ids: pre-tokenized input_ids [1, seq_len]. Overrides text.
        target_compression: target average KV compression ratio (e.g., 5.0).
            If set, quality_target is ignored.
        quality_target: target mean cosine similarity (e.g., 0.99).
            Layers are assigned the lowest bit-rate that meets this threshold.
        candidate_bits: bit-rates to evaluate, sorted ascending.

    Returns:
        Dict mapping layer_idx -> bits.
    """
    from fused_turboquant.hf.fused_cache import (
        _is_full_attention_layer,
        _resolve_config,
        _resolve_head_dim,
    )

    config = _resolve_config(model)
    if head_dim is None:
        head_dim = _resolve_head_dim(config)
    if head_dim == 0:
        raise ValueError("Could not detect head_dim from model config.")

    device = next(model.parameters()).device

    if calibration_ids is None:
        if tokenizer is None:
            raise ValueError("Provide either tokenizer or calibration_ids.")
        tokens = tokenizer(calibration_text, return_tensors="pt")
        calibration_ids = tokens["input_ids"].to(device)

    kv_captures: dict[int, dict[str, torch.Tensor]] = {}
    hooks = []
    layer_idx = 0

    for name, module in model.named_modules():
        if not _is_full_attention_layer(module, name):
            continue
        idx = layer_idx

        def make_hook(li):
            def hook_fn(mod, args, output):
                if hasattr(mod, "k_proj") and hasattr(mod, "v_proj"):
                    hidden = args[0] if isinstance(args, tuple) else args
                    if isinstance(hidden, torch.Tensor):
                        with torch.no_grad():
                            k = mod.k_proj(hidden)
                            v = mod.v_proj(hidden)
                        kv_captures[li] = {
                            "k": k.detach().reshape(-1, head_dim),
                            "v": v.detach().reshape(-1, head_dim),
                        }

            return hook_fn

        h = module.register_forward_hook(make_hook(idx))
        hooks.append(h)
        layer_idx += 1

    n_layers = layer_idx

    with torch.inference_mode():
        model(calibration_ids)

    for h in hooks:
        h.remove()

    errors: dict[int, dict[int, float]] = {}
    device_str = str(device)
    for li in range(n_layers):
        if li not in kv_captures:
            logger.warning("Layer %d: no KV captured, defaulting to 4-bit", li)
            errors[li] = {b: 0.0 for b in candidate_bits}
            continue
        cap = kv_captures[li]
        kv_data = torch.cat([cap["k"], cap["v"]], dim=0)
        errors[li] = {}
        for b in candidate_bits:
            errors[li][b] = _measure_layer_error(kv_data, head_dim, b, device_str)

    bit_map: dict[int, int] = {}

    if target_compression is not None:
        fp16_bytes = head_dim * 2
        target_avg_bytes = fp16_bytes / target_compression

        layer_options = []
        for li in range(n_layers):
            opts = []
            for b in sorted(candidate_bits):
                bpe = _compute_bytes_per_elem(head_dim, b)
                err = errors[li].get(b, 1.0)
                opts.append((b, bpe, err))
            layer_options.append(opts)

        for li in range(n_layers):
            bit_map[li] = max(candidate_bits)

        sorted_gains = []
        for li in range(n_layers):
            opts = layer_options[li]
            for i in range(len(opts) - 1):
                lower_bits, lower_bpe, lower_err = opts[i]
                higher_bits, higher_bpe, higher_err = opts[i + 1]
                byte_saved = higher_bpe - lower_bpe
                quality_cost = lower_err - higher_err
                if byte_saved > 0:
                    sorted_gains.append((quality_cost / byte_saved, li, lower_bits, byte_saved))

        sorted_gains.sort(key=lambda x: x[0])

        current_avg = (
            sum(_compute_bytes_per_elem(head_dim, bit_map[li]) for li in range(n_layers)) / n_layers
        )

        for _, li, new_bits, _ in sorted_gains:
            if current_avg <= target_avg_bytes:
                break
            old_bits = bit_map[li]
            if new_bits >= old_bits:
                continue
            old_bpe = _compute_bytes_per_elem(head_dim, old_bits)
            new_bpe = _compute_bytes_per_elem(head_dim, new_bits)
            current_avg -= (old_bpe - new_bpe) / n_layers
            bit_map[li] = new_bits

    else:
        cos_threshold = 1.0 - (quality_target or 0.99)
        for li in range(n_layers):
            assigned = max(candidate_bits)
            for b in sorted(candidate_bits):
                if errors[li].get(b, 1.0) <= cos_threshold:
                    assigned = b
                    break
            bit_map[li] = assigned

    bit_counts = {}
    for b in bit_map.values():
        bit_counts[b] = bit_counts.get(b, 0) + 1
    avg_bits = sum(bit_map.values()) / max(len(bit_map), 1)
    logger.info(
        "AdaptiveBits: %d layers, avg %.2f bits, distribution: %s",
        n_layers,
        avg_bits,
        bit_counts,
    )

    return bit_map


def adaptive_compress_v_fn(
    bit_map: dict[int, int],
    compress_v_at: Optional[set[int]] = None,
):
    """Create a per-layer compress_v callable compatible with patch_model.

    If compress_v_at is None, compresses V at all layers.
    """
    if compress_v_at is None:
        return lambda layer_idx, n_layers: True
    return lambda layer_idx, n_layers: layer_idx in compress_v_at
