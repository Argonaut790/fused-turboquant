"""
Fused TurboQuant cache with compressed KV storage and fused attention.

Stores keys in compressed form (uint8 indices + fp32 norms) and computes
Q @ K^T directly from compressed keys using our Triton fused attention kernel.
Values are also compressed (packed indices + fp32 norms) and decompressed on
the fly during the attention-weighted sum.

This is a real integration that changes the attention computation path:
- Keys are compressed via fused Triton encode kernel
- Values are compressed and stored in packed form (nibble/2-bit packed)
- Queries are pre-rotated via RHT (not dense QR matmul)
- Q @ K^T is computed from compressed indices via fused Triton kernel
- Values are decompressed from packed storage before softmax @ V matmul

Usage:
    from fused_turboquant.hf import patch_model, FusedTurboQuantRunner
    cache = patch_model(model, bits=4)
    outputs = model.generate(..., past_key_values=cache, use_cache=True)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from transformers import DynamicCache

from fused_turboquant.core.quantizer import CompressedTensor, TurboQuantMSE

logger = logging.getLogger(__name__)

KNOWN_COMPATIBLE = {
    # Dense decoder-only models
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2_5ForCausalLM",
    "Qwen3ForCausalLM",
    "GemmaForCausalLM",
    "InternLMForCausalLM",
    "InternLM2ForCausalLM",
    "YiForCausalLM",
    "BaichuanForCausalLM",
    # MoE models (attention layers identical to dense variant)
    "Qwen2MoeForCausalLM",
    "Qwen3MoeForCausalLM",
    "OlmoeForCausalLM",
    # Multimodal (text decoder patched, vision encoder skipped)
    "Qwen2VLForConditionalGeneration",
    "InternVLForConditionalGeneration",
}


def _config_uses_rope(config) -> bool:
    """Return True if the model config indicates Rotary Position Embeddings."""
    if getattr(config, "rope_theta", None) is not None:
        return True
    if getattr(config, "rope_type", None) is not None:
        return True
    if getattr(config, "rope_scaling", None) is not None:
        return True
    pos_type = getattr(config, "position_embedding_type", None)
    if pos_type is not None:
        return pos_type.lower() in ("rope", "rotary")
    return False


def _find_output_proj(module) -> str | None:
    """Return the attribute name of the output projection, or None."""
    for name in ("o_proj", "out_proj"):
        if hasattr(module, name):
            return name
    return None


def _probe_attention_module(module, config) -> dict:
    """Inspect an attention module for features that affect patching.

    Returns a dict describing the module's architecture features so that
    make_fused_attention_forward() can reject unsupported configurations
    loudly rather than producing silent garbage.
    """
    return {
        "has_separate_qkv": all(hasattr(module, p) for p in ("q_proj", "k_proj", "v_proj")),
        "has_fused_qkv": hasattr(module, "qkv_proj") or hasattr(module, "c_attn"),
        "output_proj": _find_output_proj(module),
        "is_cross_attention": getattr(module, "is_cross_attention", False),
        "sliding_window": (
            getattr(module, "sliding_window", None) or getattr(config, "sliding_window", None)
        ),
        "has_qk_norm": hasattr(module, "q_norm") or hasattr(module, "k_norm"),
        "attn_logit_softcapping": getattr(module, "attn_logit_softcapping", None),
        "rope_expected": _config_uses_rope(config),
    }


class CompressedKVCache(DynamicCache):
    """KV cache that stores compressed keys and values.

    Both keys and values are stored in packed form (nibble-packed for 4-bit,
    bitstream-packed for 3-bit, 2-bit packed for 2-bit). The fused Triton
    attention kernel unpacks key indices inline via shift+mask (no separate
    dequantization pass). Values are decompressed in bulk before the matmul.

    Supports per-layer quantizers for adaptive mixed-precision (AdaptiveBits).

    A minimal dummy tensor is passed to DynamicCache.update() so that
    transformers' internal bookkeeping (get_seq_length, etc.) stays correct.
    """

    def __init__(self, quantizer: TurboQuantMSE, compress_v: bool = True):
        super().__init__()
        self.tq = quantizer
        self._layer_tq: dict[int, TurboQuantMSE] = {}
        self.compress_v = compress_v
        self._compressed_keys: list[Optional[dict]] = []
        self._compressed_values: list[Optional[dict]] = []

    def set_layer_quantizer(self, layer_idx: int, tq: TurboQuantMSE) -> None:
        """Register a per-layer quantizer for adaptive mixed-precision."""
        self._layer_tq[layer_idx] = tq

    def get_layer_quantizer(self, layer_idx: int) -> TurboQuantMSE:
        """Get the quantizer for a layer, falling back to the default."""
        return self._layer_tq.get(layer_idx, self.tq)

    # -- Key compression (packed uint8, unpacked inline by fused kernel) ------

    def store_compressed_key(self, key_states: torch.Tensor, layer_idx: int):
        """Compress and store key states in packed form.

        The fused attention kernel unpacks nibbles/2-bit inline, so we keep
        K packed just like V for maximum memory density.
        """
        while len(self._compressed_keys) <= layer_idx:
            self._compressed_keys.append(None)

        tq = self.get_layer_quantizer(layer_idx)
        compressed = tq.encode(key_states.float())

        packed_shape = list(key_states.shape[:-1]) + [compressed.indices.shape[-1]]
        packed_indices = compressed.indices.view(packed_shape)
        norms = compressed.norms.view(*key_states.shape[:-1])

        entry = {"packed_indices": packed_indices, "norms": norms}

        if self._compressed_keys[layer_idx] is None:
            self._compressed_keys[layer_idx] = entry
        else:
            prev = self._compressed_keys[layer_idx]
            self._compressed_keys[layer_idx] = {
                "packed_indices": torch.cat(
                    [prev["packed_indices"], entry["packed_indices"]],
                    dim=2,
                ),
                "norms": torch.cat([prev["norms"], entry["norms"]], dim=2),
            }

    def get_compressed_key(self, layer_idx: int) -> Optional[dict]:
        if layer_idx < len(self._compressed_keys):
            return self._compressed_keys[layer_idx]
        return None

    # -- Value compression (packed indices for memory efficiency) -------------

    def store_compressed_value(self, value_states: torch.Tensor, layer_idx: int):
        """Compress and store value states in packed form."""
        while len(self._compressed_values) <= layer_idx:
            self._compressed_values.append(None)

        tq = self.get_layer_quantizer(layer_idx)
        compressed = tq.encode(value_states.float())

        packed_shape = list(value_states.shape[:-1]) + [compressed.indices.shape[-1]]
        packed_indices = compressed.indices.view(packed_shape)
        norms = compressed.norms.view(*value_states.shape[:-1])

        entry = {"packed_indices": packed_indices, "norms": norms}

        if self._compressed_values[layer_idx] is None:
            self._compressed_values[layer_idx] = entry
        else:
            prev = self._compressed_values[layer_idx]
            self._compressed_values[layer_idx] = {
                "packed_indices": torch.cat(
                    [prev["packed_indices"], entry["packed_indices"]],
                    dim=2,
                ),
                "norms": torch.cat([prev["norms"], entry["norms"]], dim=2),
            }

    def get_compressed_value(self, layer_idx: int) -> Optional[dict]:
        if layer_idx < len(self._compressed_values):
            return self._compressed_values[layer_idx]
        return None

    def decode_values(self, layer_idx: int) -> torch.Tensor:
        """Decompress all cached value vectors for a layer.

        Returns tensor of shape [batch, n_kv_heads, kv_len, head_dim] in float32.
        """
        tq = self.get_layer_quantizer(layer_idx)
        entry = self._compressed_values[layer_idx]
        ct = CompressedTensor(
            indices=entry["packed_indices"],
            norms=entry["norms"],
            original_dim=tq.head_dim,
            bits=tq.bits,
        )
        return tq.decode(ct)

    # -- Reset ----------------------------------------------------------------

    def reset(self):
        """Clear all cached state so the same object can be reused for a new prompt.

        We drop layer objects entirely instead of calling super().reset() because
        the parent's reset() zeroes tensors in-place, which fails on inference
        tensors created during torch.inference_mode().
        """
        self._compressed_keys.clear()
        self._compressed_values.clear()
        self.layers.clear()


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply RoPE."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA."""
    if n_rep == 1:
        return hidden_states
    batch, n_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, n_kv_heads * n_rep, slen, head_dim)


def make_fused_attention_forward(
    attn_module,
    cache: CompressedKVCache,
    quantizer: TurboQuantMSE,
    layer_idx: int,
    config=None,
    compress_v: bool = True,
):
    """Create a replacement forward for an attention layer that uses fused TurboQuant.

    Validates that the module's architecture is supported before creating the
    fused forward closure.  Raises ValueError for unsupported features (fused
    QKV, sliding window, QK norm, logit softcapping, cross-attention) so that
    users get a clear error instead of silent garbage output.
    """
    if config is not None:
        probe = _probe_attention_module(attn_module, config)

        if probe["is_cross_attention"]:
            raise ValueError(
                f"Layer {layer_idx}: cross-attention layers cannot be patched. "
                f"fused-turboquant only supports decoder self-attention."
            )

        if probe["has_fused_qkv"] and not probe["has_separate_qkv"]:
            fused_name = "qkv_proj" if hasattr(attn_module, "qkv_proj") else "c_attn"
            raise ValueError(
                f"Layer {layer_idx}: fused QKV projection ({fused_name}) is not "
                f"supported. fused-turboquant requires separate q_proj, k_proj, "
                f"v_proj linear layers."
            )

        if probe["sliding_window"] is not None:
            logger.info(
                "Layer %d: sliding window attention (window=%s) detected. "
                "fused-turboquant will compress the full KV cache and apply "
                "the window mask during attention computation.",
                layer_idx,
                probe["sliding_window"],
            )

        if probe["attn_logit_softcapping"] is not None:
            raise ValueError(
                f"Layer {layer_idx}: attention logit softcapping "
                f"(value={probe['attn_logit_softcapping']}) is not yet supported."
            )

        if not probe["rope_expected"]:
            logger.warning(
                "Layer %d: model config does not indicate RoPE usage. "
                "If this model uses ALiBi, learned positional embeddings, or no "
                "positional encoding in attention, the fused attention path will "
                "produce incorrect results. Proceed with caution.",
                layer_idx,
            )

    from fused_turboquant.kernels.triton_attention import fused_qk_scores_rht

    rht_signs = quantizer.rotation.signs
    centroids = quantizer.quantizer.levels
    head_dim = quantizer.head_dim
    bits = quantizer.bits
    scale = 1.0 / math.sqrt(head_dim)

    n_heads = getattr(attn_module, "num_heads", None)
    if n_heads is None:
        n_heads = attn_module.q_proj.out_features // head_dim
    n_kv_heads = getattr(attn_module, "num_key_value_heads", None)
    if n_kv_heads is None:
        n_kv_heads = attn_module.k_proj.out_features // head_dim
    n_kv_groups = n_heads // n_kv_heads

    sliding_window = getattr(attn_module, "sliding_window", None) or (
        getattr(config, "sliding_window", None) if config else None
    )
    n_sink_tokens = getattr(config, "sink_tokens", 4) if config else 4

    q_norm = getattr(attn_module, "q_norm", None)
    k_norm = getattr(attn_module, "k_norm", None)

    def fused_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = attn_module.q_proj(hidden_states)
        key_states = attn_module.k_proj(hidden_states)
        value_states = attn_module.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

        if q_norm is not None:
            query_states = q_norm(query_states)
        if k_norm is not None:
            key_states = k_norm(key_states)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = _apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )

        cache.store_compressed_key(key_states, layer_idx)
        if compress_v:
            cache.store_compressed_value(value_states, layer_idx)

        # Pass minimal dummy slices to DynamicCache.update() for seq_length
        # bookkeeping. Full keys live in _compressed_keys, full values in
        # _compressed_values (when compress_v=True).
        dummy_keys = key_states[:, :, :, :1]
        if compress_v:
            dummy_values = value_states[:, :, :, :1]
            cache.update(dummy_keys, dummy_values, layer_idx)
        else:
            _, full_values = cache.update(dummy_keys, value_states, layer_idx)

        if q_len == 1:
            compressed = cache.get_compressed_key(layer_idx)

            from fused_turboquant.core.hadamard import randomized_hadamard

            q_flat = query_states.float().reshape(-1, head_dim)
            q_rot = randomized_hadamard(q_flat, rht_signs)
            q_rot = q_rot.view_as(query_states)

            attn_weights = fused_qk_scores_rht(
                q_rot,
                compressed["packed_indices"],
                compressed["norms"],
                centroids,
                scale,
                bits=bits,
            )

            kv_len = compressed["packed_indices"].shape[2]

            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    attn_weights = attn_weights + attention_mask[:, :, :1, :kv_len]
                elif attention_mask.dim() == 2:
                    attn_weights = attn_weights + attention_mask[:1, :kv_len]

            if sliding_window is not None and kv_len > sliding_window:
                window_mask = torch.full(
                    (1, 1, 1, kv_len),
                    float("-inf"),
                    device=attn_weights.device,
                    dtype=attn_weights.dtype,
                )
                window_start = max(0, kv_len - sliding_window)
                window_mask[:, :, :, window_start:] = 0
                if n_sink_tokens > 0:
                    window_mask[:, :, :, :n_sink_tokens] = 0
                attn_weights = attn_weights + window_mask

            attn_weights = torch.nn.functional.softmax(
                attn_weights,
                dim=-1,
                dtype=torch.float32,
            ).to(query_states.dtype)

            if compress_v:
                decoded_v = cache.decode_values(layer_idx).to(query_states.dtype)
                full_values_expanded = _repeat_kv(decoded_v, n_kv_groups)
            else:
                full_values_expanded = _repeat_kv(full_values, n_kv_groups)
            attn_output = torch.matmul(attn_weights, full_values_expanded)
        else:
            # Prefill path: use Flash/SDPA on full FP16 keys and values to
            # avoid O(n^2) memory. KV are already compressed and stored above
            # for subsequent decode steps.
            full_keys_expanded = _repeat_kv(key_states, n_kv_groups)
            full_values_expanded = _repeat_kv(value_states, n_kv_groups)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                full_keys_expanded,
                full_values_expanded,
                is_causal=True,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        o_proj = getattr(attn_module, "o_proj", None) or getattr(attn_module, "out_proj", None)
        if o_proj is not None:
            attn_output = o_proj(attn_output)

        return attn_output, None

    return fused_forward


_SKIP_NAME_KEYWORDS = (
    "encoder_attn",
    "crossattention",
    "cross_attn",
    "visual",
    "vision_model",
    "vision_tower",
    "image_encoder",
    "vit",
    "img_attn",
)


def _is_full_attention_layer(module, name: str = "") -> bool:
    """Detect if a module is a patchable self-attention layer.

    Rejects cross-attention modules, vision encoder layers, and modules
    that lack separate Q/K/V projections.
    """
    if getattr(module, "is_cross_attention", False):
        return False
    name_lower = name.lower()
    if any(kw in name_lower for kw in _SKIP_NAME_KEYWORDS):
        return False

    required = ["q_proj", "k_proj", "v_proj"]
    output = ["o_proj", "out_proj"]
    has_qkv = all(hasattr(module, attr) for attr in required)
    has_output = any(hasattr(module, attr) for attr in output)
    return has_qkv and has_output


def _resolve_head_dim(config) -> int:
    """Extract head_dim from a HuggingFace model config."""
    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None:
        return head_dim
    hidden_size = getattr(config, "hidden_size", None)
    num_heads = getattr(config, "num_attention_heads", None)
    if hidden_size is not None and num_heads is not None and num_heads > 0:
        return hidden_size // num_heads
    return 0


def _resolve_config(model):
    """Get the text config from a (possibly multimodal) HuggingFace model."""
    config = model.config
    if hasattr(config, "text_config"):
        config = config.text_config
    return config


def check_model_compatibility(model) -> dict:
    """Check whether a HuggingFace model is compatible with fused-turboquant.

    Returns a dict with:
        - compatible (bool): True if patch_model can be used
        - head_dim_valid (bool): True if head_dim is a power of 2
        - head_dim (int): detected head dimension
        - n_q_heads (int): number of query heads
        - n_kv_heads (int): number of KV heads
        - eligible_layers (int): number of layers that would be patched
        - total_layers (int): total number of submodules scanned
        - issues (list[str]): human-readable list of problems found
        - rope_detected (bool): whether config indicates RoPE usage
        - sliding_window (int | None): detected sliding window config
        - unsupported_features (list[str]): features that would block patching
        - fused_qkv_layers (int): layers with fused QKV (not patchable)
        - cross_attention_layers (int): cross-attention layers (skipped)
        - vision_layers_skipped (int): vision encoder attention layers (skipped)
        - architecture (str): model class name
        - known_compatible (bool): whether architecture is in the tested set
    """
    config = _resolve_config(model)
    head_dim = _resolve_head_dim(config)
    n_q_heads = getattr(config, "num_attention_heads", 0)
    n_kv_heads = getattr(config, "num_key_value_heads", n_q_heads)

    arch_name = type(model).__name__
    rope_detected = _config_uses_rope(config)
    sliding_window = getattr(config, "sliding_window", None)

    issues: list[str] = []
    unsupported: list[str] = []
    eligible = 0
    total = 0
    fused_qkv_layers = 0
    cross_attention_layers = 0
    vision_layers_skipped = 0

    for _name, module in model.named_modules():
        total += 1

        if getattr(module, "is_cross_attention", False):
            cross_attention_layers += 1
            continue

        name_lower = _name.lower()
        if any(kw in name_lower for kw in _SKIP_NAME_KEYWORDS):
            has_qkv = all(hasattr(module, p) for p in ("q_proj", "k_proj", "v_proj"))
            if has_qkv:
                vision_layers_skipped += 1
            continue

        if hasattr(module, "qkv_proj") or hasattr(module, "c_attn"):
            if not all(hasattr(module, p) for p in ("q_proj", "k_proj", "v_proj")):
                fused_qkv_layers += 1

        if _is_full_attention_layer(module, _name):
            probe = _probe_attention_module(module, config)
            has_softcap = probe["attn_logit_softcapping"] is not None
            if has_softcap and "logit_softcapping" not in unsupported:
                unsupported.append("logit_softcapping")
            eligible += 1

    is_power_of_2 = head_dim >= 1 and (head_dim & (head_dim - 1)) == 0

    if head_dim == 0:
        issues.append("Could not detect head_dim from model config")
    elif not is_power_of_2:
        issues.append(f"head_dim={head_dim} is not a power of 2 — RHT requires 64, 128, 256, etc.")

    if n_kv_heads > 0 and n_q_heads % n_kv_heads != 0:
        issues.append(
            f"n_q_heads ({n_q_heads}) is not divisible by n_kv_heads ({n_kv_heads}) — "
            f"GQA grouping requires integer divisibility"
        )

    if eligible == 0:
        if fused_qkv_layers > 0:
            issues.append(
                f"No compatible attention layers found. Detected {fused_qkv_layers} "
                f"layer(s) with fused QKV projection (qkv_proj/c_attn), which is not "
                f"supported — separate q_proj, k_proj, v_proj are required."
            )
        else:
            issues.append(
                "No compatible attention layers found (need separate q_proj, "
                "k_proj, v_proj and o_proj/out_proj projections)"
            )

    if not rope_detected:
        issues.append(
            "Model config does not indicate RoPE usage. fused-turboquant requires "
            "models that use Rotary Position Embeddings."
        )

    if unsupported:
        issues.append(
            f"Unsupported attention features detected: {', '.join(unsupported)}. "
            f"These would cause incorrect results."
        )

    compatible = is_power_of_2 and eligible > 0 and len(issues) == 0 and len(unsupported) == 0

    return {
        "compatible": compatible,
        "head_dim_valid": is_power_of_2,
        "head_dim": head_dim,
        "n_q_heads": n_q_heads,
        "n_kv_heads": n_kv_heads,
        "eligible_layers": eligible,
        "total_layers": total,
        "issues": issues,
        "rope_detected": rope_detected,
        "sliding_window": sliding_window,
        "unsupported_features": unsupported,
        "fused_qkv_layers": fused_qkv_layers,
        "cross_attention_layers": cross_attention_layers,
        "vision_layers_skipped": vision_layers_skipped,
        "architecture": arch_name,
        "known_compatible": arch_name in KNOWN_COMPATIBLE,
    }


def _smoke_test(
    model,
    cache: CompressedKVCache,
    originals: dict[str, object],
    config,
    head_dim: int,
) -> None:
    """Run a single-token forward pass and verify fused output is reasonable.

    Compares cosine similarity of logits between the fused and original
    attention paths.  Raises RuntimeError if the similarity is too low,
    which signals a silent correctness bug (wrong RoPE, missing mask, bad
    head mapping, etc.).

    The model is left in its patched state with a clean cache on return.
    """
    device = next(model.parameters()).device
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        logger.debug("Smoke test skipped: could not detect hidden_size")
        return

    vocab_size = getattr(config, "vocab_size", 32000)
    dummy_ids = torch.randint(0, vocab_size, (1, 1), device=device)

    try:
        with torch.inference_mode():
            fused_out = model(dummy_ids, past_key_values=cache, use_cache=True)
            fused_logits = fused_out.logits[0, -1].float()
    except Exception as exc:
        cache.reset()
        raise RuntimeError(
            f"Smoke test failed: fused forward raised {type(exc).__name__}: {exc}. "
            f"This model architecture may not be compatible with fused-turboquant. "
            f"Use check_model_compatibility(model) for details."
        ) from exc

    cache.reset()

    fused_forwards: dict[str, object] = {}
    for name, module in model.named_modules():
        if name in originals:
            fused_forwards[name] = module.forward
            module.forward = originals[name]

    try:
        with torch.inference_mode():
            ref_out = model(dummy_ids, use_cache=False)
            ref_logits = ref_out.logits[0, -1].float()
    except Exception:
        logger.debug("Smoke test skipped: reference forward failed")
        for name, module in model.named_modules():
            if name in fused_forwards:
                module.forward = fused_forwards[name]
        return
    finally:
        for name, module in model.named_modules():
            if name in fused_forwards:
                module.forward = fused_forwards[name]

    cos_sim = torch.nn.functional.cosine_similarity(
        fused_logits.unsqueeze(0),
        ref_logits.unsqueeze(0),
    ).item()

    if cos_sim < 0.8:
        raise RuntimeError(
            f"Smoke test failed: cosine similarity between fused and reference "
            f"logits is {cos_sim:.4f} (threshold: 0.8). This indicates a "
            f"correctness bug in the fused attention path for this model "
            f"architecture ({type(model).__name__}). "
            f"Use check_model_compatibility(model) for details, or pass "
            f"verify=False to skip this check."
        )

    logger.info(
        "Smoke test passed: logit cosine similarity = %.4f",
        cos_sim,
    )


def _resolve_compress_v(compress_v, layer_idx: int, n_layers: int) -> bool:
    """Resolve per-layer V compression decision.

    Supports bool, callable, or preset strings for flexible layer-aware
    compression strategies.
    """
    if isinstance(compress_v, bool):
        return compress_v
    if callable(compress_v):
        return compress_v(layer_idx, n_layers)
    if compress_v == "boundary":
        return 2 <= layer_idx < n_layers - 2
    raise ValueError(
        f"compress_v must be bool, callable(layer_idx, n_layers) -> bool, "
        f"or 'boundary', got {compress_v!r}"
    )


def patch_model(
    model,
    bits: int = 4,
    head_dim: int | None = None,
    verify: bool = True,
    compress_v: bool | str = True,
    max_iterations: int = 300,
    num_grid_points: int = 50000,
    strategy: str | None = None,
    target_compression: float | None = None,
    quality_target: float | None = None,
    tokenizer=None,
    calibration_text: str | None = None,
) -> CompressedKVCache:
    """Patch all full-attention layers in a model to use fused TurboQuant.

    Auto-detects head_dim from model config. Skips DeltaNet/linear-attention layers.
    Raises ValueError if the model is not compatible (non-power-of-2 head_dim, etc.).

    Args:
        model: A HuggingFace CausalLM model.
        bits: Quantization bit-width (2, 3, or 4). Ignored when strategy="adaptive".
        head_dim: Override head dimension. Auto-detected from config if None.
        verify: Run a single-token smoke test after patching to catch silent
            correctness bugs. Set to False to skip (e.g., for benchmarking).
        compress_v: Controls value cache compression. Accepts:
            - True: compress V in all layers (default, maximum memory savings)
            - False: no V compression (K-only)
            - "boundary": keep first 2 + last 2 layers at fp16 V, compress rest
            - callable(layer_idx, n_layers) -> bool: custom per-layer strategy
        max_iterations: Lloyd-Max codebook iterations.
        num_grid_points: Lloyd-Max grid density.
        strategy: "adaptive" to auto-assign per-layer bit-rates via calibration.
            When set, `bits` is used as the default/fallback.
        target_compression: (adaptive only) target average KV compression ratio.
        quality_target: (adaptive only) target mean cosine similarity (0-1).
        tokenizer: (adaptive only) tokenizer for calibration text.
        calibration_text: (adaptive only) custom calibration text.

    Returns a CompressedKVCache to pass as past_key_values to model.generate().
    """
    config = _resolve_config(model)

    if head_dim is None:
        head_dim = _resolve_head_dim(config)
        if head_dim == 0:
            raise ValueError(
                "Could not detect head_dim from model config. "
                "Pass head_dim explicitly: patch_model(model, bits=4, head_dim=128)"
            )

    if head_dim < 1 or (head_dim & (head_dim - 1)) != 0:
        raise ValueError(
            f"head_dim={head_dim} is not a power of 2. "
            f"TurboQuant requires power-of-2 head dimensions (64, 128, 256, ...) "
            f"because the Randomized Hadamard Transform uses butterfly operations. "
            f"This model is not compatible with patch_model(). "
            f"Use check_model_compatibility(model) for details."
        )

    if bits not in (2, 3, 4):
        raise ValueError(
            f"bits must be 2, 3, or 4, got {bits}. "
            f"Lloyd-Max codebooks are only precomputed for these bit-widths."
        )

    n_q_heads = getattr(config, "num_attention_heads", 0)
    n_kv_heads = getattr(config, "num_key_value_heads", n_q_heads)
    if n_kv_heads > 0 and n_q_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_q_heads ({n_q_heads}) is not divisible by n_kv_heads ({n_kv_heads}). "
            f"GQA grouping requires integer divisibility."
        )

    device = next(model.parameters()).device

    eligible_names = [
        name for name, module in model.named_modules() if _is_full_attention_layer(module, name)
    ]
    n_layers = len(eligible_names)

    bit_map: dict[int, int] | None = None
    if strategy == "adaptive":
        from fused_turboquant.core.adaptive import calibrate_layer_bits

        cal_kwargs: dict = {"head_dim": head_dim}
        if tokenizer is not None:
            cal_kwargs["tokenizer"] = tokenizer
        if calibration_text is not None:
            cal_kwargs["calibration_text"] = calibration_text
        if target_compression is not None:
            cal_kwargs["target_compression"] = target_compression
        elif quality_target is not None:
            cal_kwargs["quality_target"] = quality_target
        bit_map = calibrate_layer_bits(model, **cal_kwargs)

    default_bits = bits
    tq_cache: dict[int, TurboQuantMSE] = {}

    def get_tq(b: int) -> TurboQuantMSE:
        if b not in tq_cache:
            tq_cache[b] = TurboQuantMSE(
                head_dim=head_dim,
                bits=b,
                device=str(device),
                max_iterations=max_iterations,
                num_grid_points=num_grid_points,
            )
        return tq_cache[b]

    primary_tq = get_tq(default_bits)
    any_v = not isinstance(compress_v, bool) or compress_v
    cache = CompressedKVCache(primary_tq, compress_v=any_v)

    patched = 0
    layer_idx = 0
    originals = {}
    v_compressed_count = 0

    for name, module in model.named_modules():
        if _is_full_attention_layer(module, name):
            layer_bits = bit_map[layer_idx] if bit_map else default_bits
            layer_tq = get_tq(layer_bits)
            layer_compress_v = _resolve_compress_v(compress_v, layer_idx, n_layers)
            if layer_compress_v:
                v_compressed_count += 1
            originals[name] = module.forward
            module.forward = make_fused_attention_forward(
                module,
                cache,
                layer_tq,
                layer_idx,
                config=config,
                compress_v=layer_compress_v,
            )
            patched += 1
            layer_idx += 1

    model._fused_tq_originals = originals

    if patched == 0:
        logger.warning(
            "No attention layers were patched. This model may not use standard "
            "q_proj/k_proj/v_proj projections. Use check_model_compatibility(model) "
            "to diagnose."
        )
    else:
        arch_name = type(model).__name__
        if arch_name not in KNOWN_COMPATIBLE:
            logger.info(
                "Architecture %s has not been tested with fused-turboquant. "
                "Running compatibility checks...",
                arch_name,
            )

        if v_compressed_count == patched:
            kv_mode = "K+V"
        elif v_compressed_count == 0:
            kv_mode = "K-only"
        else:
            kv_mode = f"K+V({v_compressed_count}/{patched} layers)"
        logger.info(
            "Patched %d attention layers with fused TurboQuant (%d-bit, %s compression)",
            patched,
            bits,
            kv_mode,
        )

        if verify:
            _smoke_test(model, cache, originals, config, head_dim)
            cache.reset()

    return cache


def unpatch_model(model) -> None:
    """Restore original attention forward methods."""
    originals = getattr(model, "_fused_tq_originals", {})
    for name, module in model.named_modules():
        if name in originals:
            module.forward = originals[name]
    model._fused_tq_originals = {}
    logger.info("Unpatched all fused TurboQuant layers")


class FusedTurboQuantRunner:
    """High-level runner: patches model, generates text, unpatches.

    Usage:
        runner = FusedTurboQuantRunner(model, tokenizer, bits=4)
        text = runner.generate("What is 2+2?", max_new_tokens=100)
    """

    def __init__(self, model, tokenizer, bits: int = 4):
        self.model = model
        self.tokenizer = tokenizer
        self.bits = bits

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        do_sample: bool = False,
    ) -> str:
        cache = patch_model(self.model, bits=self.bits)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                past_key_values=cache,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_cache=True,
            )

        gen_ids = out[0][input_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        unpatch_model(self.model)
        return text
