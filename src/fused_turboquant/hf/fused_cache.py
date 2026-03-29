"""
Fused TurboQuant cache with compressed key storage and fused attention.

Stores keys in compressed form (uint8 indices + fp32 norms) and computes
Q @ K^T directly from compressed keys using our Triton fused attention kernel.
Values are stored in fp16 (standard).

This is a real integration that changes the attention computation path:
- Keys are compressed via fused Triton encode kernel
- Queries are pre-rotated via RHT (not dense QR matmul)
- Q @ K^T is computed from compressed indices via fused Triton kernel
- Only the softmax @ V matmul uses standard fp16

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

from fused_turboquant.core.quantizer import TurboQuantMSE, CompressedTensor
from fused_turboquant.core.packing import unpack_nibbles, unpack_2bit

logger = logging.getLogger(__name__)


class CompressedKVCache(DynamicCache):
    """KV cache that stores compressed keys (uint8 indices + norms).

    Keys are quantized on insertion. During attention, the fused Triton
    kernel reads compressed keys directly — no fp16 key dequantization needed.

    Values are stored in fp16 (standard) since softmax @ V benefits less
    from compression.
    """

    def __init__(self, quantizer: TurboQuantMSE):
        super().__init__()
        self.tq = quantizer
        self._compressed_keys: list[Optional[dict]] = []

    def store_compressed_key(self, key_states: torch.Tensor, layer_idx: int):
        """Compress and store key states for a layer."""
        while len(self._compressed_keys) <= layer_idx:
            self._compressed_keys.append(None)

        compressed = self.tq.encode(key_states.float())

        if compressed.bits == 4:
            indices = unpack_nibbles(compressed.indices, compressed.original_dim)
        elif compressed.bits == 2:
            indices = unpack_2bit(compressed.indices, compressed.original_dim)
        else:
            indices = compressed.indices

        indices = indices.view(key_states.shape).to(torch.uint8)
        norms = compressed.norms.view(*key_states.shape[:-1])

        entry = {"indices": indices, "norms": norms}

        if self._compressed_keys[layer_idx] is None:
            self._compressed_keys[layer_idx] = entry
        else:
            prev = self._compressed_keys[layer_idx]
            self._compressed_keys[layer_idx] = {
                "indices": torch.cat([prev["indices"], entry["indices"]], dim=2),
                "norms": torch.cat([prev["norms"], entry["norms"]], dim=2),
            }

    def get_compressed_key(self, layer_idx: int) -> Optional[dict]:
        if layer_idx < len(self._compressed_keys):
            return self._compressed_keys[layer_idx]
        return None

    def get_kv_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self._compressed_keys) and self._compressed_keys[layer_idx] is not None:
            return self._compressed_keys[layer_idx]["indices"].shape[2]
        return 0


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply RoPE."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA."""
    if n_rep == 1:
        return hidden_states
    batch, n_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, n_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, n_kv_heads * n_rep, slen, head_dim)


def make_fused_attention_forward(
    attn_module,
    cache: CompressedKVCache,
    quantizer: TurboQuantMSE,
    layer_idx: int,
):
    """Create a replacement forward for an attention layer that uses fused TurboQuant.

    The key innovation vs Dejan.ai: we pre-rotate queries with RHT (O(d log d))
    instead of dense QR matmul (O(d^2)), and use our fused Triton attention kernel.
    """
    from fused_turboquant.kernels.triton_attention import fused_qk_scores_rht

    rht_signs = quantizer.rotation.signs
    centroids = quantizer.quantizer.levels
    head_dim = quantizer.head_dim
    scale = 1.0 / math.sqrt(head_dim)

    n_heads = getattr(attn_module, "num_heads", None)
    config = getattr(attn_module, "config", None)
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads) if config else n_heads
    if n_kv_heads is None:
        n_kv_heads = n_heads
    n_kv_groups = n_heads // n_kv_heads if n_heads and n_kv_heads else 1

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

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = _apply_rotary_pos_emb(
                query_states, key_states, cos, sin,
            )

        cache.store_compressed_key(key_states, layer_idx)

        while len(cache.value_cache) <= layer_idx:
            cache.key_cache.append(torch.empty(0, device=hidden_states.device))
            cache.value_cache.append(torch.empty(0, device=hidden_states.device))

        if cache.value_cache[layer_idx].numel() == 0:
            cache.value_cache[layer_idx] = value_states
        else:
            cache.value_cache[layer_idx] = torch.cat(
                [cache.value_cache[layer_idx], value_states], dim=2
            )
        cache.key_cache[layer_idx] = cache.value_cache[layer_idx]

        full_values = cache.value_cache[layer_idx]
        compressed = cache.get_compressed_key(layer_idx)

        # Pre-rotate queries via RHT: q_rot = fwht(q * signs) (O(d log d))
        from fused_turboquant.core.hadamard import randomized_hadamard
        q_flat = query_states.float().reshape(-1, head_dim)
        q_rot = randomized_hadamard(q_flat, rht_signs)
        q_rot = q_rot.view_as(query_states)

        attn_weights = fused_qk_scores_rht(
            q_rot,
            compressed["indices"],
            compressed["norms"],
            centroids,
            scale,
        )

        if attention_mask is not None:
            kv_len = compressed["indices"].shape[2]
            if attention_mask.dim() == 4:
                attn_weights = attn_weights + attention_mask[:, :, :q_len, :kv_len]
            elif attention_mask.dim() == 2:
                attn_weights = attn_weights + attention_mask[:q_len, :kv_len]

        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32,
        ).to(query_states.dtype)

        full_values_expanded = _repeat_kv(full_values, n_kv_groups)
        attn_output = torch.matmul(attn_weights, full_values_expanded)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        o_proj = getattr(attn_module, "o_proj", None) or getattr(attn_module, "out_proj", None)
        if o_proj is not None:
            attn_output = o_proj(attn_output)

        return attn_output, None

    return fused_forward


def _is_full_attention_layer(module) -> bool:
    """Detect if a module is a full-attention layer (not DeltaNet/linear attention)."""
    required = ["q_proj", "k_proj", "v_proj"]
    output = ["o_proj", "out_proj"]
    has_qkv = all(hasattr(module, attr) for attr in required)
    has_output = any(hasattr(module, attr) for attr in output)
    has_heads = hasattr(module, "num_heads")
    return has_qkv and has_output and has_heads


def patch_model(
    model,
    bits: int = 4,
    head_dim: int | None = None,
) -> CompressedKVCache:
    """Patch all full-attention layers in a model to use fused TurboQuant.

    Auto-detects head_dim from model config. Skips DeltaNet/linear-attention layers.

    Returns a CompressedKVCache to pass as past_key_values to model.generate().
    """
    config = model.config
    if hasattr(config, "text_config"):
        config = config.text_config

    if head_dim is None:
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(config, "hidden_size", 4096)
            num_heads = getattr(config, "num_attention_heads", 32)
            head_dim = hidden_size // num_heads

    device = next(model.parameters()).device
    tq = TurboQuantMSE(head_dim=head_dim, bits=bits, device=str(device))
    cache = CompressedKVCache(tq)

    patched = 0
    layer_idx = 0
    originals = {}

    for name, module in model.named_modules():
        if _is_full_attention_layer(module):
            originals[name] = module.forward
            module.forward = make_fused_attention_forward(
                module, cache, tq, layer_idx,
            )
            patched += 1
            layer_idx += 1

    model._fused_tq_originals = originals
    logger.info(f"Patched {patched} attention layers with fused TurboQuant ({bits}-bit)")
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
