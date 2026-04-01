"""
Fused TurboQuant attention for Gemma 3.

Replaces standard Q@K^T with a Triton kernel that operates directly on
compressed uint8 key indices — never materializing fp16 keys.

Usage:
    from turboquant_fused import FusedTurboQuantRunner
    runner = FusedTurboQuantRunner(model, processor, bits=4)
    text = runner.generate("What is 2+2?", max_new_tokens=30)
"""

import math

import torch
from transformers import DynamicCache
from triton_attention import fused_qk_scores
from turboquant_core import TurboQuantMSE


class CompressedKVCache(DynamicCache):
    """KV cache that stores compressed keys (uint8 indices + norms).

    Keys are quantized on insertion. During attention, the fused Triton
    kernel reads compressed keys directly — no fp16 dequantization needed.

    Values are stored in fp16 (standard) since the softmax@V matmul
    benefits less from compression.
    """

    def __init__(self, quantizer: TurboQuantMSE):
        super().__init__()
        self.tq = quantizer
        # Per-layer compressed key storage
        self._compressed_keys: list[dict | None] = []

    def store_compressed_key(self, key_states: torch.Tensor, layer_idx: int):
        """Quantize and store key states. Called from patched attention."""
        while len(self._compressed_keys) <= layer_idx:
            self._compressed_keys.append(None)

        q = self.tq.quantize(key_states.float())

        if self._compressed_keys[layer_idx] is None:
            self._compressed_keys[layer_idx] = q
        else:
            prev = self._compressed_keys[layer_idx]
            self._compressed_keys[layer_idx] = {
                "idx": torch.cat([prev["idx"], q["idx"]], dim=2),
                "norms": torch.cat([prev["norms"], q["norms"]], dim=2),
            }

    def get_compressed_key(self, layer_idx: int) -> dict | None:
        if layer_idx < len(self._compressed_keys):
            return self._compressed_keys[layer_idx]
        return None

    def get_kv_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self._compressed_keys) and self._compressed_keys[layer_idx] is not None:
            return self._compressed_keys[layer_idx]["idx"].shape[2]
        return 0


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply RoPE — copied from transformers to avoid import issues."""
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
    attn_module, cache: CompressedKVCache, quantizer: TurboQuantMSE, layer_index: int
):
    """Create a replacement forward for a Gemma3 attention layer."""

    # Cache the rotation matrix for pre-rotating queries
    Q_T = quantizer.Q_T  # [head_dim, head_dim]
    centroids = quantizer.centroids
    head_dim = quantizer.d
    scale = 1.0 / math.sqrt(head_dim)
    n_heads = attn_module.num_heads
    # num_key_value_heads lives in config, not on the module
    cfg = attn_module.config
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    n_kv_groups = n_heads // n_kv_heads
    layer_idx = layer_index  # passed in from enumeration

    # Check if this layer uses sliding window attention
    is_sliding = getattr(attn_module, "is_sliding", False)
    sliding_window = getattr(attn_module, "sliding_window", None)
    if is_sliding and sliding_window is None:
        # Try to get from config
        config = getattr(attn_module, "config", None)
        if config:
            sliding_window = getattr(config, "sliding_window", None)

    def fused_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # Q/K/V projections
        query_states = attn_module.q_proj(hidden_states)
        key_states = attn_module.k_proj(hidden_states)
        value_states = attn_module.v_proj(hidden_states)

        # Reshape to [batch, n_heads, q_len, head_dim]
        query_states = query_states.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Store compressed keys
        cache.store_compressed_key(key_states, layer_idx)

        # Store values normally in the parent DynamicCache
        # We need to call the parent's update for values only
        # Hack: store values directly
        while len(cache.value_cache) <= layer_idx:
            cache.key_cache.append(torch.empty(0))  # placeholder
            cache.value_cache.append(torch.empty(0))
        if cache.value_cache[layer_idx].numel() == 0:
            cache.value_cache[layer_idx] = value_states
        else:
            cache.value_cache[layer_idx] = torch.cat(
                [cache.value_cache[layer_idx], value_states], dim=2
            )
        # Keep key_cache in sync for length tracking
        cache.key_cache[layer_idx] = cache.value_cache[layer_idx]  # dummy, same shape

        # Get full accumulated values
        full_values = cache.value_cache[layer_idx]
        kv_len = full_values.shape[2]

        # Get compressed keys
        compressed = cache.get_compressed_key(layer_idx)

        # --- Fused attention scores ---
        # Pre-rotate queries: q_rot = q @ Q_T
        q_rot = query_states.float() @ Q_T.unsqueeze(0).unsqueeze(0)

        # Use Triton kernel
        attn_weights = fused_qk_scores(
            q_rot, compressed["idx"], compressed["norms"], centroids, scale
        )

        # Apply attention mask (causal + sliding window if applicable)
        if attention_mask is not None:
            # attention_mask shape depends on transformers version
            # Typically [batch, 1, q_len, kv_len] or similar
            causal_mask = attention_mask
            if causal_mask.dim() == 4:
                attn_weights = attn_weights + causal_mask[:, :, :q_len, :kv_len]
            elif causal_mask.dim() == 2:
                attn_weights = attn_weights + causal_mask[:q_len, :kv_len]

        # Softmax
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(query_states.dtype)

        # Expand values for GQA and compute output
        full_values_expanded = _repeat_kv(full_values, n_kv_groups)
        attn_output = torch.matmul(attn_weights, full_values_expanded)

        # Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = attn_module.out_proj(attn_output)

        return attn_output, None  # (output, attn_weights)

    return fused_forward


def install_fused_attention(model, bits: int = 4) -> CompressedKVCache:
    """Patch all attention layers in a Gemma3 model to use fused TurboQuant.

    Returns a CompressedKVCache to pass as past_key_values to generate().
    """
    # Detect head_dim
    config = model.config
    if hasattr(config, "text_config"):
        text_config = config.text_config
    else:
        text_config = config
    head_dim = getattr(text_config, "head_dim", 256)

    # Create quantizer
    tq = TurboQuantMSE(d=head_dim, bits=bits, device="cuda")

    # Create compressed cache
    cache = CompressedKVCache(tq)

    # Find and patch text attention layers (not vision encoder)
    patched = 0
    layer_idx = 0
    for name, module in model.named_modules():
        if all(
            hasattr(module, attr)
            for attr in ["q_proj", "k_proj", "v_proj", "out_proj", "num_heads"]
        ):
            module.forward = make_fused_attention_forward(module, cache, tq, layer_idx)
            patched += 1
            layer_idx += 1

    print(f"  Installed fused TurboQuant ({bits}-bit) on {patched} attention layers")
    return cache


class FusedTurboQuantRunner:
    """High-level runner: patches model, generates, unpatches.

    Usage:
        runner = FusedTurboQuantRunner(model, processor, bits=4)
        text = runner.generate("What is 2+2?", max_new_tokens=30)
    """

    def __init__(self, model, processor, bits: int = 4):
        self.model = model
        self.processor = processor
        self.bits = bits
        # Save original forwards for unpatching
        self._originals: dict[str, callable] = {}
        for name, module in model.named_modules():
            if all(
                hasattr(module, attr)
                for attr in ["q_proj", "k_proj", "v_proj", "out_proj", "num_heads"]
            ):
                self._originals[name] = module.forward

    def generate(
        self, prompt: str, max_new_tokens: int = 200, system: str = "You are a helpful assistant."
    ):
        # Install fused attention (creates fresh cache)
        cache = install_fused_attention(self.model, self.bits)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                past_key_values=cache,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        gen_ids = out[0][input_len:]
        text = self.processor.decode(gen_ids, skip_special_tokens=True)

        # Unpatch
        self._unpatch()

        return text

    def _unpatch(self):
        for name, module in self.model.named_modules():
            if name in self._originals:
                module.forward = self._originals[name]
