"""
Chunked prefill with incremental KV compression.

For very long prompts, the full FP16 KV cache during prefill is the memory
bottleneck. This module implements chunked prefill: split the prompt into
chunks, process each chunk with SDPA, compress the chunk's KV immediately,
then discard the FP16 KV before processing the next chunk.

Peak prefill memory = model_weights + chunk_size * KV_fp16 + total_compressed_KV,
instead of model_weights + total_prompt * KV_fp16.

Usage:
    from fused_turboquant.hf.chunked_prefill import chunked_generate

    output = chunked_generate(
        model, tokenizer, prompt,
        bits=4, chunk_size=2048, max_new_tokens=100,
    )
"""

from __future__ import annotations

import logging

import torch

from fused_turboquant.hf.fused_cache import (
    CompressedKVCache,
    patch_model,
)

logger = logging.getLogger(__name__)


def chunked_prefill(
    model,
    input_ids: torch.Tensor,
    cache: CompressedKVCache,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Process a long prompt in chunks, compressing KV incrementally.

    Instead of running the full prompt through the model (which requires
    O(prompt_len * head_dim * n_layers) FP16 KV cache memory), this
    processes the prompt in fixed-size chunks. After each chunk, KV is
    compressed and the FP16 intermediates are freed.

    Args:
        model: HuggingFace CausalLM model (already patched with patch_model).
        input_ids: [1, seq_len] input token IDs.
        cache: CompressedKVCache from patch_model().
        chunk_size: tokens per chunk. Smaller = less peak memory, more overhead.

    Returns:
        Last hidden states logits from the final chunk.
    """
    seq_len = input_ids.shape[1]
    device = input_ids.device

    if seq_len <= chunk_size:
        with torch.inference_mode():
            outputs = model(
                input_ids,
                past_key_values=cache,
                use_cache=True,
            )
        return outputs.logits

    n_chunks = (seq_len + chunk_size - 1) // chunk_size
    logger.info(
        "Chunked prefill: %d tokens in %d chunks of %d",
        seq_len,
        n_chunks,
        chunk_size,
    )

    logits = None
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, seq_len)
        chunk_ids = input_ids[:, start:end]

        with torch.inference_mode():
            outputs = model(
                chunk_ids,
                past_key_values=cache,
                use_cache=True,
            )

        logits = outputs.logits

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return logits


def chunked_generate(
    model,
    tokenizer,
    prompt: str,
    bits: int = 4,
    chunk_size: int = 2048,
    max_new_tokens: int = 200,
    do_sample: bool = False,
    compress_v: bool | str = True,
    verify: bool = False,
) -> str:
    """Generate text with chunked prefill for long prompts.

    Patches the model, runs chunked prefill to compress the prompt's KV
    incrementally, then generates tokens autoregressively.

    Args:
        model: HuggingFace CausalLM model.
        tokenizer: tokenizer.
        prompt: input text.
        bits: quantization bit-width.
        chunk_size: tokens per prefill chunk.
        max_new_tokens: max tokens to generate.
        do_sample: whether to sample.
        compress_v: V compression strategy.
        verify: run smoke test.

    Returns:
        Generated text (excluding prompt).
    """
    cache = patch_model(model, bits=bits, compress_v=compress_v, verify=verify)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    logits = chunked_prefill(model, input_ids, cache, chunk_size=chunk_size)

    generated_ids = []
    next_token = (
        logits[:, -1:].argmax(dim=-1)
        if not do_sample
        else torch.multinomial(
            torch.nn.functional.softmax(logits[:, -1], dim=-1),
            1,
        )
    )
    generated_ids.append(next_token)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    with torch.inference_mode():
        for _ in range(max_new_tokens - 1):
            outputs = model(
                next_token,
                past_key_values=cache,
                use_cache=True,
            )
            next_logits = outputs.logits[:, -1]

            if do_sample:
                next_token = torch.multinomial(
                    torch.nn.functional.softmax(next_logits, dim=-1),
                    1,
                )
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated_ids.append(next_token)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    gen_ids = torch.cat(generated_ids, dim=1)[0]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    from fused_turboquant.hf.fused_cache import unpatch_model

    unpatch_model(model)

    return text
