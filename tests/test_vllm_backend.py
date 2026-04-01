"""
Integration tests for the FUSED_TURBOQUANT vLLM attention backend.

Tests are split into two groups:
1. Component tests (no vLLM dependency) — validate cache_ops, backend config, etc.
2. Full vLLM integration tests — require vLLM and a GPU, skipped otherwise.
"""

from __future__ import annotations

import pytest
import torch

from fused_turboquant.core.quantizer import TurboQuantMSE
from fused_turboquant.vllm_plugin.backend import FusedTurboQuantBackend
from fused_turboquant.vllm_plugin.cache_ops import (
    compressed_copy_blocks,
    compressed_swap_blocks,
    compute_compressed_elem_size,
    gather_compressed_kv_batched,
)

# ---------------------------------------------------------------------------
# Component tests (no vLLM required)
# ---------------------------------------------------------------------------


class TestCompressedElemSize:
    def test_4bit_128(self):
        assert compute_compressed_elem_size(128, 4) == 68

    def test_3bit_128(self):
        assert compute_compressed_elem_size(128, 3) == 52

    def test_2bit_128(self):
        assert compute_compressed_elem_size(128, 2) == 36

    def test_4bit_64(self):
        assert compute_compressed_elem_size(64, 4) == 36

    def test_invalid_bits(self):
        with pytest.raises(ValueError):
            compute_compressed_elem_size(128, 5)


class TestKVCacheShape:
    def test_shape_4bit(self):
        shape = FusedTurboQuantBackend.get_kv_cache_shape(
            num_blocks=100,
            block_size=16,
            num_kv_heads=8,
            head_size=128,
        )
        assert shape == (2, 100, 16, 8, 68)

    def test_shape_3bit(self):
        import os

        old = os.environ.get("TURBOQUANT_BITS")
        os.environ["TURBOQUANT_BITS"] = "3"
        try:
            from importlib import reload

            import fused_turboquant.vllm_plugin.backend as bmod

            reload(bmod)
            shape = bmod.FusedTurboQuantBackend.get_kv_cache_shape(
                num_blocks=50,
                block_size=16,
                num_kv_heads=4,
                head_size=128,
            )
            assert shape == (2, 50, 16, 4, 52)
        finally:
            if old is not None:
                os.environ["TURBOQUANT_BITS"] = old
            else:
                os.environ.pop("TURBOQUANT_BITS", None)
            reload(bmod)


class TestBackendMeta:
    def test_name(self):
        assert FusedTurboQuantBackend.get_name() == "FUSED_TURBOQUANT"

    def test_supported_head_sizes(self):
        sizes = FusedTurboQuantBackend.get_supported_head_sizes()
        assert 128 in sizes
        assert 64 in sizes
        assert 256 in sizes


class TestSwapBlocks:
    def test_swap_basic(self):
        src = torch.randint(0, 255, (10, 16, 4, 68), dtype=torch.uint8)
        dst = torch.zeros_like(src)
        mapping = torch.tensor([[0, 5], [3, 7]])
        compressed_swap_blocks(src, dst, mapping)
        assert torch.equal(dst[5], src[0])
        assert torch.equal(dst[7], src[3])

    def test_copy_blocks(self):
        cache1 = torch.randint(0, 255, (2, 10, 16, 4, 68), dtype=torch.uint8)
        cache2 = torch.randint(0, 255, (2, 10, 16, 4, 68), dtype=torch.uint8)
        mapping = torch.tensor([[2, 8]])
        compressed_copy_blocks([cache1, cache2], mapping)
        assert torch.equal(cache1[:, 8], cache1[:, 2])
        assert torch.equal(cache2[:, 8], cache2[:, 2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestGatherCompressed:
    def test_gather_roundtrip(self):
        """Write compressed data to cache, gather it back, verify match."""
        bits = 4
        head_dim = 128
        num_kv_heads = 4
        block_size = 16
        num_blocks = 10
        seq_len = 5
        packed_dim = head_dim // 2
        elem_size = compute_compressed_elem_size(head_dim, bits)

        tq = TurboQuantMSE(head_dim=head_dim, bits=bits, device="cuda")

        kv_cache = torch.zeros(
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            elem_size,
            dtype=torch.uint8,
            device="cuda",
        )

        fake_keys = torch.randn(
            seq_len,
            num_kv_heads,
            head_dim,
            device="cuda",
        )
        compressed = tq.encode(fake_keys.float())

        for pos in range(seq_len):
            block_idx = pos // block_size
            offset = pos % block_size
            kv_cache[0, block_idx, offset, :, :packed_dim] = compressed.indices[pos]
            norm_bytes = (
                compressed.norms[pos]
                .float()
                .contiguous()
                .view(torch.uint8)
                .reshape(num_kv_heads, 4)
            )
            kv_cache[0, block_idx, offset, :, packed_dim : packed_dim + 4] = norm_bytes

        block_tables = torch.tensor([[0]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")

        gathered_packed, gathered_norms = gather_compressed_kv_batched(
            kv_cache,
            block_tables,
            seq_lens,
            kv_type=0,
            packed_dim=packed_dim,
            max_seq_len=seq_len,
        )

        assert gathered_packed.shape == (1, num_kv_heads, seq_len, packed_dim)
        assert gathered_norms.shape == (1, num_kv_heads, seq_len)

        for pos in range(seq_len):
            assert torch.equal(
                gathered_packed[0, :, pos, :],
                compressed.indices[pos],
            )
            assert torch.allclose(
                gathered_norms[0, :, pos],
                compressed.norms[pos].float(),
                atol=1e-6,
            )


# ---------------------------------------------------------------------------
# Full vLLM integration tests (skipped if vLLM not installed)
# ---------------------------------------------------------------------------

try:
    from vllm import LLM, SamplingParams

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


@pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestVLLMIntegration:
    """End-to-end tests using vLLM's LLM class with FUSED_TURBOQUANT backend."""

    def test_offline_inference_small_model(self):
        """Run offline inference with a small model."""
        llm = LLM(
            model="Qwen/Qwen2.5-0.5B",
            attention_backend="FUSED_TURBOQUANT",
            gpu_memory_utilization=0.5,
            enforce_eager=True,
            trust_remote_code=True,
        )
        outputs = llm.generate(
            ["The capital of France is"],
            SamplingParams(max_tokens=20, temperature=0.0),
        )
        assert len(outputs) == 1
        text = outputs[0].outputs[0].text
        assert len(text) > 0
        assert "paris" in text.lower() or "Paris" in text

    def test_batch_inference(self):
        """Run batch inference with multiple prompts."""
        llm = LLM(
            model="Qwen/Qwen2.5-0.5B",
            attention_backend="FUSED_TURBOQUANT",
            gpu_memory_utilization=0.5,
            enforce_eager=True,
            trust_remote_code=True,
        )
        prompts = [
            "1 + 1 =",
            "The color of the sky is",
        ]
        outputs = llm.generate(
            prompts,
            SamplingParams(max_tokens=10, temperature=0.0),
        )
        assert len(outputs) == 2
        for out in outputs:
            assert len(out.outputs[0].text) > 0
