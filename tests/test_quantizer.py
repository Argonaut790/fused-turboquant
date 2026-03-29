"""Tests for the TurboQuant_MSE quantizer pipeline."""

import pytest
import torch

from fused_turboquant.core.lloyd_max import LloydMaxQuantizer
from fused_turboquant.core.packing import pack_2bit, pack_nibbles, unpack_2bit, unpack_nibbles
from fused_turboquant.core.quantizer import TurboQuantMSE


class TestLloydMax:
    def test_levels_are_sorted(self, device):
        q = LloydMaxQuantizer(dim=256, bits=4, device=device)
        levels = q.levels.cpu()
        diffs = levels[1:] - levels[:-1]
        assert (diffs > 0).all(), "Levels must be strictly increasing"

    def test_symmetric_levels(self, device):
        q = LloydMaxQuantizer(dim=256, bits=4, device=device)
        levels = q.levels.cpu()
        n = len(levels)
        for i in range(n // 2):
            assert abs(levels[i] + levels[n - 1 - i]) < 1e-3, (
                f"Level {i}={levels[i]:.6f} != -Level {n-1-i}={levels[n-1-i]:.6f}"
            )

    def test_quantize_dequantize(self, device):
        q = LloydMaxQuantizer(dim=256, bits=4, device=device)
        x = torch.randn(100, device=device) * 0.1
        indices = q.quantize(x)
        x_hat = q.dequantize(indices)
        assert indices.dtype == torch.uint8
        assert (indices < 16).all()
        error = (x - x_hat).abs().mean()
        assert error < 0.05, f"Mean abs error {error:.4f} too large"


class TestPacking:
    def test_nibble_roundtrip(self, device):
        original = torch.randint(0, 16, (32, 256), dtype=torch.uint8, device=device)
        packed = pack_nibbles(original)
        assert packed.shape == (32, 128)
        unpacked = unpack_nibbles(packed, 256)
        torch.testing.assert_close(unpacked, original)

    def test_2bit_roundtrip(self, device):
        original = torch.randint(0, 4, (32, 256), dtype=torch.uint8, device=device)
        packed = pack_2bit(original)
        assert packed.shape == (32, 64)
        unpacked = unpack_2bit(packed, 256)
        torch.testing.assert_close(unpacked, original)


class TestTurboQuantMSE:
    def test_roundtrip_quality_4bit(self, device):
        """4-bit TurboQuant should achieve >0.99 cosine similarity."""
        tq = TurboQuantMSE(head_dim=256, bits=4, device=device)
        x = torch.randn(64, 256, device=device)

        compressed = tq.encode(x)
        x_hat = tq.decode(compressed)

        x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
        xhat_norm = x_hat / (torch.norm(x_hat, dim=-1, keepdim=True) + 1e-8)
        cosine = torch.mean(torch.sum(x_norm * xhat_norm, dim=-1)).item()

        assert cosine > 0.99, f"4-bit cosine similarity {cosine:.4f} < 0.99"

    def test_roundtrip_quality_2bit(self, device):
        """2-bit should still achieve reasonable quality (>0.90)."""
        tq = TurboQuantMSE(head_dim=256, bits=2, device=device)
        x = torch.randn(64, 256, device=device)
        x_hat = tq.roundtrip(x)

        x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
        xhat_norm = x_hat / (torch.norm(x_hat, dim=-1, keepdim=True) + 1e-8)
        cosine = torch.mean(torch.sum(x_norm * xhat_norm, dim=-1)).item()

        assert cosine > 0.90, f"2-bit cosine similarity {cosine:.4f} < 0.90"

    def test_compression_ratio_4bit(self, device):
        tq = TurboQuantMSE(head_dim=256, bits=4, device=device)
        x = torch.randn(128, 256, device=device)
        compressed = tq.encode(x)
        ratio = compressed.compression_ratio
        assert ratio > 3.0, f"4-bit compression ratio {ratio:.1f}x < 3.0x"

    def test_batched_shapes(self, device):
        """Should handle multi-dimensional inputs (batch, heads, seq, dim)."""
        tq = TurboQuantMSE(head_dim=128, bits=4, device=device)
        x = torch.randn(2, 4, 16, 128, device=device)
        compressed = tq.encode(x)
        x_hat = tq.decode(compressed)
        assert x_hat.shape == x.shape

    def test_different_head_dims(self, device, head_dim):
        tq = TurboQuantMSE(head_dim=head_dim, bits=4, device=device)
        x = torch.randn(32, head_dim, device=device)
        x_hat = tq.roundtrip(x)
        assert x_hat.shape == x.shape

    def test_rejects_invalid_dim(self, device):
        with pytest.raises(ValueError, match="power of 2"):
            TurboQuantMSE(head_dim=100, bits=4, device=device)

    def test_rejects_invalid_bits(self, device):
        with pytest.raises(ValueError, match="bits must be"):
            TurboQuantMSE(head_dim=128, bits=5, device=device)
