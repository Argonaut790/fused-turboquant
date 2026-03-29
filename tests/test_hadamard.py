"""Tests for the Randomized Hadamard Transform."""

import math

import pytest
import torch

from fused_turboquant.core.hadamard import (
    DenseQRRotation,
    RHTRotation,
    fwht,
    generate_rht_signs,
    inverse_fwht,
    randomized_hadamard,
    inverse_randomized_hadamard,
)


class TestFWHT:
    def test_roundtrip_identity(self, device):
        """H(H(x)) should return x (FWHT is its own inverse up to normalization)."""
        x = torch.randn(32, 128, device=device)
        y = fwht(x)
        x_rec = inverse_fwht(y)
        torch.testing.assert_close(x_rec, x, atol=1e-5, rtol=1e-5)

    def test_orthogonality(self, device):
        """FWHT should preserve vector norms (orthogonal transform)."""
        x = torch.randn(64, 256, device=device)
        y = fwht(x)
        norms_x = torch.norm(x, dim=-1)
        norms_y = torch.norm(y, dim=-1)
        torch.testing.assert_close(norms_x, norms_y, atol=1e-4, rtol=1e-4)

    def test_known_values_d2(self, device):
        """FWHT of [1, 0] should be [1/√2, 1/√2]."""
        x = torch.tensor([[1.0, 0.0]], device=device)
        y = fwht(x)
        expected = torch.tensor([[1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]], device=device)
        torch.testing.assert_close(y, expected, atol=1e-6, rtol=1e-6)

    def test_rejects_non_power_of_two(self, device):
        x = torch.randn(10, 100, device=device)
        with pytest.raises(ValueError, match="power of 2"):
            fwht(x)

    def test_batched_shapes(self, device):
        """FWHT should handle arbitrary leading dimensions."""
        for shape in [(2, 3, 128), (4, 8, 2, 64), (256,)]:
            x = torch.randn(*shape, device=device)
            y = fwht(x)
            assert y.shape == x.shape


class TestRHT:
    def test_roundtrip(self, device, head_dim):
        """RHT → inverse RHT should recover the original vector."""
        signs = generate_rht_signs(head_dim, seed=0, device=device)
        x = torch.randn(16, head_dim, device=device)
        y = randomized_hadamard(x, signs)
        x_rec = inverse_randomized_hadamard(y, signs)
        torch.testing.assert_close(x_rec, x, atol=1e-4, rtol=1e-4)

    def test_norm_preservation(self, device, head_dim):
        """RHT should preserve norms (it's an orthogonal rotation)."""
        signs = generate_rht_signs(head_dim, seed=0, device=device)
        x = torch.randn(64, head_dim, device=device)
        y = randomized_hadamard(x, signs)
        torch.testing.assert_close(
            torch.norm(x, dim=-1), torch.norm(y, dim=-1), atol=1e-4, rtol=1e-4
        )

    def test_coordinate_distribution(self, device):
        """
        After RHT, coordinates of unit vectors should be concentrated
        around 0 with std ≈ 1/√d (Beta distribution concentration).
        """
        d = 256
        signs = generate_rht_signs(d, seed=0, device=device)
        x = torch.randn(10000, d, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = randomized_hadamard(x, signs)

        expected_std = 1.0 / math.sqrt(d)
        actual_std = y.std().item()
        assert abs(actual_std - expected_std) < 0.01, (
            f"Expected std ≈ {expected_std:.4f}, got {actual_std:.4f}"
        )

    def test_deterministic_with_seed(self, device):
        """Same seed should produce the same signs."""
        s1 = generate_rht_signs(128, seed=42, device=device)
        s2 = generate_rht_signs(128, seed=42, device=device)
        torch.testing.assert_close(s1, s2)


class TestRHTModule:
    def test_module_roundtrip(self, device, head_dim):
        rot = RHTRotation(head_dim, seed=0, device=device)
        x = torch.randn(8, 4, head_dim, device=device)
        y = rot(x)
        x_rec = rot.inverse(y)
        torch.testing.assert_close(x_rec, x, atol=1e-4, rtol=1e-4)

    def test_rht_matches_dense_qr_quality(self, device):
        """
        RHT and Dense QR should produce the same distributional properties
        (both are random orthogonal rotations on the sphere).
        """
        d = 256
        n = 5000
        x = torch.randn(n, d, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        rht = RHTRotation(d, seed=0, device=device)
        dense = DenseQRRotation(d, seed=0, device=device)

        y_rht = rht(x)
        y_dense = dense(x)

        std_rht = y_rht.std().item()
        std_dense = y_dense.std().item()

        assert abs(std_rht - std_dense) < 0.005, (
            f"RHT std={std_rht:.4f} vs Dense std={std_dense:.4f}"
        )

    def test_memory_comparison(self):
        """RHT should use O(d) memory vs Dense QR's O(d²)."""
        d = 256
        rht = RHTRotation(d)
        dense = DenseQRRotation(d)

        rht_bytes = sum(p.numel() * p.element_size() for p in rht.buffers())
        dense_bytes = sum(p.numel() * p.element_size() for p in dense.buffers())

        assert rht_bytes == d * 4  # d float32 signs
        assert dense_bytes == d * d * 4  # d×d float32 matrix
        assert dense_bytes / rht_bytes == d  # 256x difference for d=256
