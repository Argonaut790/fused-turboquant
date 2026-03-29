"""
Nibble packing utilities for sub-byte quantization indices.

4-bit: pack 2 indices per uint8 byte → 2x storage reduction over raw uint8.
3-bit: pack 8 indices per 3 bytes → 2.67x storage reduction.
2-bit: pack 4 indices per uint8 byte → 4x storage reduction.
"""

from __future__ import annotations

import torch


def pack_nibbles(indices: torch.Tensor) -> torch.Tensor:
    """
    Pack 4-bit indices (0-15) into uint8 nibble pairs.

    Two consecutive indices along the last dim are packed into one byte:
    byte = (high_nibble << 4) | low_nibble

    Args:
        indices: uint8 tensor of shape (..., d) where d is even.

    Returns:
        Packed uint8 tensor of shape (..., d // 2).
    """
    assert indices.shape[-1] % 2 == 0, "Last dimension must be even for nibble packing"
    flat = indices.view(*indices.shape[:-1], -1, 2)
    low = flat[..., 0].to(torch.uint8)
    high = flat[..., 1].to(torch.uint8)
    return (high << 4) | low


def unpack_nibbles(packed: torch.Tensor, original_dim: int) -> torch.Tensor:
    """
    Unpack uint8 nibble pairs back to 4-bit indices.

    Args:
        packed: uint8 tensor of shape (..., d // 2).
        original_dim: the original last dimension d.

    Returns:
        uint8 tensor of shape (..., d) with values in [0, 15].
    """
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).view(*packed.shape[:-1], original_dim)


def pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    """
    Pack 2-bit indices (0-3) into uint8 (4 indices per byte).

    Args:
        indices: uint8 tensor of shape (..., d) where d is divisible by 4.

    Returns:
        Packed uint8 tensor of shape (..., d // 4).
    """
    d = indices.shape[-1]
    assert d % 4 == 0, "Last dimension must be divisible by 4 for 2-bit packing"
    flat = indices.view(*indices.shape[:-1], -1, 4).to(torch.uint8)
    packed = flat[..., 0] | (flat[..., 1] << 2) | (flat[..., 2] << 4) | (flat[..., 3] << 6)
    return packed


def unpack_2bit(packed: torch.Tensor, original_dim: int) -> torch.Tensor:
    """
    Unpack uint8 back to 2-bit indices (4 indices per byte).

    Args:
        packed: uint8 tensor of shape (..., d // 4).
        original_dim: the original last dimension d.

    Returns:
        uint8 tensor of shape (..., d) with values in [0, 3].
    """
    b0 = packed & 0x03
    b1 = (packed >> 2) & 0x03
    b2 = (packed >> 4) & 0x03
    b3 = (packed >> 6) & 0x03
    return torch.stack([b0, b1, b2, b3], dim=-1).view(*packed.shape[:-1], original_dim)
