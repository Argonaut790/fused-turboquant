"""
DeltaKV-inspired inter-token residual compression.

Instead of independently quantizing each token's KV, stores deltas from a
running mean reference vector. Adjacent tokens often have similar KV vectors,
so the delta has smaller magnitude and quantizes with fewer bits for the same
quality.

This is orthogonal to the RHT + Lloyd-Max pipeline and stacks with it:
    original_vector -> compute delta from reference -> RHT -> quantize delta

Reference: DeltaKV (Feb 2026) — inter-token residual compression for KV caches.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from fused_turboquant.core.quantizer import CompressedTensor, TurboQuantMSE

logger = logging.getLogger(__name__)


class DeltaKVCompressor:
    """Inter-token delta compression for KV cache.

    Maintains a per-layer running reference vector (exponential moving average).
    New tokens store the delta (token - reference) rather than the full vector,
    achieving better quantization quality at the same bit-rate because deltas
    have smaller magnitude.

    The reference is updated with each new token to track the distribution drift.
    """

    def __init__(
        self,
        head_dim: int,
        bits: int = 4,
        momentum: float = 0.95,
        device: torch.device | str = "cpu",
    ):
        self.head_dim = head_dim
        self.bits = bits
        self.momentum = momentum
        self.device = device

        self.tq = TurboQuantMSE(head_dim=head_dim, bits=bits, device=str(device))

        self._reference: Optional[torch.Tensor] = None
        self._ref_count = 0

    def encode(self, x: torch.Tensor) -> tuple[CompressedTensor, torch.Tensor]:
        """Encode a token's KV vector as delta from reference.

        Args:
            x: tensor of shape (..., head_dim).

        Returns:
            (compressed_delta, reference) where reference is needed for decode.
        """
        if self._reference is None:
            self._reference = torch.zeros(
                *x.shape[:-1],
                self.head_dim,
                device=x.device,
                dtype=torch.float32,
            )

        ref = self._reference
        if ref.shape != x.shape:
            ref = ref.expand_as(x)

        delta = x.float() - ref
        compressed = self.tq.encode(delta)

        with torch.no_grad():
            self._reference = self.momentum * ref + (1 - self.momentum) * x.float().detach()
            self._ref_count += 1

        return compressed, ref

    def decode(
        self,
        compressed_delta: CompressedTensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Decode delta and add back the reference.

        Args:
            compressed_delta: compressed delta from encode().
            reference: reference vector from encode().

        Returns:
            Reconstructed original vector.
        """
        decoded_delta = self.tq.decode(compressed_delta)
        return decoded_delta + reference

    def reset(self) -> None:
        """Reset the running reference for a new sequence."""
        self._reference = None
        self._ref_count = 0

    def to(self, device: torch.device | str) -> "DeltaKVCompressor":
        self.device = device
        self.tq = self.tq.to(device)
        if self._reference is not None:
            self._reference = self._reference.to(device)
        return self


class DeltaKVCache:
    """KV cache with inter-token delta compression.

    Stores per-layer compressed deltas and references. Compatible with
    the same attention kernels since decode produces the same fp16 tensors.
    """

    def __init__(
        self,
        head_dim: int,
        n_layers: int,
        bits: int = 4,
        momentum: float = 0.95,
        device: torch.device | str = "cpu",
    ):
        self.compressors = {
            ("k", i): DeltaKVCompressor(head_dim, bits, momentum, device) for i in range(n_layers)
        }
        self.compressors.update(
            {("v", i): DeltaKVCompressor(head_dim, bits, momentum, device) for i in range(n_layers)}
        )
        self._stored: dict[tuple[str, int], list[tuple[CompressedTensor, torch.Tensor]]] = {}

    def store(
        self,
        kv_type: str,
        layer_idx: int,
        states: torch.Tensor,
    ) -> None:
        """Compress and store KV states for a layer.

        Args:
            kv_type: "k" or "v".
            layer_idx: attention layer index.
            states: [batch, n_heads, seq_len, head_dim] float tensor.
        """
        comp = self.compressors[(kv_type, layer_idx)]
        key = (kv_type, layer_idx)
        if key not in self._stored:
            self._stored[key] = []

        seq_len = states.shape[2]
        for t in range(seq_len):
            token_states = states[:, :, t : t + 1, :]
            compressed, ref = comp.encode(token_states)
            self._stored[key].append((compressed, ref.clone()))

    def retrieve(
        self,
        kv_type: str,
        layer_idx: int,
    ) -> torch.Tensor:
        """Decompress and return all stored KV states for a layer.

        Returns:
            [batch, n_heads, total_seq_len, head_dim] float tensor.
        """
        key = (kv_type, layer_idx)
        comp = self.compressors[key]

        decoded = []
        for compressed, ref in self._stored.get(key, []):
            decoded.append(comp.decode(compressed, ref))

        if not decoded:
            raise ValueError(f"No data stored for {kv_type} layer {layer_idx}")

        return torch.cat(decoded, dim=2)

    def reset(self) -> None:
        """Clear all cached data and reset references."""
        self._stored.clear()
        for comp in self.compressors.values():
            comp.reset()
