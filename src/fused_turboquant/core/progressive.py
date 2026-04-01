"""
Progressive age-based compression for KV cache.

Tokens are not equally important for attention. Recent tokens are critical,
older tokens can tolerate more compression. This module manages tiered
compression where tokens are periodically re-quantized at lower precision
as they age during generation.

Default tiers (configurable):
    Tokens [seq_len-64, seq_len):       4-bit  (3.88x compression)
    Tokens [seq_len-256, seq_len-64):   3-bit  (5.12x compression)
    Tokens [0, seq_len-256):            2-bit  (7.53x compression)

For long contexts this achieves ~6-7x average compression while maintaining
quality for the attention-critical recent window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from fused_turboquant.core.quantizer import CompressedTensor, TurboQuantMSE

logger = logging.getLogger(__name__)


@dataclass
class CompressionTier:
    """A compression tier defining bit-rate for a token age range."""

    bits: int
    window_size: int  # tokens at the END of the sequence use this tier


@dataclass
class ProgressiveConfig:
    """Configuration for progressive age-based compression.

    Tiers are ordered from highest quality (recent) to lowest (oldest).
    The last tier covers all remaining tokens.
    """

    tiers: list[CompressionTier] = field(
        default_factory=lambda: [
            CompressionTier(bits=4, window_size=64),
            CompressionTier(bits=3, window_size=192),  # tokens 64-256 from end
            CompressionTier(bits=2, window_size=0),  # 0 = everything older
        ]
    )
    recompress_interval: int = 64  # re-quantize every N new tokens

    def get_tier_for_position(self, pos: int, seq_len: int) -> int:
        """Get the bit-rate for a token at position `pos` in a sequence of length `seq_len`."""
        age = seq_len - 1 - pos  # 0 = newest
        cumulative = 0
        for tier in self.tiers:
            if tier.window_size == 0:
                return tier.bits
            cumulative += tier.window_size
            if age < cumulative:
                return tier.bits
        return self.tiers[-1].bits

    def get_tier_boundaries(self, seq_len: int) -> list[tuple[int, int, int]]:
        """Return (start, end, bits) for each tier given current seq_len.

        Boundaries are in terms of token position [start, end).
        """
        boundaries = []
        remaining = seq_len
        for tier in self.tiers:
            if remaining <= 0:
                break
            if tier.window_size == 0:
                boundaries.append((0, remaining, tier.bits))
                break
            start = max(0, remaining - tier.window_size)
            end = remaining
            if start < end:
                boundaries.append((start, end, tier.bits))
            remaining = start
        boundaries.reverse()
        return boundaries


class ProgressiveKVStore:
    """Manages tiered KV cache with progressive re-compression.

    Stores compressed KV at different bit-rates based on token age.
    When a recompression event triggers, older tiers are re-quantized
    at lower precision to free memory.

    Each tier maintains its own TurboQuantMSE instance with the
    appropriate bit-rate.
    """

    def __init__(
        self,
        head_dim: int,
        config: ProgressiveConfig | None = None,
        device: torch.device | str = "cpu",
    ):
        self.head_dim = head_dim
        self.config = config or ProgressiveConfig()
        self.device = device

        self._quantizers: dict[int, TurboQuantMSE] = {}
        for tier in self.config.tiers:
            if tier.bits not in self._quantizers:
                self._quantizers[tier.bits] = TurboQuantMSE(
                    head_dim=head_dim,
                    bits=tier.bits,
                    device=str(device),
                )

        self._tokens_since_recompress = 0
        self._segments: list[dict] = []

    def add_token(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Add a new token's KV at the highest-quality tier.

        Args:
            key_states: [batch, n_kv_heads, 1, head_dim] float.
            value_states: [batch, n_kv_heads, 1, head_dim] float.
        """
        best_bits = self.config.tiers[0].bits
        tq = self._quantizers[best_bits]
        k_compressed = tq.encode(key_states.float())
        v_compressed = tq.encode(value_states.float())
        self._segments.append(
            {
                "k": k_compressed,
                "v": v_compressed,
                "bits": best_bits,
            }
        )
        self._tokens_since_recompress += 1

        if self._tokens_since_recompress >= self.config.recompress_interval:
            self._recompress()
            self._tokens_since_recompress = 0

    def _recompress(self) -> None:
        """Re-quantize older segments at lower precision based on current position."""
        seq_len = len(self._segments)
        boundaries = self.config.get_tier_boundaries(seq_len)

        for start, end, target_bits in boundaries:
            for pos in range(start, end):
                seg = self._segments[pos]
                if seg["bits"] <= target_bits:
                    continue

                old_tq = self._quantizers[seg["bits"]]
                new_tq = self._quantizers[target_bits]

                k_decoded = old_tq.decode(seg["k"])
                v_decoded = old_tq.decode(seg["v"])
                seg["k"] = new_tq.encode(k_decoded)
                seg["v"] = new_tq.encode(v_decoded)
                seg["bits"] = target_bits

        bits_count: dict[int, int] = {}
        for seg in self._segments:
            b = seg["bits"]
            bits_count[b] = bits_count.get(b, 0) + 1
        logger.debug(
            "Progressive recompress: %d tokens, distribution: %s",
            seq_len,
            bits_count,
        )

    def get_all_keys(self) -> list[CompressedTensor]:
        """Return all compressed keys in order."""
        return [seg["k"] for seg in self._segments]

    def get_all_values(self) -> list[CompressedTensor]:
        """Return all compressed values in order."""
        return [seg["v"] for seg in self._segments]

    def get_average_bits(self) -> float:
        """Return the current average bit-rate across all tokens."""
        if not self._segments:
            return 0.0
        return sum(seg["bits"] for seg in self._segments) / len(self._segments)

    @property
    def seq_len(self) -> int:
        return len(self._segments)

    def reset(self) -> None:
        """Clear all stored KV data."""
        self._segments.clear()
        self._tokens_since_recompress = 0

    def to(self, device: torch.device | str) -> "ProgressiveKVStore":
        self.device = device
        for bits, tq in self._quantizers.items():
            self._quantizers[bits] = tq.to(device)
        return self
