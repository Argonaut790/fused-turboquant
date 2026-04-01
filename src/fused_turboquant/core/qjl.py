"""
QJL (Quantized Johnson-Lindenstrauss) residual correction.

After TurboQuant quantization, the residual (original - reconstructed) contains
information lost during quantization. QJL projects this residual via a random
JL matrix and stores only the 1-bit signs, which are used during decode to
correct the quantization bias.

Cost: ~1 extra bit per element (d/8 bytes per vector for the sign bits).
Benefit: significant quality recovery, especially for low bit-rates (2-3 bit).

Reference: TurboQuant paper §3.2 (Zandieh et al., 2025).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class QJLCorrection:
    """Stored correction data for QJL residual."""

    sign_bits: torch.Tensor  # packed uint8, d/8 bytes per vector
    projection_scale: float  # scaling factor for correction


class QJLResidualCorrector:
    """QJL residual correction using random sign-bit projections.

    After quantization, computes residual = original - reconstructed,
    projects via a random matrix, stores 1-bit signs. During decode,
    uses signs + projection to add a correction term.

    The random projection matrix is generated from a seed (deterministic)
    and never stored — it's regenerated on the fly during decode.
    """

    def __init__(
        self,
        dim: int,
        projection_dim: Optional[int] = None,
        seed: int = 137,
        device: torch.device | str = "cpu",
    ):
        self.dim = dim
        self.projection_dim = projection_dim or dim
        self.seed = seed
        self.device = device
        self._proj = self._generate_projection()

    def _generate_projection(self) -> torch.Tensor:
        """Generate a random ±1/√m projection matrix (Rademacher)."""
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        signs = (
            torch.randint(
                0,
                2,
                (self.projection_dim, self.dim),
                generator=gen,
                dtype=torch.float32,
            )
            * 2
            - 1
        )
        scale = 1.0 / (self.projection_dim**0.5)
        return (signs * scale).to(self.device)

    def encode(self, residual: torch.Tensor) -> QJLCorrection:
        """Project residual and store 1-bit signs.

        Args:
            residual: [..., dim] float tensor (original - reconstructed).

        Returns:
            QJLCorrection with packed sign bits.
        """
        projected = residual.float() @ self._proj.T  # [..., projection_dim]
        scale = projected.abs().mean().item()
        signs = (projected >= 0).to(torch.uint8)  # [..., projection_dim]

        packed = self._pack_bits(signs)
        return QJLCorrection(sign_bits=packed, projection_scale=scale)

    def decode(self, correction: QJLCorrection) -> torch.Tensor:
        """Reconstruct approximate residual from sign bits.

        Args:
            correction: QJLCorrection from encode().

        Returns:
            Approximate residual tensor [..., dim].
        """
        signs = self._unpack_bits(correction.sign_bits)
        signed = signs.float() * 2 - 1  # {0,1} -> {-1,+1}
        reconstructed = signed @ self._proj  # [..., dim]
        return reconstructed * correction.projection_scale

    def _pack_bits(self, bits: torch.Tensor) -> torch.Tensor:
        """Pack boolean/uint8 bits into uint8 bytes (8 per byte)."""
        shape = bits.shape
        flat = bits.reshape(*shape[:-1], -1, 8)
        packed = torch.zeros(*flat.shape[:-1], dtype=torch.uint8, device=bits.device)
        for i in range(8):
            packed |= flat[..., i].to(torch.uint8) << i
        return packed

    def _unpack_bits(self, packed: torch.Tensor) -> torch.Tensor:
        """Unpack uint8 bytes back to individual bits."""
        bits = []
        for i in range(8):
            bits.append((packed >> i) & 1)
        return torch.stack(bits, dim=-1).reshape(
            *packed.shape[:-1],
            packed.shape[-1] * 8,
        )

    def to(self, device: torch.device | str) -> "QJLResidualCorrector":
        self.device = device
        self._proj = self._proj.to(device)
        return self


def apply_qjl_correction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    corrector: QJLResidualCorrector,
) -> tuple[torch.Tensor, QJLCorrection]:
    """Encode QJL correction and return corrected reconstruction.

    Args:
        original: original vectors [..., dim].
        reconstructed: TurboQuant decoded vectors [..., dim].
        corrector: QJLResidualCorrector instance.

    Returns:
        (corrected_reconstruction, correction_data)
    """
    residual = original.float() - reconstructed.float()
    correction = corrector.encode(residual)
    corrected = reconstructed.float() + corrector.decode(correction)
    return corrected, correction
