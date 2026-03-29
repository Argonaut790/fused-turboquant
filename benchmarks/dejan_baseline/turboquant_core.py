"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

Reference implementation based on arXiv:2504.19874 (ICLR 2026).

Key change from v2: the random rotation uses a precomputed orthogonal matrix
(via QR decomposition) instead of a Python-loop Hadamard transform. Same
mathematical properties (random rotation on S^{d-1}), but executes as a
single torch.matmul — fast on GPU.
"""

import torch
import math


# ============================================================================
# Random orthogonal rotation (replaces Hadamard)
# ============================================================================

def make_rotation_matrix(d: int, seed: int = 0, device: str = "cpu") -> torch.Tensor:
    """Generate a random orthogonal d×d matrix via QR decomposition.
    Deterministic given seed. Stored once, reused for all quantization calls."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    # Ensure proper rotation (det=+1) by fixing sign ambiguity
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
    return Q.to(device)


# ============================================================================
# Lloyd-Max codebook for Beta-distributed coordinates on S^{d-1}
# ============================================================================

def _beta_pdf_unnorm(x: torch.Tensor, d: int) -> torch.Tensor:
    """Unnormalized PDF of coordinate distribution after rotation on S^{d-1}."""
    alpha = (d - 1) / 2.0
    return torch.exp((alpha - 1) * torch.log(torch.clamp(1 - x * x, min=1e-30)))


def build_lloyd_max_codebook(
    d: int, bits: int, n_iter: int = 300, grid_size: int = 50000
) -> torch.Tensor:
    """Optimal Lloyd-Max centroids for Beta((d-1)/2, (d-1)/2) on [-1, 1].
    Runs on CPU (one-time cost)."""
    n_levels = 1 << bits
    sigma = 1.0 / math.sqrt(d) if d > 1 else 0.5
    lo = max(-1.0 + 1e-7, -6 * sigma)
    hi = min(1.0 - 1e-7, 6 * sigma)
    grid = torch.linspace(lo, hi, grid_size)
    pdf = _beta_pdf_unnorm(grid, d)
    pdf = pdf / pdf.sum()

    cdf = pdf.cumsum(0)
    cdf = cdf / cdf[-1]
    targets = torch.linspace(1 / (2 * n_levels), 1 - 1 / (2 * n_levels), n_levels)
    centroid_idx = torch.searchsorted(cdf, targets).clamp(0, grid_size - 1)
    centroids = grid[centroid_idx]

    for _ in range(n_iter):
        dists = (grid.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=1)
        new_centroids = torch.zeros_like(centroids)
        for i in range(n_levels):
            mask = assignments == i
            if mask.any():
                w = pdf[mask]
                new_centroids[i] = (grid[mask] * w).sum() / w.sum()
            else:
                new_centroids[i] = centroids[i]
        centroids = new_centroids

    return centroids.sort().values


# ============================================================================
# TurboQuant_mse
# ============================================================================

class TurboQuantMSE:
    """MSE-optimal TurboQuant: random orthogonal rotation + b-bit Lloyd-Max.

    Args:
        d: vector dimension (head_dim for KV cache)
        bits: bits per coordinate (3 or 4 recommended)
        device: torch device string
        rotation_seed: seed for the orthogonal matrix
    """

    def __init__(self, d: int, bits: int = 4, device: str = "cpu", rotation_seed: int = 0):
        self.d = d
        self.bits = bits
        self.device = device

        # Precompute rotation matrix (d × d) — lives on device
        self.Q = make_rotation_matrix(d, seed=rotation_seed, device=device)
        self.Q_T = self.Q.T.contiguous()  # cache transpose for inverse

        # Build codebook on CPU, then move to device
        print(f"  Building Lloyd-Max codebook (d={d}, bits={bits})...", end=" ", flush=True)
        centroids = build_lloyd_max_codebook(d, bits)
        self.centroids = centroids.to(device)
        self.boundaries = ((centroids[:-1] + centroids[1:]) / 2).to(device)
        print("done.")

    def quantize(self, x: torch.Tensor) -> dict:
        """Quantize vectors.
        Args:  x: (..., d) float tensors
        Returns: dict with 'idx' and 'norms'
        """
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms

        # Rotate: single matmul, (..., d) @ (d, d) -> (..., d)
        x_rot = x_unit @ self.Q_T

        # Scalar quantize each coordinate
        idx = torch.searchsorted(self.boundaries, x_rot.contiguous())

        return {
            "idx": idx.to(torch.uint8 if self.bits <= 8 else torch.int16),
            "norms": norms.squeeze(-1).half(),
        }

    def dequantize(self, q: dict) -> torch.Tensor:
        """Reconstruct vectors. Returns (..., orig_d)."""
        x_rot = self.centroids[q["idx"].long()]

        # Inverse rotate: (..., d) @ (d, d)
        x_unit = x_rot @ self.Q

        return x_unit * q["norms"].float().unsqueeze(-1)

    def compressed_size_bytes(self, q: dict) -> int:
        idx_bits = q["idx"].numel() * self.bits
        norm_bytes = q["norms"].numel() * 2
        return idx_bits // 8 + norm_bytes


# ============================================================================
# Self-test
# ============================================================================

def self_test():
    torch.manual_seed(123)
    d, n = 256, 64
    x = torch.randn(n, d)

    for bits in [2, 3, 4]:
        tq = TurboQuantMSE(d=d, bits=bits)
        q = tq.quantize(x)
        x_hat = tq.dequantize(q)

        mse = ((x - x_hat) ** 2).mean().item()
        cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()

        pairs = 32
        ip_true = (x[:pairs].unsqueeze(1) * x[pairs:2*pairs].unsqueeze(0)).sum(-1)
        ip_est = (x_hat[:pairs].unsqueeze(1) * x[pairs:2*pairs].unsqueeze(0)).sum(-1)
        ip_corr = torch.corrcoef(torch.stack([ip_true.flatten(), ip_est.flatten()]))[0, 1].item()

        orig_bytes = x.numel() * 4
        comp_bytes = tq.compressed_size_bytes(q)

        print(f"TurboQuant_mse  d={d}  bits={bits}  n={n}")
        print(f"  MSE:                {mse:.6f}")
        print(f"  Mean cosine sim:    {cos:.4f}")
        print(f"  Inner-product corr: {ip_corr:.4f}")
        print(f"  Size: {orig_bytes:,} -> {comp_bytes:,} bytes  ({orig_bytes / comp_bytes:.1f}x)")
        print()


if __name__ == "__main__":
    self_test()
