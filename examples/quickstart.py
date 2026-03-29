"""
fused-turboquant quickstart example.

Demonstrates the core quantization pipeline on random vectors,
showing quality metrics at different bit-widths.

Usage:
    uv run python examples/quickstart.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from fused_turboquant import TurboQuantMSE


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    head_dim = 256
    num_vectors = 512
    x = torch.randn(num_vectors, head_dim, device=device)

    print(f"\nInput: {num_vectors} vectors × {head_dim} dimensions")
    print(f"Original size: {x.numel() * 2:,} bytes (fp16)")
    print()

    for bits in [4, 3, 2]:
        tq = TurboQuantMSE(head_dim=head_dim, bits=bits, device=device)

        compressed = tq.encode(x)
        x_hat = tq.decode(compressed)

        x_n = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
        xh_n = x_hat / (torch.norm(x_hat, dim=-1, keepdim=True) + 1e-8)
        cosine = torch.mean(torch.sum(x_n * xh_n, dim=-1)).item()
        mse = torch.mean((x - x_hat) ** 2).item()

        print(f"  {bits}-bit TurboQuant_MSE:")
        print(f"    Cosine similarity : {cosine:.6f}")
        print(f"    MSE               : {mse:.6f}")
        print(f"    Compression ratio : {compressed.compression_ratio:.1f}x")
        print()


if __name__ == "__main__":
    main()
