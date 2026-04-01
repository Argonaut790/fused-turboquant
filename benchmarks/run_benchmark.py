"""
Run the full fused-turboquant benchmark suite.

Usage:
    uv run python benchmarks/run_benchmark.py
    uv run python benchmarks/run_benchmark.py --dim 256 --device cuda
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fused_turboquant.benchmark.runner import (
    BenchmarkSuite,
    benchmark_quality,
    benchmark_rotation,
    print_results,
)


def main():
    parser = argparse.ArgumentParser(description="fused-turboquant Benchmarks")
    parser.add_argument("--dim", type=int, default=256, help="Vector dimension (power of 2)")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    print(f"\nRunning benchmarks with dim={args.dim} on {args.device}")
    print("=" * 80)

    suite = BenchmarkSuite()

    print("\n[1/2] Rotation benchmark...")
    for batch in [256, 1024, 4096, 16384]:
        suite.rotation_results.extend(
            benchmark_rotation(dim=args.dim, batch_size=batch, device=args.device)
        )

    print("[2/2] Quality benchmark...")
    suite.quality_results.extend(
        benchmark_quality(dim=args.dim, num_vectors=2048, device=args.device)
    )

    print_results(suite)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "rotation": [asdict(r) for r in suite.rotation_results],
            "quality": [asdict(r) for r in suite.quality_results],
        }
        output_path.write_text(json.dumps(data, indent=2))
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
