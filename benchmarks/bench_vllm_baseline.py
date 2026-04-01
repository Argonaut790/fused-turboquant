"""
vLLM FP16 baseline benchmark: offline inference throughput.

Runs the same prompts through vLLM to establish an FP16 throughput ceiling.
This is the production inference engine baseline — our fused TurboQuant HF
numbers can be compared against this.

Usage:
    uv run python benchmarks/bench_vllm_baseline.py --model Qwen/Qwen3.5-9B
    uv run python benchmarks/bench_vllm_baseline.py --model Qwen/Qwen2.5-0.5B --fp8
"""

from __future__ import annotations

import argparse
import time

PROMPTS = [
    "Explain the theory of general relativity in simple terms.",
    "Write a Python function to compute the fibonacci sequence using dynamic programming.",
    "What are the main differences between TCP and UDP protocols?",
    "Describe the process of photosynthesis in detail.",
    "What is the significance of the Turing test in artificial intelligence?",
    "Explain how a neural network learns through backpropagation.",
    "What are the key principles of object-oriented programming?",
    "Describe the difference between supervised and unsupervised learning.",
]


def run_vllm_benchmark(
    model_name: str,
    max_tokens: int = 200,
    tp: int = 1,
    kv_cache_dtype: str = "auto",
    gpu_memory_utilization: float = 0.90,
):
    """Run offline inference via vLLM and measure throughput."""
    from vllm import LLM, SamplingParams

    print("\nvLLM Offline Benchmark")
    print(f"  Model: {model_name}")
    print(f"  KV cache dtype: {kv_cache_dtype}")
    print(f"  Tensor parallel: {tp}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Prompts: {len(PROMPTS)}")

    print("\nLoading model into vLLM...")
    t0 = time.time()
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp,
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
    )
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )

    # Warmup
    print("\nWarmup...")
    _ = llm.generate(PROMPTS[:2], sampling_params)

    # Actual benchmark
    print("Running benchmark...")
    t_start = time.perf_counter()
    outputs = llm.generate(PROMPTS, sampling_params)
    t_end = time.perf_counter()

    total_gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    total_time = t_end - t_start
    overall_tps = total_gen_tokens / total_time

    print(f"\n{'=' * 60}")
    print("vLLM RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total generated tokens: {total_gen_tokens}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Overall throughput: {overall_tps:.1f} tok/s")
    print()

    per_request = []
    for i, output in enumerate(outputs):
        gen_len = len(output.outputs[0].token_ids)
        per_request.append(gen_len)
        first_50 = output.outputs[0].text[:50].replace("\n", " ")
        print(f'  Prompt {i + 1}: {gen_len} tokens — "{first_50}..."')

    print(f"\n  Avg tokens/request: {sum(per_request) / len(per_request):.0f}")
    print(f"  Throughput: {overall_tps:.1f} tok/s (all {len(PROMPTS)} prompts batched)")

    return {
        "model": model_name,
        "kv_cache_dtype": kv_cache_dtype,
        "total_tokens": total_gen_tokens,
        "total_time_s": total_time,
        "throughput_tps": overall_tps,
        "per_request_tokens": per_request,
    }


def main():
    parser = argparse.ArgumentParser(description="vLLM FP16 baseline throughput benchmark")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-0.5B", help="HuggingFace model name"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=200, help="Max tokens to generate per prompt"
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--fp8", action="store_true", help="Use FP8 KV cache (fp8_e5m2)")
    parser.add_argument(
        "--gpu-mem", type=float, default=0.90, help="GPU memory utilization for vLLM"
    )
    args = parser.parse_args()

    kv_dtype = "fp8_e5m2" if args.fp8 else "auto"

    results = [
        run_vllm_benchmark(
            args.model,
            args.max_tokens,
            args.tp,
            kv_cache_dtype=kv_dtype,
            gpu_memory_utilization=args.gpu_mem,
        )
    ]

    if not args.fp8:
        print("\n\nTip: run with --fp8 to also benchmark vLLM's FP8 KV cache (fp8_e5m2)")
        print("     uv run python benchmarks/bench_vllm_baseline.py --model <model> --fp8")


if __name__ == "__main__":
    main()
