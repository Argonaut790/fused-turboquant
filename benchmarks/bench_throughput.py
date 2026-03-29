"""
Throughput benchmark: measure tokens/second for generation.

Compares:
  1. FP16 baseline (no compression, standard HF generate)
  2. SimulationCache (roundtrip compression, measures overhead)
  3. FusedTurboQuant cache (compressed keys + fused attention)
  4. Dejan.ai (Dense QR rotation, roundtrip quantize/dequantize)

Usage:
    uv run python benchmarks/bench_throughput.py --model Qwen/Qwen2.5-0.5B
    uv run python benchmarks/bench_throughput.py --model Qwen/Qwen3.5-9B --bits 4
    uv run python benchmarks/bench_throughput.py --model Qwen/Qwen2.5-0.5B --no-dejan
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DEJAN_DIR = Path(__file__).resolve().parent / "dejan_baseline"


def load_model_and_tokenizer(model_name: str, dtype: torch.dtype = torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_name}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s, device={next(model.parameters()).device}")
    return model, tokenizer


PROMPTS = [
    "Explain the theory of general relativity in simple terms.",
    "Write a Python function to compute the fibonacci sequence using dynamic programming.",
    "What are the main differences between TCP and UDP protocols?",
]


def measure_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    cache_factory=None,
    warmup_runs: int = 1,
    label: str = "",
) -> dict:
    """Measure time-to-first-token and generation throughput."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    for _ in range(warmup_runs):
        kwargs = {}
        if cache_factory is not None:
            kwargs["past_key_values"] = cache_factory()
            kwargs["use_cache"] = True
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=5, do_sample=False, **kwargs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    kwargs = {}
    if cache_factory is not None:
        kwargs["past_key_values"] = cache_factory()
        kwargs["use_cache"] = True

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event.record()

    t_wall_start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **kwargs,
        )

    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        gpu_time_ms = start_event.elapsed_time(end_event)
    else:
        gpu_time_ms = None

    t_wall_end = time.perf_counter()
    wall_time_s = t_wall_end - t_wall_start

    gen_tokens = output.shape[-1] - input_len
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    timing = gpu_time_ms / 1000.0 if gpu_time_ms is not None else wall_time_s
    tps = gen_tokens / timing if timing > 0 else 0

    return {
        "label": label,
        "gen_tokens": gen_tokens,
        "wall_time_s": wall_time_s,
        "gpu_time_s": gpu_time_ms / 1000.0 if gpu_time_ms else None,
        "tokens_per_sec": tps,
        "peak_memory_mb": peak_mem,
        "input_len": input_len,
    }


def _try_import_dejan():
    """Try to import Dejan.ai baseline. Returns make_quantized_cache or None."""
    try:
        sys.path.insert(0, str(DEJAN_DIR))
        from turboquant_kv_cache import make_quantized_cache

        return make_quantized_cache
    except ImportError:
        return None


def run_benchmark(
    model, tokenizer, max_new_tokens: int, bits: int, include_dejan: bool = True,
):
    """Run full throughput comparison."""
    from fused_turboquant.hf.fused_cache import patch_model, unpatch_model
    from fused_turboquant.hf.simulation_cache import make_simulation_cache

    methods = {
        "FP16 baseline": None,
        f"SimulationCache TQ{bits}": lambda b=bits: make_simulation_cache(bits=b),
    }

    if include_dejan:
        make_quantized_cache = _try_import_dejan()
        if make_quantized_cache is not None:
            methods[f"Dejan.ai TQ{bits}"] = (
                lambda b=bits, f=make_quantized_cache: f(bits=b)
            )
        else:
            print("  WARNING: Could not import Dejan baseline — skipping.")
            print(f"           Expected at: {DEJAN_DIR}/turboquant_kv_cache.py")

    all_results = {k: [] for k in methods}
    all_results[f"FusedTurboQuant TQ{bits}"] = []

    for prompt_idx, prompt in enumerate(PROMPTS):
        print(f"\n  Prompt {prompt_idx + 1}/{len(PROMPTS)}: \"{prompt[:60]}...\"")

        for method_name, factory in methods.items():
            r = measure_generation(
                model, tokenizer, prompt, max_new_tokens,
                cache_factory=factory, label=method_name,
            )
            all_results[method_name].append(r)
            print(f"    {method_name}: {r['tokens_per_sec']:.1f} tok/s "
                  f"({r['gen_tokens']} tokens, {r['wall_time_s']:.2f}s, "
                  f"peak={r['peak_memory_mb']:.0f} MB)")

        fused_label = f"FusedTurboQuant TQ{bits}"
        cache = patch_model(model, bits=bits)
        r = measure_generation(
            model, tokenizer, prompt, max_new_tokens,
            cache_factory=lambda c=cache: c, label=fused_label,
        )
        all_results[fused_label].append(r)
        unpatch_model(model)
        print(f"    {fused_label}: {r['tokens_per_sec']:.1f} tok/s "
              f"({r['gen_tokens']} tokens, {r['wall_time_s']:.2f}s, "
              f"peak={r['peak_memory_mb']:.0f} MB)")

    return all_results


def print_summary(all_results: dict, max_new_tokens: int, model_name: str):
    print(f"\n{'=' * 70}")
    print("THROUGHPUT SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Model: {model_name}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Prompts: {len(PROMPTS)}")
    print()

    header = f"  {'Method':<30s} | {'Avg TPS':>10s} | {'Avg Time':>10s} | {'Peak Mem':>10s}"
    print(header)
    print(f"  {'-' * 30}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")

    for method, results in all_results.items():
        if not results:
            continue
        avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
        avg_time = sum(r["wall_time_s"] for r in results) / len(results)
        max_mem = max(r["peak_memory_mb"] for r in results)
        print(f"  {method:<30s} | {avg_tps:>8.1f}/s | {avg_time:>8.2f}s | {max_mem:>8.0f} MB")


def main():
    parser = argparse.ArgumentParser(description="Throughput benchmark: token generation speed")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="HuggingFace model name")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4],
                        help="Quantization bit-width")
    parser.add_argument("--max-new-tokens", type=int, default=200,
                        help="Max tokens to generate per prompt")
    parser.add_argument("--no-dejan", action="store_true",
                        help="Skip Dejan.ai comparison")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"],
                        help="Model dtype")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=dtype)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nBenchmarking with {args.bits}-bit TurboQuant, max_new_tokens={args.max_new_tokens}")
    all_results = run_benchmark(
        model, tokenizer, args.max_new_tokens, args.bits,
        include_dejan=not args.no_dejan,
    )
    print_summary(all_results, args.max_new_tokens, args.model)


if __name__ == "__main__":
    main()
