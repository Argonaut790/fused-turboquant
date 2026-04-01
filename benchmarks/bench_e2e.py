"""
End-to-end benchmark: comparison on a real HuggingFace model.

Compares:
  1. FP16 baseline (no compression, standard HF generate)
  2. FusedTQ (compressed key storage + fused Triton attention)
  3. Dejan.ai TQ (Dense QR rotation, roundtrip quantize/dequantize)

Modes:
  - Standard throughput (default): 5-prompt average, short context
  - Long-context decode (--long-context): sweep 4K-32K token prompts
  - Batch throughput (--batch-search): find max batch size per method

Usage:
    uv run python benchmarks/bench_e2e.py --model Qwen/Qwen3-8B --bits 3
    uv run python benchmarks/bench_e2e.py --model Qwen/Qwen3-8B --bits 3 --long-context
    uv run python benchmarks/bench_e2e.py --model Qwen/Qwen3-8B --bits 3 --batch-search
    uv run python benchmarks/bench_e2e.py --model Qwen/Qwen3-8B --bits 3 --json results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DEJAN_DIR = Path(__file__).resolve().parent / "dejan_baseline"

PROMPTS = [
    "Explain the theory of general relativity in simple terms.",
    "Write a Python function to compute the fibonacci sequence using dynamic programming.",
    "What are the main differences between TCP and UDP protocols?",
    "Describe the process of photosynthesis in detail.",
    "What is the significance of the Turing test in artificial intelligence?",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(model_name: str, dtype: torch.dtype = torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_name}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s, device={next(model.parameters()).device}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Throughput measurement
# ---------------------------------------------------------------------------


def _reset_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def measure_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    cache_factory=None,
    warmup_runs: int = 1,
) -> dict:
    """Measure tok/s and peak GPU memory for a single prompt."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    for _ in range(warmup_runs):
        kwargs = {}
        if cache_factory is not None:
            kwargs["past_key_values"] = cache_factory()
            kwargs["use_cache"] = True
        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                **kwargs,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    _reset_gpu()

    if torch.cuda.is_available():
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_evt.record()

    t_start = time.perf_counter()

    kwargs = {}
    if cache_factory is not None:
        kwargs["past_key_values"] = cache_factory()
        kwargs["use_cache"] = True

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **kwargs,
        )

    if torch.cuda.is_available():
        end_evt.record()
        torch.cuda.synchronize()
        gpu_ms = start_evt.elapsed_time(end_evt)
    else:
        gpu_ms = None

    wall_s = time.perf_counter() - t_start
    gen_tokens = output.shape[-1] - input_len
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    timing = gpu_ms / 1000.0 if gpu_ms is not None else wall_s
    tps = gen_tokens / timing if timing > 0 else 0

    return {
        "gen_tokens": gen_tokens,
        "wall_time_s": wall_s,
        "tokens_per_sec": tps,
        "peak_memory_mb": peak_mem,
    }


# ---------------------------------------------------------------------------
# Perplexity measurement (optional)
# ---------------------------------------------------------------------------


def load_wikitext(tokenizer, max_length: int = 2048, stride: int = 512):
    from datasets import load_dataset

    print("Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    total_tokens = input_ids.shape[1]
    print(f"  Total tokens: {total_tokens:,}")

    windows = []
    for begin in range(0, total_tokens - max_length, stride):
        end = begin + max_length
        windows.append((begin, end))
        if len(windows) >= 50:
            break
    print(f"  Using {len(windows)} windows (length={max_length}, stride={stride})")
    return input_ids, windows


def compute_perplexity(
    model,
    input_ids: torch.Tensor,
    windows: list[tuple[int, int]],
    cache_factory=None,
    stride: int = 512,
) -> dict:
    device = next(model.parameters()).device
    nlls = []
    for i, (begin, end) in enumerate(windows):
        ids = input_ids[:, begin:end].to(device)
        target_len = ids.shape[1] - stride if begin > 0 else ids.shape[1]

        kwargs = {}
        if cache_factory is not None:
            kwargs["past_key_values"] = cache_factory()
            kwargs["use_cache"] = True

        with torch.inference_mode():
            outputs = model(ids, **kwargs)

        logits = outputs.logits
        shift_logits = logits[:, -(target_len - 1) :, :].contiguous()
        shift_labels = ids[:, -(target_len - 1) + 1 : end].contiguous()

        min_len = min(shift_logits.shape[1], shift_labels.shape[1])
        shift_logits = shift_logits[:, :min_len, :]
        shift_labels = shift_labels[:, :min_len]

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )
        nlls.append(loss.item())

        if (i + 1) % 10 == 0:
            print(f"    [{i + 1}/{len(windows)}] running ppl={np.exp(np.mean(nlls)):.2f}")

    return {"perplexity": float(np.exp(np.mean(nlls))), "n_windows": len(windows)}


# ---------------------------------------------------------------------------
# Method definitions
# ---------------------------------------------------------------------------


def build_methods(model, bits: int, include_dejan: bool):
    """Return a list of method dicts, each with label, factory, setup, teardown."""
    from fused_turboquant.hf.fused_cache import patch_model, unpatch_model

    methods: list[dict] = []

    methods.append(
        {
            "label": "FP16 baseline",
            "factory": None,
            "setup": None,
            "teardown": None,
        }
    )

    methods.append(
        {
            "label": f"FusedTQ{bits} (ours)",
            "factory": "fused",
            "setup": lambda: patch_model(model, bits=bits),
            "teardown": lambda: unpatch_model(model),
        }
    )

    if include_dejan:
        try:
            sys.path.insert(0, str(DEJAN_DIR))
            from turboquant_kv_cache import make_quantized_cache

            methods.append(
                {
                    "label": f"TQ{bits} (Dejan.ai)",
                    "factory": lambda b=bits: make_quantized_cache(bits=b),
                    "setup": None,
                    "teardown": None,
                }
            )
        except ImportError:
            print("  WARNING: Could not import Dejan baseline — skipping.")
            print(f"           Expected at: {DEJAN_DIR}/turboquant_kv_cache.py")

    return methods


# ---------------------------------------------------------------------------
# Main benchmark logic
# ---------------------------------------------------------------------------


def run_throughput(model, tokenizer, methods, max_new_tokens: int) -> dict[str, dict]:
    results = {}
    for m in methods:
        label = m["label"]
        print(f"\n  --- {label} ---")

        try:
            cache_obj = None
            if m["setup"] is not None:
                cache_obj = m["setup"]()

            all_tps, all_mem, all_wall = [], [], []
            for i, prompt in enumerate(PROMPTS):
                if m["factory"] == "fused":

                    def fused_factory(c=cache_obj):
                        c.reset()
                        return c

                    factory = fused_factory
                else:
                    factory = m["factory"]

                r = measure_generation(model, tokenizer, prompt, max_new_tokens, factory)
                all_tps.append(r["tokens_per_sec"])
                all_mem.append(r["peak_memory_mb"])
                all_wall.append(r["wall_time_s"])
                print(
                    f"    Prompt {i + 1}: {r['tokens_per_sec']:.1f} tok/s, "
                    f"{r['peak_memory_mb']:.0f} MB peak, {r['wall_time_s']:.2f}s"
                )

            if m["teardown"] is not None:
                m["teardown"]()

            results[label] = {
                "avg_tps": sum(all_tps) / len(all_tps),
                "max_peak_memory_mb": max(all_mem),
                "avg_wall_time_s": sum(all_wall) / len(all_wall),
            }
        except Exception as e:
            print(f"    ERROR: {e}")
            print(f"    Skipping {label}")
            if m["teardown"] is not None:
                try:
                    m["teardown"]()
                except Exception:
                    pass

    return results


def run_quality(model, tokenizer, methods, max_length: int, stride: int) -> dict[str, dict]:
    input_ids, windows = load_wikitext(tokenizer, max_length, stride)
    results = {}

    for m in methods:
        label = m["label"]
        print(f"\n  --- {label} ---")
        _reset_gpu()

        try:
            if m["factory"] == "fused":
                cache_obj = m["setup"]()

                def fused_factory(c=cache_obj):
                    c.reset()
                    return c

                factory = fused_factory
            elif m["factory"] is not None:
                factory = m["factory"]
            else:
                factory = None

            t0 = time.time()
            r = compute_perplexity(model, input_ids, windows, factory, stride)
            elapsed = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

            if m["factory"] == "fused" and m["teardown"] is not None:
                m["teardown"]()

            r["time_s"] = elapsed
            r["peak_memory_mb"] = peak
            results[label] = r
            print(f"    Perplexity: {r['perplexity']:.2f} ({elapsed:.1f}s, {peak:.0f} MB peak)")
        except Exception as e:
            print(f"    ERROR: {e}")
            print(f"    Skipping {label}")
            if m["factory"] == "fused" and m["teardown"] is not None:
                try:
                    m["teardown"]()
                except Exception:
                    pass

    return results


# ---------------------------------------------------------------------------
# Long-context decode benchmark
# ---------------------------------------------------------------------------


def _load_long_text_ids(tokenizer, max_tokens: int) -> torch.Tensor:
    """Load WikiText-2 and tokenize into a single long sequence."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids
    if ids.shape[1] < max_tokens:
        reps = (max_tokens // ids.shape[1]) + 1
        ids = ids.repeat(1, reps)
    return ids[:, :max_tokens]


def measure_generation_from_ids(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    cache_factory=None,
) -> dict:
    """Measure decode tok/s and peak memory from pre-tokenized input_ids."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    input_len = input_ids.shape[-1]

    _reset_gpu()

    kwargs = {}
    if cache_factory is not None:
        kwargs["past_key_values"] = cache_factory()
        kwargs["use_cache"] = True

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

    t_start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **kwargs,
        )

    if torch.cuda.is_available():
        end_evt.record()
        torch.cuda.synchronize()
        gpu_ms = start_evt.elapsed_time(end_evt)
    else:
        gpu_ms = None

    wall_s = time.perf_counter() - t_start
    gen_tokens = output.shape[-1] - input_len
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    timing = gpu_ms / 1000.0 if gpu_ms is not None else wall_s
    tps = gen_tokens / timing if timing > 0 else 0

    return {
        "gen_tokens": gen_tokens,
        "wall_time_s": wall_s,
        "tokens_per_sec": tps,
        "peak_memory_mb": peak_mem,
        "context_length": input_len,
    }


def run_long_context(
    model,
    tokenizer,
    methods,
    context_lengths: list[int],
    gen_tokens: int = 100,
) -> dict[str, dict]:
    """Sweep context lengths and measure decode throughput for each method."""
    max_ctx = max(context_lengths)
    print(f"  Loading long text ({max_ctx:,} tokens)...")
    all_ids = _load_long_text_ids(tokenizer, max_ctx)
    print(f"  Loaded {all_ids.shape[1]:,} tokens from WikiText-2")

    results: dict[str, dict] = {}

    for m in methods:
        label = m["label"]
        print(f"\n  --- {label} ---")
        method_results: dict[int, dict] = {}

        try:
            cache_obj = None
            if m["setup"] is not None:
                cache_obj = m["setup"]()

            for ctx in context_lengths:
                ids = all_ids[:, :ctx]

                if m["factory"] == "fused":

                    def fused_factory(c=cache_obj):
                        c.reset()
                        return c

                    factory = fused_factory
                else:
                    factory = m["factory"]

                _reset_gpu()
                r = measure_generation_from_ids(model, ids, gen_tokens, factory)
                method_results[ctx] = r
                print(
                    f"    ctx={ctx:>6,d}: {r['tokens_per_sec']:>6.1f} tok/s, "
                    f"{r['peak_memory_mb']:>8.0f} MB peak, "
                    f"{r['gen_tokens']} tokens in {r['wall_time_s']:.1f}s"
                )

            if m["teardown"] is not None:
                m["teardown"]()

            results[label] = method_results

        except Exception as e:
            print(f"    ERROR: {e}")
            print(f"    Skipping {label}")
            if m["teardown"] is not None:
                try:
                    m["teardown"]()
                except Exception:
                    pass

    return results


def print_long_context_table(
    results: dict[str, dict],
    model_name: str,
    bits: int,
):
    print(f"\n{'=' * 90}")
    print("LONG-CONTEXT DECODE THROUGHPUT")
    print(f"{'=' * 90}")
    print(f"  Model: {model_name}  |  Bits: {bits}")
    print()

    methods = list(results.keys())
    ctx_lengths = sorted({ctx for m_res in results.values() for ctx in m_res})

    header_parts = [f"{'Context':>8s}"]
    for m in methods:
        header_parts.append(f"{m[:20]:>22s}")
    print("  " + " | ".join(header_parts))
    print("  " + "-+-".join(["-" * 8] + ["-" * 22] * len(methods)))

    for ctx in ctx_lengths:
        row = [f"{ctx:>8,d}"]
        for m in methods:
            if ctx in results[m]:
                r = results[m][ctx]
                row.append(f"{r['tokens_per_sec']:>6.1f}/s {r['peak_memory_mb']:>7.0f}MB")
            else:
                row.append(f"{'N/A':>22s}")
        print("  " + " | ".join(row))


# ---------------------------------------------------------------------------
# Batch throughput benchmark
# ---------------------------------------------------------------------------


def _try_batch_generate(
    model,
    tokenizer,
    prompt: str,
    batch_size: int,
    max_new_tokens: int,
    cache_factory=None,
) -> dict | None:
    """Try generating with a given batch size; return results or None on OOM."""
    device = next(model.parameters()).device
    prompts = [prompt] * batch_size
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    input_len = inputs["input_ids"].shape[-1]

    _reset_gpu()

    kwargs = {}
    if cache_factory is not None:
        kwargs["past_key_values"] = cache_factory()
        kwargs["use_cache"] = True

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()

        t_start = time.perf_counter()

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                **kwargs,
            )

        if torch.cuda.is_available():
            end_evt.record()
            torch.cuda.synchronize()
            gpu_ms = start_evt.elapsed_time(end_evt)
        else:
            gpu_ms = None

        wall_s = time.perf_counter() - t_start
        total_gen = (output.shape[-1] - input_len) * batch_size
        peak_mem = (
            torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        )
        timing = gpu_ms / 1000.0 if gpu_ms is not None else wall_s
        agg_tps = total_gen / timing if timing > 0 else 0

        return {
            "batch_size": batch_size,
            "total_gen_tokens": total_gen,
            "aggregate_tps": agg_tps,
            "per_seq_tps": agg_tps / batch_size,
            "wall_time_s": wall_s,
            "peak_memory_mb": peak_mem,
        }

    except torch.cuda.OutOfMemoryError:
        _reset_gpu()
        return None


def run_batch_throughput(
    model,
    tokenizer,
    methods,
    gen_tokens: int = 100,
) -> dict[str, dict]:
    """Find max batch size per method and measure aggregate throughput."""
    prompt = PROMPTS[0]
    results: dict[str, dict] = {}

    for m in methods:
        label = m["label"]
        print(f"\n  --- {label} ---")

        try:
            cache_obj = None
            if m["setup"] is not None:
                cache_obj = m["setup"]()

            if m["factory"] == "fused":

                def fused_factory(c=cache_obj):
                    c.reset()
                    return c

                factory = fused_factory
            else:
                factory = m["factory"]

            max_batch = 0
            best_result = None
            for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
                print(f"    Trying batch_size={bs}...", end=" ", flush=True)
                r = _try_batch_generate(
                    model,
                    tokenizer,
                    prompt,
                    bs,
                    gen_tokens,
                    factory,
                )
                if r is None:
                    print("OOM")
                    break
                print(f"{r['aggregate_tps']:.1f} agg tok/s, {r['peak_memory_mb']:.0f} MB")
                max_batch = bs
                best_result = r

            if m["teardown"] is not None:
                m["teardown"]()

            if best_result is not None:
                results[label] = best_result

        except Exception as e:
            print(f"    ERROR: {e}")
            print(f"    Skipping {label}")
            if m["teardown"] is not None:
                try:
                    m["teardown"]()
                except Exception:
                    pass

    return results


def print_batch_table(results: dict[str, dict], model_name: str, bits: int):
    print(f"\n{'=' * 90}")
    print("BATCH THROUGHPUT (max batch before OOM)")
    print(f"{'=' * 90}")
    print(f"  Model: {model_name}  |  Bits: {bits}")
    print()

    header = (
        f"  {'Method':<28s} | {'MaxBatch':>8s} | {'Agg TPS':>10s} "
        f"| {'Per-Seq':>10s} | {'Peak Mem':>10s}"
    )
    print(header)
    print(f"  {'-' * 28}-+-{'-' * 8}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")

    for label, r in results.items():
        print(
            f"  {label:<28s} | {r['batch_size']:>8d} | "
            f"{r['aggregate_tps']:>8.1f}/s | "
            f"{r['per_seq_tps']:>8.1f}/s | "
            f"{r['peak_memory_mb']:>8.0f} MB"
        )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_throughput_table(results: dict[str, dict], model_name: str, bits: int):
    print(f"\n{'=' * 78}")
    print("THROUGHPUT & MEMORY COMPARISON")
    print(f"{'=' * 78}")
    print(f"  Model: {model_name}  |  Bits: {bits}  |  Prompts: {len(PROMPTS)}")
    print()

    header = f"  {'Method':<28s} | {'Avg TPS':>10s} | {'Peak Mem':>10s} | {'Avg Time':>10s}"
    print(header)
    print(f"  {'-' * 28}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")

    baseline_tps = None
    for label, r in results.items():
        if baseline_tps is None:
            baseline_tps = r["avg_tps"]
        ratio = r["avg_tps"] / baseline_tps if baseline_tps else 0
        print(
            f"  {label:<28s} | {r['avg_tps']:>8.1f}/s | "
            f"{r['max_peak_memory_mb']:>8.0f} MB | {r['avg_wall_time_s']:>8.2f}s"
            f"  ({ratio:.2f}x)"
        )


def print_quality_table(results: dict[str, dict]):
    print(f"\n{'=' * 78}")
    print("PERPLEXITY COMPARISON (WikiText-2)")
    print(f"{'=' * 78}")

    header = (
        f"  {'Method':<28s} | {'Perplexity':>12s} | {'Delta':>8s} "
        f"| {'Time':>8s} | {'Peak Mem':>10s}"
    )
    print(header)
    print(f"  {'-' * 28}-+-{'-' * 12}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 10}")

    baseline_ppl = None
    for label, r in results.items():
        if baseline_ppl is None:
            baseline_ppl = r["perplexity"]
        delta = r["perplexity"] - baseline_ppl
        delta_str = "baseline" if delta == 0 else f"+{delta:.2f}"
        print(
            f"  {label:<28s} | {r['perplexity']:>12.2f} | {delta_str:>8s} | "
            f"{r['time_s']:>7.1f}s | {r['peak_memory_mb']:>8.0f} MB"
        )


def save_json(
    model_name: str,
    bits: int,
    path: str,
    throughput: dict | None = None,
    quality: dict | None = None,
    long_context: dict | None = None,
    batch: dict | None = None,
):
    # Convert int keys (context lengths) to strings for JSON serialization
    lc_serializable = None
    if long_context is not None:
        lc_serializable = {
            method: {str(k): v for k, v in ctx_dict.items()}
            for method, ctx_dict in long_context.items()
        }

    data = {
        "model": model_name,
        "bits": bits,
        "throughput": throughput,
        "quality": quality,
        "long_context": lc_serializable,
        "batch": batch,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end benchmark: FusedTQ vs TQ vs FP16",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="HuggingFace model name")
    parser.add_argument(
        "--bits", type=int, default=3, choices=[2, 3, 4], help="Quantization bit-width"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Max tokens to generate per prompt"
    )
    parser.add_argument(
        "--quality", action="store_true", help="Also run WikiText-2 perplexity benchmark (slow)"
    )
    parser.add_argument(
        "--max-length", type=int, default=2048, help="Max sequence length for perplexity windows"
    )
    parser.add_argument("--stride", type=int, default=512, help="Stride between perplexity windows")
    parser.add_argument("--no-dejan", action="store_true", help="Skip Dejan.ai comparison")
    parser.add_argument(
        "--json", type=str, default=None, metavar="PATH", help="Save results to JSON file"
    )
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "bfloat16"], help="Model dtype"
    )
    parser.add_argument(
        "--long-context", action="store_true", help="Run long-context decode sweep (4K-32K tokens)"
    )
    parser.add_argument(
        "--context-lengths",
        type=str,
        default="4096,8192,16384,32768",
        help="Comma-separated context lengths for --long-context",
    )
    parser.add_argument(
        "--long-context-gen",
        type=int,
        default=100,
        help="Tokens to generate per context in long-context mode",
    )
    parser.add_argument(
        "--batch-search",
        action="store_true",
        help="Run batch throughput search (find max batch per method)",
    )
    parser.add_argument(
        "--batch-gen", type=int, default=100, help="Tokens to generate per sequence in batch mode"
    )
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=dtype)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    methods = build_methods(model, args.bits, include_dejan=not args.no_dejan)

    print(f"\n{'=' * 78}")
    print(f"BENCHMARKING: {args.model} @ {args.bits}-bit")
    print(f"{'=' * 78}")

    throughput_results = None
    quality_results = None
    long_context_results = None
    batch_results = None

    run_standard = not args.long_context and not args.batch_search

    # --- Standard throughput (default if no other mode selected) ---
    if run_standard:
        print("\n[1/2] Throughput & Memory")
        throughput_results = run_throughput(
            model,
            tokenizer,
            methods,
            args.max_new_tokens,
        )
        print_throughput_table(throughput_results, args.model, args.bits)

        if args.quality:
            print("\n[2/2] Perplexity (WikiText-2)")
            quality_results = run_quality(
                model,
                tokenizer,
                methods,
                args.max_length,
                args.stride,
            )
            print_quality_table(quality_results)
        else:
            print("\n[2/2] Perplexity: skipped (use --quality to enable)")

    # --- Long-context decode ---
    if args.long_context:
        ctx_lengths = [int(x) for x in args.context_lengths.split(",")]
        print(f"\nLONG-CONTEXT DECODE (contexts: {ctx_lengths})")
        long_context_results = run_long_context(
            model,
            tokenizer,
            methods,
            ctx_lengths,
            args.long_context_gen,
        )
        print_long_context_table(
            long_context_results,
            args.model,
            args.bits,
        )

    # --- Batch throughput ---
    if args.batch_search:
        print("\nBATCH THROUGHPUT SEARCH")
        batch_results = run_batch_throughput(
            model,
            tokenizer,
            methods,
            args.batch_gen,
        )
        print_batch_table(batch_results, args.model, args.bits)

    # --- JSON export ---
    if args.json:
        save_json(
            args.model,
            args.bits,
            args.json,
            throughput=throughput_results,
            quality=quality_results,
            long_context=long_context_results,
            batch=batch_results,
        )


if __name__ == "__main__":
    main()
