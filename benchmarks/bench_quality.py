"""
Quality benchmark: measure perplexity on WikiText-2 with TurboQuant KV cache.

Compares:
  1. FP16 baseline (no compression)
  2. SimulationCache at 4-bit
  3. SimulationCache at 2-bit

Usage:
    uv run python benchmarks/bench_quality.py --model Qwen/Qwen2.5-0.5B --bits 4
    uv run python benchmarks/bench_quality.py --model Qwen/Qwen3.5-9B --bits 4 --bits 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


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


def load_wikitext(tokenizer, max_length: int = 2048, stride: int = 512):
    """Load WikiText-2 and prepare sliding-window evaluation."""
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
    max_length: int = 2048,
    stride: int = 512,
) -> dict:
    """Compute perplexity over sliding windows."""
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
        shift_logits = logits[:, -(target_len - 1):, :].contiguous()
        shift_labels = ids[:, -(target_len - 1) + 1:end].contiguous()

        if shift_logits.shape[1] != shift_labels.shape[1]:
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
            ppl_so_far = np.exp(np.mean(nlls))
            print(f"    [{i + 1}/{len(windows)}] running ppl={ppl_so_far:.2f}")

    mean_nll = np.mean(nlls)
    ppl = np.exp(mean_nll)
    return {"perplexity": ppl, "mean_nll": mean_nll, "n_windows": len(windows)}


def measure_peak_memory() -> float:
    """Peak GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Quality benchmark: perplexity on WikiText-2")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="HuggingFace model name")
    parser.add_argument("--bits", type=int, nargs="+", default=[4],
                        help="Bit-widths to test (e.g., --bits 4 2)")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Max sequence length per window")
    parser.add_argument("--stride", type=int, default=512,
                        help="Stride between windows")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"],
                        help="Model dtype")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=dtype)
    input_ids, windows = load_wikitext(tokenizer, args.max_length, args.stride)

    results = {}

    # --- FP16 baseline ---
    print("\n" + "=" * 60)
    print("FP16 BASELINE (no compression)")
    print("=" * 60)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.time()
    r = compute_perplexity(model, input_ids, windows,
                           max_length=args.max_length, stride=args.stride)
    elapsed = time.time() - t0
    mem = measure_peak_memory()
    r["time_s"] = elapsed
    r["peak_memory_mb"] = mem
    results["fp16"] = r
    print(f"  Perplexity: {r['perplexity']:.2f}")
    print(f"  Time: {elapsed:.1f}s, Peak memory: {mem:.0f} MB")

    # --- TurboQuant at each bit-width ---
    from fused_turboquant.hf.simulation_cache import make_simulation_cache

    for bits in args.bits:
        print(f"\n{'=' * 60}")
        print(f"TURBOQUANT {bits}-BIT (simulation cache)")
        print("=" * 60)
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        t0 = time.time()
        r = compute_perplexity(
            model, input_ids, windows,
            cache_factory=lambda b=bits: make_simulation_cache(bits=b),
            max_length=args.max_length, stride=args.stride,
        )
        elapsed = time.time() - t0
        mem = measure_peak_memory()
        r["time_s"] = elapsed
        r["peak_memory_mb"] = mem
        results[f"tq{bits}"] = r
        print(f"  Perplexity: {r['perplexity']:.2f}")
        print(f"  Time: {elapsed:.1f}s, Peak memory: {mem:.0f} MB")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  {'Method':<25s} | {'Perplexity':>12s} | {'Delta':>8s} | {'Time':>8s} | {'Peak Mem':>10s}")
    print(f"  {'-' * 25}-+-{'-' * 12}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 10}")

    baseline_ppl = results["fp16"]["perplexity"]
    for key, r in results.items():
        delta = r["perplexity"] - baseline_ppl
        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        if key == "fp16":
            delta_str = "baseline"
        print(f"  {key:<25s} | {r['perplexity']:>12.2f} | {delta_str:>8s} | "
              f"{r['time_s']:>7.1f}s | {r['peak_memory_mb']:>8.0f} MB")


if __name__ == "__main__":
    main()
