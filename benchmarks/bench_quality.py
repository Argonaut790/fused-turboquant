"""
Quality benchmark: measure perplexity on WikiText-2 with TurboQuant KV cache.

Compares:
  1. FP16 baseline (no compression)
  2. Ours fused (compressed keys + fused Triton attention)
  3. Ours simulation (roundtrip encode/decode, standard attention)
  4. Dejan.ai (Dense QR rotation, roundtrip quantize/dequantize)

Usage:
    uv run python benchmarks/bench_quality.py --model Qwen/Qwen2.5-0.5B --bits 4
    uv run python benchmarks/bench_quality.py --model Qwen/Qwen3.5-9B --bits 4 --bits 2
    uv run python benchmarks/bench_quality.py --model Qwen/Qwen2.5-0.5B --bits 4 --no-dejan
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
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


def _run_perplexity(label, model, input_ids, windows, cache_factory, max_length, stride):
    """Run a single perplexity measurement and return results dict."""
    print(f"\n{'=' * 60}")
    print(label)
    print("=" * 60)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    r = compute_perplexity(
        model, input_ids, windows,
        cache_factory=cache_factory,
        max_length=max_length, stride=stride,
    )
    elapsed = time.time() - t0
    mem = measure_peak_memory()
    r["time_s"] = elapsed
    r["peak_memory_mb"] = mem
    print(f"  Perplexity: {r['perplexity']:.2f}")
    print(f"  Time: {elapsed:.1f}s, Peak memory: {mem:.0f} MB")
    return r


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
    parser.add_argument("--no-dejan", action="store_true",
                        help="Skip Dejan.ai comparison")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"],
                        help="Model dtype")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=dtype)
    input_ids, windows = load_wikitext(tokenizer, args.max_length, args.stride)

    from fused_turboquant.hf.fused_cache import patch_model, unpatch_model
    from fused_turboquant.hf.simulation_cache import make_simulation_cache

    results = {}

    # --- FP16 baseline ---
    results["FP16 baseline"] = _run_perplexity(
        "FP16 BASELINE (no compression)",
        model, input_ids, windows, None, args.max_length, args.stride,
    )

    for bits in args.bits:
        # --- Ours: fused ---
        cache = patch_model(model, bits=bits)
        results[f"Ours fused TQ{bits}"] = _run_perplexity(
            f"OURS FUSED {bits}-BIT (compressed keys + fused attention)",
            model, input_ids, windows,
            cache_factory=lambda c=cache: c,
            max_length=args.max_length, stride=args.stride,
        )
        unpatch_model(model)

        # --- Ours: simulation ---
        results[f"Ours sim TQ{bits}"] = _run_perplexity(
            f"OURS SIMULATION {bits}-BIT (roundtrip encode/decode)",
            model, input_ids, windows,
            cache_factory=lambda b=bits: make_simulation_cache(bits=b),
            max_length=args.max_length, stride=args.stride,
        )

        # --- Dejan.ai ---
        if not args.no_dejan:
            try:
                sys.path.insert(0, str(DEJAN_DIR))
                from turboquant_kv_cache import make_quantized_cache

                results[f"Dejan.ai TQ{bits}"] = _run_perplexity(
                    f"DEJAN.AI {bits}-BIT (Dense QR, roundtrip)",
                    model, input_ids, windows,
                    cache_factory=lambda b=bits: make_quantized_cache(bits=b),
                    max_length=args.max_length, stride=args.stride,
                )
            except ImportError:
                print("\n  WARNING: Could not import Dejan baseline — skipping.")
                print(f"           Expected at: {DEJAN_DIR}/turboquant_kv_cache.py")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Model: {args.model}")
    hdr = (f"  {'Method':<28s} | {'Perplexity':>12s} | {'Delta':>8s} "
           f"| {'Time':>8s} | {'Peak Mem':>10s}")
    print(hdr)
    print(f"  {'-' * 28}-+-{'-' * 12}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 10}")

    baseline_ppl = results["FP16 baseline"]["perplexity"]
    for key, r in results.items():
        delta = r["perplexity"] - baseline_ppl
        delta_str = "baseline" if key == "FP16 baseline" else f"+{delta:.2f}"
        print(f"  {key:<28s} | {r['perplexity']:>12.2f} | {delta_str:>8s} | "
              f"{r['time_s']:>7.1f}s | {r['peak_memory_mb']:>8.0f} MB")


if __name__ == "__main__":
    main()
