#!/usr/bin/env python3
"""
TurboQuant demo on Gemma 3 4B IT

Requirements:
    pip install torch transformers accelerate

Usage:
    python run_demo.py --core-test-only          # algorithm only, no GPU
    python run_demo.py                            # full demo
    python run_demo.py --prompt "Your text here"  # custom prompt
    python run_demo.py --long-context             # needle-in-haystack
    python run_demo.py --skip-baseline --bits 4   # skip fp16 for speed
"""

import argparse
import time
import torch
from turboquant_core import self_test as core_self_test


DEFAULT_PROMPTS = [
    "Explain the difference between a compiler and an interpreter in three sentences.",
    "Write a short Python function that checks if a string is a palindrome.",
    "What are the main causes of the French Revolution? Be concise.",
]


def load_model(model_id: str):
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    print(f"Loading {model_id} ...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"Model loaded on {model.device}\n")
    return model, processor


def generate(model, processor, prompt: str, bits: int | None = None,
             max_new_tokens: int = 200, fused: bool = False) -> dict:
    """Generate text. If bits is set, uses TurboQuant KV compression.
    If fused=True, uses Triton kernel (requires bits to be set)."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated()
    t0 = time.perf_counter()

    with torch.inference_mode():
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        if fused and bits is not None:
            from turboquant_fused import FusedTurboQuantRunner
            runner = FusedTurboQuantRunner(model, processor, bits=bits)
            text = runner.generate(prompt, max_new_tokens=max_new_tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            peak_mb = (torch.cuda.max_memory_allocated() - vram_before) / 1e6 if torch.cuda.is_available() else 0
            n_tokens = len(processor.tokenizer.encode(text))
            return {"text": text, "time_s": elapsed,
                    "tokens": n_tokens, "tok_per_s": n_tokens / elapsed,
                    "peak_mb": peak_mb}
        elif bits is not None:
            from turboquant_kv_cache import TurboQuantWrapper
            wrapper = TurboQuantWrapper(model, bits=bits)
            outputs = wrapper.generate(**gen_kwargs)
        else:
            outputs = model.generate(**gen_kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mb = (torch.cuda.max_memory_allocated() - vram_before) / 1e6 if torch.cuda.is_available() else 0

    gen_ids = outputs[0][input_len:]
    text = processor.decode(gen_ids, skip_special_tokens=True)
    return {
        "text": text,
        "time_s": elapsed,
        "tokens": len(gen_ids),
        "tok_per_s": len(gen_ids) / elapsed,
        "peak_mb": peak_mb,
    }


def needle_test(model, processor, bits: int | None, haystack_tokens: int = 2048, fused: bool = False) -> dict:
    needle = "The secret password for project Orion is 'blue-giraffe-42'."
    question = "What is the secret password for project Orion?"
    filler = (
        "The annual report covers financial performance across all divisions. "
        "Revenue grew steadily in Q3 driven by international expansion. "
        "Operating costs remained within budget targets for the fiscal year. "
    )
    # Rough estimate of filler tokens
    n_repeats = max(1, haystack_tokens // 80)
    prompt = (
        f"{filler * n_repeats}\n{needle}\n{filler * n_repeats}\n\n"
        f"Question: {question}"
    )
    result = generate(model, processor, prompt, bits=bits, max_new_tokens=50, fused=fused)
    found = "blue-giraffe-42" in result["text"].lower()
    return {"found": found, "answer": result["text"].strip(), "time_s": result["time_s"]}


def main():
    parser = argparse.ArgumentParser(description="TurboQuant KV cache demo")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 3])
    parser.add_argument("--long-context", action="store_true")
    parser.add_argument("--haystack-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--fused", action="store_true",
                        help="Use Triton fused attention kernel (requires triton)")
    parser.add_argument("--core-test-only", action="store_true")
    parser.add_argument("--kernel-test", action="store_true",
                        help="Run Triton kernel correctness test and benchmark")
    args = parser.parse_args()

    # --- Kernel test ---
    if args.kernel_test:
        from triton_attention import test_fused_kernel, benchmark_fused_vs_standard
        print("=" * 60)
        print("Triton fused attention kernel tests")
        print("=" * 60)
        from turboquant_core import TurboQuantMSE  # noqa: ensure codebook is built
        ok = test_fused_kernel()
        if ok:
            print("Benchmarking fused vs standard attention scores:")
            benchmark_fused_vs_standard()
        return

    # --- Core self-test ---
    print("=" * 70)
    print("Stage 0: TurboQuant core algorithm self-test")
    print("=" * 70)
    core_self_test()
    if args.core_test_only:
        return

    # --- Load model ---
    model, processor = load_model(args.model)
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    configs: list[tuple[str, int | None, bool]] = []  # (label, bits, fused)
    if not args.skip_baseline:
        configs.append(("fp16 baseline", None, False))
    for b in args.bits:
        configs.append((f"TurboQuant {b}-bit", b, False))
        if args.fused:
            configs.append((f"TurboQuant {b}-bit FUSED", b, True))

    # --- Generation comparison ---
    for prompt in prompts:
        print("=" * 70)
        print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print("=" * 70)

        for label, bits, fused in configs:
            try:
                result = generate(model, processor, prompt, bits=bits,
                                  max_new_tokens=args.max_new_tokens, fused=fused)
                print(f"\n--- {label} ---")
                print(f"  Tokens: {result['tokens']}  Time: {result['time_s']:.2f}s  "
                      f"({result['tok_per_s']:.1f} tok/s)  "
                      f"VRAM delta: {result['peak_mb']:.0f} MB")
                print(f"  Output: {result['text'][:500]}")
            except Exception as e:
                import traceback
                print(f"\n--- {label} --- ERROR:")
                traceback.print_exc()
        print()

    # --- Needle-in-a-haystack ---
    if args.long_context:
        print("=" * 70)
        print(f"Needle-in-a-haystack (~{args.haystack_tokens} tokens)")
        print("=" * 70)
        for label, bits, fused in configs:
            try:
                r = needle_test(model, processor, bits=bits,
                                haystack_tokens=args.haystack_tokens, fused=fused)
                status = "FOUND" if r["found"] else "MISSED"
                print(f"  {label:25s}  [{status}]  {r['time_s']:.1f}s")
                print(f"    Answer: {r['answer'][:150]}")
            except Exception as e:
                import traceback
                print(f"  {label:25s}  ERROR:")
                traceback.print_exc()
        print()


if __name__ == "__main__":
    main()
