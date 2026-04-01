"""
Benchmark: Fused vs Unfused fused-turboquant Pipeline

Measures the real-world impact of fusing the entire encode/decode pipeline
into single Triton kernels vs the multi-kernel PyTorch approach.

Fused encode:  1 kernel  (RHT + norm + quantize + pack)
Unfused encode: 5+ kernels (RHT, torch.norm, division, bucketize, pack_nibbles)

Fused decode:  1 kernel  (unpack + dequant + denorm + inv RHT)
Unfused decode: 4+ kernels (unpack, gather, multiply, inv RHT)
"""

import sys
import time

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))

from fused_turboquant.core.quantizer import TurboQuantMSE


def _fmt_tps(tps: float) -> str:
    if tps >= 1e6:
        return f"{tps / 1e6:.1f}M"
    elif tps >= 1e3:
        return f"{tps / 1e3:.0f}K"
    return f"{tps:.0f}"


def _timer(fn, warmup=10, repeats=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000


def bench_fused_vs_unfused(device: str):
    """Compare fused single-kernel vs unfused multi-kernel encode/decode."""
    print("=" * 72)
    print("FUSED vs UNFUSED PIPELINE (encode + decode)")
    print("=" * 72)
    print()

    DIMS = [64, 128, 256]
    BITS_LIST = [4, 2]
    N_VECTORS = 2048

    for bits in BITS_LIST:
        print(f"--- {bits}-bit quantization, batch={N_VECTORS} ---")
        print(
            f"{'head_dim':>10s} | {'Fused enc':>12s} | {'Unfused enc':>12s} | "
            f"{'Enc speedup':>12s} | {'Fused TPS':>12s} | {'Unfused TPS':>12s} | "
            f"{'Dec speedup':>12s} | {'Quality':>8s}"
        )
        print("-" * 110)

        for dim in DIMS:
            torch.manual_seed(42)
            x = torch.randn(N_VECTORS, dim, device=device)

            tq_fused = TurboQuantMSE(head_dim=dim, bits=bits, device=device)
            assert tq_fused._use_fused_triton, "Fused Triton should be enabled"

            tq_unfused = TurboQuantMSE(head_dim=dim, bits=bits, device=device)
            tq_unfused._use_fused_triton = False

            # --- Encode benchmark ---
            t_fused_enc = _timer(lambda: tq_fused.encode(x))
            t_unfused_enc = _timer(lambda: tq_unfused.encode(x))
            speedup_enc = t_unfused_enc / t_fused_enc

            # --- Decode benchmark ---
            compressed_fused = tq_fused.encode(x)
            compressed_unfused = tq_unfused.encode(x)

            t_fused_dec = _timer(lambda: tq_fused.decode(compressed_fused))
            t_unfused_dec = _timer(lambda: tq_unfused.decode(compressed_unfused))
            speedup_dec = t_unfused_dec / t_fused_dec

            # --- Quality check ---
            x_hat_fused = tq_fused.decode(compressed_fused)
            x_hat_unfused = tq_unfused.decode(compressed_unfused)

            cos_fused = torch.nn.functional.cosine_similarity(x, x_hat_fused, dim=-1).mean().item()
            cos_unfused = (
                torch.nn.functional.cosine_similarity(x, x_hat_unfused, dim=-1).mean().item()
            )

            tps_fused = N_VECTORS / (t_fused_enc / 1000)
            tps_unfused = N_VECTORS / (t_unfused_enc / 1000)

            print(
                f"{dim:>10d} | {t_fused_enc:>10.3f}ms | {t_unfused_enc:>10.3f}ms | "
                f"{speedup_enc:>11.2f}x | {_fmt_tps(tps_fused):>12s} | {_fmt_tps(tps_unfused):>12s} | "
                f"{speedup_dec:>11.2f}x | {cos_fused:>7.4f}"
            )

        print()


def bench_scaling_encode(device: str):
    """Measure encode latency scaling with batch size."""
    print("=" * 72)
    print("FUSED ENCODE SCALING (4-bit, head_dim=256)")
    print("=" * 72)
    print()

    dim = 256
    bits = 4

    tq_fused = TurboQuantMSE(head_dim=dim, bits=bits, device=device)
    tq_unfused = TurboQuantMSE(head_dim=dim, bits=bits, device=device)
    tq_unfused._use_fused_triton = False

    print(
        f"{'Batch':>10s} | {'Fused (ms)':>12s} | {'Unfused (ms)':>12s} | "
        f"{'Speedup':>8s} | {'Fused tok/s':>12s} | {'Unfused tok/s':>12s}"
    )
    print("-" * 80)

    for n in [128, 512, 1024, 2048, 4096, 8192]:
        torch.manual_seed(42)
        x = torch.randn(n, dim, device=device)

        t_fused = _timer(lambda: tq_fused.encode(x))
        t_unfused = _timer(lambda: tq_unfused.encode(x))
        speedup = t_unfused / t_fused

        tok_fused = n / (t_fused / 1000)
        tok_unfused = n / (t_unfused / 1000)

        print(
            f"{n:>10d} | {t_fused:>10.3f}   | {t_unfused:>10.3f}   | "
            f"{speedup:>7.2f}x | {tok_fused:>10.0f}   | {tok_unfused:>10.0f}"
        )

    print()


def bench_dejan_comparison(device: str):
    """Comprehensive comparison: our fused pipeline vs Dejan.ai's multi-kernel."""
    print("=" * 72)
    print("OUR FUSED PIPELINE vs DEJAN.AI's MULTI-KERNEL PIPELINE")
    print("=" * 72)
    print()

    try:
        sys.path.insert(
            0, str(__import__("pathlib").Path(__file__).resolve().parent / "dejan_baseline")
        )
        from turboquant_core import TurboQuantMSE as DejanTQ
    except ImportError:
        print("  Dejan.ai baseline not available, skipping.")
        print()
        return

    # ---- Part 1: Throughput across batch sizes (4-bit, d=256) ----
    print("  Part 1: Encode/Decode TPS across batch sizes (4-bit, head_dim=256)")
    print()
    dim = 256
    bits = 4

    tq_ours = TurboQuantMSE(head_dim=dim, bits=bits, device=device)
    tq_dejan = DejanTQ(d=dim, bits=bits, device=device)

    hdr = (
        f"  {'Batch':>8s} | {'Ours enc':>10s} | {'Dejan enc':>10s} | "
        f"{'Enc TPS ours':>13s} | {'Enc TPS Dejan':>13s} | {'Enc x':>6s} | "
        f"{'Dec TPS ours':>13s} | {'Dec TPS Dejan':>13s} | {'Dec x':>6s}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for n in [128, 512, 2048, 4096, 8192]:
        torch.manual_seed(42)
        x = torch.randn(n, dim, device=device)

        t_ours_enc = _timer(lambda: tq_ours.encode(x))
        t_dejan_enc = _timer(lambda: tq_dejan.quantize(x.float()))

        comp_ours = tq_ours.encode(x)
        dejan_q = tq_dejan.quantize(x.float())
        t_ours_dec = _timer(lambda: tq_ours.decode(comp_ours))
        t_dejan_dec = _timer(lambda: tq_dejan.dequantize(dejan_q))

        tps_ours_enc = n / (t_ours_enc / 1000)
        tps_dejan_enc = n / (t_dejan_enc / 1000)
        tps_ours_dec = n / (t_ours_dec / 1000)
        tps_dejan_dec = n / (t_dejan_dec / 1000)
        enc_x = t_dejan_enc / t_ours_enc
        dec_x = t_dejan_dec / t_ours_dec

        print(
            f"  {n:>8d} | {t_ours_enc:>8.3f}ms | {t_dejan_enc:>8.3f}ms | "
            f"{_fmt_tps(tps_ours_enc):>13s} | {_fmt_tps(tps_dejan_enc):>13s} | {enc_x:>5.2f}x | "
            f"{_fmt_tps(tps_ours_dec):>13s} | {_fmt_tps(tps_dejan_dec):>13s} | {dec_x:>5.2f}x"
        )

    print()

    # ---- Part 2: Across head_dims (4-bit, batch=2048) ----
    print("  Part 2: TPS across head_dims (4-bit, batch=2048)")
    print()
    n = 2048

    hdr2 = (
        f"  {'head_dim':>8s} | {'Ours enc TPS':>13s} | {'Dejan enc TPS':>13s} | {'Enc x':>6s} | "
        f"{'Ours dec TPS':>13s} | {'Dejan dec TPS':>13s} | {'Dec x':>6s} | "
        f"{'Ours cos':>9s} | {'Dejan cos':>9s}"
    )
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))

    for dim in [64, 128, 256]:
        torch.manual_seed(42)
        x = torch.randn(n, dim, device=device)

        tq_ours_d = TurboQuantMSE(head_dim=dim, bits=bits, device=device)
        tq_dejan_d = DejanTQ(d=dim, bits=bits, device=device)

        t_ours_enc = _timer(lambda: tq_ours_d.encode(x))
        t_dejan_enc = _timer(lambda: tq_dejan_d.quantize(x.float()))

        comp_ours = tq_ours_d.encode(x)
        dejan_q = tq_dejan_d.quantize(x.float())
        t_ours_dec = _timer(lambda: tq_ours_d.decode(comp_ours))
        t_dejan_dec = _timer(lambda: tq_dejan_d.dequantize(dejan_q))

        x_hat_ours = tq_ours_d.decode(comp_ours)
        x_hat_dejan = tq_dejan_d.dequantize(dejan_q).float()

        cos_ours = torch.nn.functional.cosine_similarity(x, x_hat_ours, dim=-1).mean().item()
        cos_dejan = torch.nn.functional.cosine_similarity(x, x_hat_dejan, dim=-1).mean().item()

        tps_ours_enc = n / (t_ours_enc / 1000)
        tps_dejan_enc = n / (t_dejan_enc / 1000)
        tps_ours_dec = n / (t_ours_dec / 1000)
        tps_dejan_dec = n / (t_dejan_dec / 1000)
        enc_x = t_dejan_enc / t_ours_enc
        dec_x = t_dejan_dec / t_ours_dec

        print(
            f"  {dim:>8d} | {_fmt_tps(tps_ours_enc):>13s} | {_fmt_tps(tps_dejan_enc):>13s} | {enc_x:>5.2f}x | "
            f"{_fmt_tps(tps_ours_dec):>13s} | {_fmt_tps(tps_dejan_dec):>13s} | {dec_x:>5.2f}x | "
            f"{cos_ours:>9.4f} | {cos_dejan:>9.4f}"
        )

    print()

    # ---- Part 3: Across bit-widths (d=256, batch=2048) ----
    print("  Part 3: TPS across bit-widths (head_dim=256, batch=2048)")
    print()
    dim = 256
    n = 2048

    hdr3 = (
        f"  {'Bits':>5s} | {'Ours enc TPS':>13s} | {'Dejan enc TPS':>13s} | {'Enc x':>6s} | "
        f"{'Ours dec TPS':>13s} | {'Dejan dec TPS':>13s} | {'Dec x':>6s} | "
        f"{'Ours cos':>9s} | {'Dejan cos':>9s}"
    )
    print(hdr3)
    print("  " + "-" * (len(hdr3) - 2))

    for bits in [4, 3, 2]:
        torch.manual_seed(42)
        x = torch.randn(n, dim, device=device)

        tq_ours_b = TurboQuantMSE(head_dim=dim, bits=bits, device=device)
        tq_dejan_b = DejanTQ(d=dim, bits=bits, device=device)

        t_ours_enc = _timer(lambda: tq_ours_b.encode(x))
        t_dejan_enc = _timer(lambda: tq_dejan_b.quantize(x.float()))

        comp_ours = tq_ours_b.encode(x)
        dejan_q = tq_dejan_b.quantize(x.float())
        t_ours_dec = _timer(lambda: tq_ours_b.decode(comp_ours))
        t_dejan_dec = _timer(lambda: tq_dejan_b.dequantize(dejan_q))

        x_hat_ours = tq_ours_b.decode(comp_ours)
        x_hat_dejan = tq_dejan_b.dequantize(dejan_q).float()

        cos_ours = torch.nn.functional.cosine_similarity(x, x_hat_ours, dim=-1).mean().item()
        cos_dejan = torch.nn.functional.cosine_similarity(x, x_hat_dejan, dim=-1).mean().item()

        tps_ours_enc = n / (t_ours_enc / 1000)
        tps_dejan_enc = n / (t_dejan_enc / 1000)
        tps_ours_dec = n / (t_ours_dec / 1000)
        tps_dejan_dec = n / (t_dejan_dec / 1000)
        enc_x = t_dejan_enc / t_ours_enc
        dec_x = t_dejan_dec / t_ours_dec

        ours_bytes = comp_ours.indices.numel() + comp_ours.norms.numel() * 4
        dejan_bytes = dejan_q["idx"].numel() + dejan_q["norms"].numel() * 2

        print(
            f"  {bits:>5d} | {_fmt_tps(tps_ours_enc):>13s} | {_fmt_tps(tps_dejan_enc):>13s} | {enc_x:>5.2f}x | "
            f"{_fmt_tps(tps_ours_dec):>13s} | {_fmt_tps(tps_dejan_dec):>13s} | {dec_x:>5.2f}x | "
            f"{cos_ours:>9.4f} | {cos_dejan:>9.4f}"
        )

    print()

    # ---- Part 4: Memory per token comparison ----
    print("  Part 4: Memory per token (head_dim=256)")
    print()
    dim = 256
    print(
        f"  {'Bits':>5s} | {'Ours (bytes/token)':>18s} | {'Dejan (bytes/token)':>19s} | {'Savings':>8s} | {'Reason':>35s}"
    )
    print(f"  {'-' * 5}-+-{'-' * 18}-+-{'-' * 19}-+-{'-' * 8}-+-{'-' * 35}")
    for bits in [4, 3, 2]:
        if bits == 4:
            ours_per_tok = dim // 2 + 4  # packed nibbles + fp32 norm
            dejan_per_tok = dim * 1 + 2  # 1 byte per index + fp16 norm
            reason = "nibble packing: 2 indices/byte"
        elif bits == 3:
            ours_per_tok = dim * 1 + 4  # no packing for 3-bit + fp32 norm
            dejan_per_tok = dim * 1 + 2  # 1 byte per index + fp16 norm
            reason = "same: no sub-byte packing at 3-bit"
        else:
            ours_per_tok = dim // 4 + 4  # 4 indices/byte + fp32 norm
            dejan_per_tok = dim * 1 + 2  # 1 byte per index + fp16 norm
            reason = "2-bit packing: 4 indices/byte"

        savings = dejan_per_tok / ours_per_tok
        print(
            f"  {bits:>5d} | {ours_per_tok:>18d} | {dejan_per_tok:>19d} | {savings:>7.2f}x | {reason:>35s}"
        )

    print()

    # ---- Part 5: Architectural summary ----
    print("  Part 5: Architectural Differences")
    print()
    print(f"  {'Feature':>30s} | {'fused-turboquant (ours)':>25s} | {'Dejan.ai':>25s}")
    print(f"  {'-' * 30}-+-{'-' * 25}-+-{'-' * 25}")
    print(f"  {'Pipeline':>30s} | {'Fused 1-kernel Triton':>25s} | {'Multi-kernel PyTorch':>25s}")
    print(f"  {'Rotation':>30s} | {'RHT O(d log d)':>25s} | {'Dense QR O(d^2)':>25s}")
    print(
        f"  {'Rotation storage/layer':>30s} | {'1 KB (d signs)':>25s} | {'256 KB (d x d matrix)':>25s}"
    )
    print(f"  {'Kernel launches (encode)':>30s} | {'1':>25s} | {'3+':>25s}")
    print(f"  {'Kernel launches (decode)':>30s} | {'1':>25s} | {'3+':>25s}")
    print(
        f"  {'Nibble packing (4-bit)':>30s} | {'Yes (2 per byte)':>25s} | {'No (1 per byte)':>25s}"
    )
    print(f"  {'2-bit packing':>30s} | {'Yes (4 per byte)':>25s} | {'No (1 per byte)':>25s}")
    print(f"  {'Value (V) compression':>30s} | {'Yes':>25s} | {'No (K only)':>25s}")
    print(f"  {'Norm precision':>30s} | {'fp32':>25s} | {'fp16':>25s}")
    print(f"  {'vLLM plugin':>30s} | {'Yes':>25s} | {'No':>25s}")
    print()


def bench_four_way(device: str):
    """4-way comparison: FP16 baseline vs our fused vs our unfused vs Dejan.ai.

    Measures both kernel-level TPS (encode/decode in isolation) AND end-to-end
    inference TPS which simulates a real decode step:
        For each token: encode new KV -> decode all cached KVs -> Q @ K^T attention
    """
    print("=" * 72)
    print("4-WAY COMPARISON (4-bit, head_dim=256)")
    print("=" * 72)
    print()

    dim = 256
    bits = 4
    n_heads_kv = 4
    scale = 1.0 / (dim**0.5)
    fp16_bytes_per_tok = dim * 2

    sys.path.insert(
        0, str(__import__("pathlib").Path(__file__).resolve().parent / "dejan_baseline")
    )
    try:
        from turboquant_core import TurboQuantMSE as DejanTQ

        has_dejan = True
    except ImportError:
        has_dejan = False

    # ---- Part 1: Kernel-level encode/decode TPS (batch=2048) ----
    print("  Part 1: Kernel-level encode/decode TPS (batch=2048)")
    print()

    n = 2048
    torch.manual_seed(42)
    x = torch.randn(n, dim, device=device)

    tq_fused = TurboQuantMSE(head_dim=dim, bits=bits, device=device)
    assert tq_fused._use_fused_triton
    t_fused_enc = _timer(lambda: tq_fused.encode(x))
    comp_fused = tq_fused.encode(x)
    t_fused_dec = _timer(lambda: tq_fused.decode(comp_fused))
    x_hat_fused = tq_fused.decode(comp_fused)
    cos_fused = torch.nn.functional.cosine_similarity(x, x_hat_fused, dim=-1).mean().item()
    fused_bytes = comp_fused.indices.numel() / n + comp_fused.norms.numel() / n * 4

    tq_unfused = TurboQuantMSE(head_dim=dim, bits=bits, device=device)
    tq_unfused._use_fused_triton = False
    t_unfused_enc = _timer(lambda: tq_unfused.encode(x))
    comp_unfused = tq_unfused.encode(x)
    t_unfused_dec = _timer(lambda: tq_unfused.decode(comp_unfused))
    x_hat_unfused = tq_unfused.decode(comp_unfused)
    cos_unfused = torch.nn.functional.cosine_similarity(x, x_hat_unfused, dim=-1).mean().item()
    unfused_bytes = comp_unfused.indices.numel() / n + comp_unfused.norms.numel() / n * 4

    dejan_bytes = None
    cos_dejan = None
    t_dejan_enc = t_dejan_dec = None
    if has_dejan:
        tq_dejan = DejanTQ(d=dim, bits=bits, device=device)
        t_dejan_enc = _timer(lambda: tq_dejan.quantize(x.float()))
        dejan_q = tq_dejan.quantize(x.float())
        t_dejan_dec = _timer(lambda: tq_dejan.dequantize(dejan_q))
        x_hat_dejan = tq_dejan.dequantize(dejan_q).float()
        cos_dejan = torch.nn.functional.cosine_similarity(x, x_hat_dejan, dim=-1).mean().item()
        dejan_bytes = dejan_q["idx"].numel() / n + dejan_q["norms"].numel() / n * 2

    W = 18

    def _header():
        print(
            f"  {'':>22s} | {'FP16 (no quant)':>{W}s} | {'Ours (fused)':>{W}s} | {'Ours (unfused)':>{W}s}",
            end="",
        )
        if has_dejan:
            print(f" | {'Dejan.ai':>{W}s}", end="")
        print()
        print(f"  {'-' * 22}-+-{'-' * W}-+-{'-' * W}-+-{'-' * W}", end="")
        if has_dejan:
            print(f"-+-{'-' * W}", end="")
        print()

    def _row(
        label,
        fp16_val,
        fused_val,
        unfused_val,
        dejan_val=None,
        fmt="s",
        best="max",
        include_fp16_in_best=False,
    ):
        vals = [fused_val, unfused_val]
        if dejan_val is not None:
            vals.append(dejan_val)
        if include_fp16_in_best and fp16_val is not None:
            vals.append(fp16_val)
        numeric_vals = [v for v in vals if v is not None]
        if best == "max" and numeric_vals:
            best_val = max(numeric_vals)
        elif best == "min" and numeric_vals:
            best_val = min(numeric_vals)
        else:
            best_val = None

        def _f(v, is_fp16=False):
            if v is None:
                return f"{'--':>{W}s}"
            if fmt == "tps":
                s = _fmt_tps(v)
            elif fmt == "cos":
                s = f"{v:.4f}"
            elif fmt == "bytes":
                s = f"{v:.0f}"
            elif fmt == "ratio":
                s = f"{v:.1f}x"
            elif fmt == "ms":
                s = f"{v:.3f}ms"
            else:
                s = str(v)
            skip_bold = is_fp16 and not include_fp16_in_best
            if not skip_bold and v == best_val:
                s = f"*{s}*"
            return f"{s:>{W}s}"

        print(
            f"  {label:>22s} | {_f(fp16_val, True)} | {_f(fused_val)} | {_f(unfused_val)}", end=""
        )
        if has_dejan:
            print(f" | {_f(dejan_val)}", end="")
        print()

    _header()
    tps_fused_enc = n / (t_fused_enc / 1000)
    tps_unfused_enc = n / (t_unfused_enc / 1000)
    tps_fused_dec = n / (t_fused_dec / 1000)
    tps_unfused_dec = n / (t_unfused_dec / 1000)
    tps_dejan_enc = n / (t_dejan_enc / 1000) if has_dejan else None
    tps_dejan_dec = n / (t_dejan_dec / 1000) if has_dejan else None

    _row("Encode TPS", None, tps_fused_enc, tps_unfused_enc, tps_dejan_enc, fmt="tps", best="max")
    _row("Decode TPS", None, tps_fused_dec, tps_unfused_dec, tps_dejan_dec, fmt="tps", best="max")
    _row("Quality (cosine sim)", 1.0, cos_fused, cos_unfused, cos_dejan, fmt="cos", best="max")
    _row(
        "Bytes/token",
        fp16_bytes_per_tok,
        fused_bytes,
        unfused_bytes,
        dejan_bytes,
        fmt="bytes",
        best="min",
    )
    _row(
        "Compression",
        1.0,
        fp16_bytes_per_tok / fused_bytes,
        fp16_bytes_per_tok / unfused_bytes,
        fp16_bytes_per_tok / dejan_bytes if dejan_bytes else None,
        fmt="ratio",
        best="max",
    )
    print()

    # ---- Part 2: End-to-end inference TPS (simulated decode step) ----
    # Simulates a real autoregressive decode iteration per head:
    #   1. Encode new K,V into compressed cache
    #   2. Decode all cached K,V from compressed cache
    #   3. Compute Q @ K^T attention scores (standard matmul)
    # For FP16: no encode/decode, just store and matmul directly.
    print("  Part 2: End-to-end inference TPS (single-token decode step per head)")
    print("  = encode new KV + decode all cached KV + Q @ K^T attention")
    print()

    for ctx_len in [512, 2048, 8192]:
        print(f"  --- Context length: {ctx_len} tokens ---")
        print()
        _header()

        torch.manual_seed(42)
        q = torch.randn(1, dim, device=device, dtype=torch.float32)
        new_kv = torch.randn(1, dim, device=device, dtype=torch.float32)

        # Pre-fill caches
        cached_fp16 = torch.randn(ctx_len, dim, device=device, dtype=torch.float16)

        tq_f = TurboQuantMSE(head_dim=dim, bits=bits, device=device)
        assert tq_f._use_fused_triton
        cached_comp_f = tq_f.encode(cached_fp16.float())

        tq_u = TurboQuantMSE(head_dim=dim, bits=bits, device=device)
        tq_u._use_fused_triton = False
        cached_comp_u = tq_u.encode(cached_fp16.float())

        if has_dejan:
            tq_d = DejanTQ(d=dim, bits=bits, device=device)
            cached_comp_d = tq_d.quantize(cached_fp16.float())

        def _e2e_fp16():
            all_k = torch.cat([cached_fp16, new_kv.half()], dim=0)
            scores = (q.half() @ all_k.T) * scale
            return scores

        def _e2e_fused():
            _ = tq_f.encode(new_kv)
            keys = tq_f.decode(cached_comp_f)
            scores = (q @ keys.T) * scale
            return scores

        def _e2e_unfused():
            _ = tq_u.encode(new_kv)
            keys = tq_u.decode(cached_comp_u)
            scores = (q @ keys.T) * scale
            return scores

        def _e2e_dejan():
            _ = tq_d.quantize(new_kv.float())
            keys = tq_d.dequantize(cached_comp_d).float()
            scores = (q @ keys.T) * scale
            return scores

        t_fp16 = _timer(_e2e_fp16, warmup=20, repeats=200)
        t_e2e_fused = _timer(_e2e_fused, warmup=20, repeats=200)
        t_e2e_unfused = _timer(_e2e_unfused, warmup=20, repeats=200)
        t_e2e_dejan = _timer(_e2e_dejan, warmup=20, repeats=200) if has_dejan else None

        # TPS = 1 token / time_per_step (since each step produces 1 output token)
        # For multi-head: multiply by n_heads since all heads run in parallel in practice
        tps_fp16 = n_heads_kv / (t_fp16 / 1000)
        tps_e2e_fused = n_heads_kv / (t_e2e_fused / 1000)
        tps_e2e_unfused = n_heads_kv / (t_e2e_unfused / 1000)
        tps_e2e_dejan = n_heads_kv / (t_e2e_dejan / 1000) if t_e2e_dejan else None

        _row(
            "Inference TPS",
            tps_fp16,
            tps_e2e_fused,
            tps_e2e_unfused,
            tps_e2e_dejan,
            fmt="tps",
            best="max",
            include_fp16_in_best=True,
        )
        _row(
            "Latency/step",
            t_fp16,
            t_e2e_fused,
            t_e2e_unfused,
            t_e2e_dejan,
            fmt="ms",
            best="min",
            include_fp16_in_best=True,
        )

        # Memory for full cache at this context length
        fp16_mem = ctx_len * dim * 2
        fused_mem = ctx_len * fused_bytes
        unfused_mem = ctx_len * unfused_bytes
        dejan_mem = ctx_len * dejan_bytes if dejan_bytes else None

        def _fmt_mem(b):
            if b is None:
                return None
            if b >= 1024 * 1024:
                return f"{b / 1024 / 1024:.1f} MB"
            return f"{b / 1024:.1f} KB"

        print(
            f"  {'KV cache memory':>22s} | {_fmt_mem(fp16_mem):>{W}s} | {_fmt_mem(fused_mem):>{W}s} | {_fmt_mem(unfused_mem):>{W}s}",
            end="",
        )
        if has_dejan:
            print(f" | {_fmt_mem(dejan_mem):>{W}s}", end="")
        print()
        print()

    # ---- Qualitative row ----
    print("  Part 3: Architecture")
    print()
    _header()

    qual_rows = [
        ("Pipeline", "--", "1 Triton kernel", "5+ PyTorch ops", "3+ PyTorch ops"),
        ("Rotation", "--", "RHT O(d log d)", "RHT O(d log d)", "Dense QR O(d^2)"),
        ("Rotation storage", "--", "1 KB", "1 KB", "256 KB"),
        ("Packing", "--", "2 idx/byte", "2 idx/byte", "1 idx/byte"),
        ("V compression", "--", "Yes", "Yes", "No (K only)"),
    ]
    for label, fp16, fused, unfused, dejan in qual_rows:
        print(f"  {label:>22s} | {fp16:>{W}s} | {fused:>{W}s} | {unfused:>{W}s}", end="")
        if has_dejan:
            print(f" | {dejan:>{W}s}", end="")
        print()
    print()
    print("  * = best among all methods (including FP16 where applicable)")
    print()


def bench_kernel_launch_overhead(device: str):
    """Demonstrate kernel launch overhead: 1 fused launch vs N separate."""
    print("=" * 72)
    print("KERNEL LAUNCH OVERHEAD ANALYSIS")
    print("=" * 72)
    print()

    dim = 256
    bits = 4

    tq = TurboQuantMSE(head_dim=dim, bits=bits, device=device)

    torch.manual_seed(42)
    x_small = torch.randn(1, dim, device=device)
    x_large = torch.randn(8192, dim, device=device)

    tq._use_fused_triton = True
    t_fused_small = _timer(lambda: tq.encode(x_small), warmup=20, repeats=200)
    t_fused_large = _timer(lambda: tq.encode(x_large), warmup=10, repeats=100)

    tq._use_fused_triton = False
    t_unfused_small = _timer(lambda: tq.encode(x_small), warmup=20, repeats=200)
    t_unfused_large = _timer(lambda: tq.encode(x_large), warmup=10, repeats=100)

    tq._use_fused_triton = True

    print(
        f"  Single vector  (batch=1):     fused={t_fused_small:.3f}ms  unfused={t_unfused_small:.3f}ms  "
        f"speedup={t_unfused_small / t_fused_small:.2f}x"
    )
    print(
        f"  Large batch (batch=8192):    fused={t_fused_large:.3f}ms  unfused={t_unfused_large:.3f}ms  "
        f"speedup={t_unfused_large / t_fused_large:.2f}x"
    )
    print()
    print("  At batch=1, kernel launch overhead dominates.")
    print("  The fused kernel avoids ~4 extra launches (~5us each = ~20us overhead).")
    print()


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark requires a GPU.")
        return

    device = "cuda"
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name} ({gpu.total_memory / 1024**3:.1f} GB)")
    print(f"PyTorch: {torch.__version__}")
    try:
        import triton

        print(f"Triton: {triton.__version__}")
    except ImportError:
        print("Triton: NOT AVAILABLE")
        return
    print()

    bench_four_way(device)
    bench_fused_vs_unfused(device)
    bench_scaling_encode(device)
    bench_dejan_comparison(device)
    bench_kernel_launch_overhead(device)

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print()
    print("  The fused pipeline combines RHT + norm + quantize + pack into")
    print("  a single Triton kernel launch, eliminating 4+ separate GPU kernel")
    print("  launches and their associated HBM round-trips.")
    print()
    print("  This fusion is uniquely enabled by RHT: the O(d) sign vector fits")
    print("  in SRAM, allowing the rotation to be fused with post-rotation ops.")
    print("  Dense QR's O(d^2) matrix cannot be fused the same way.")
    print()


if __name__ == "__main__":
    main()
