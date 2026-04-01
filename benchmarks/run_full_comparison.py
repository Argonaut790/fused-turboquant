"""
Comprehensive fused-turboquant benchmark suite.

Compares across multiple dimensions:
  1. QUALITY:      RHT (ours, Triton) vs Dense QR (Dejan.ai style) — fairness
  2. ATTENTION:    Attention score accuracy (Q.K^T with quantized K)
  3. ROTATION:     Triton fused RHT vs Dense QR matmul — GPU latency
  4. MEMORY:       KV cache sizes for Qwen3.5-9B at different context lengths
  5. STORAGE:      Rotation parameter overhead (RHT vs Dense QR)
  6. THROUGHPUT:   Encode/decode tokens per second
  7. DEJAN:        Direct comparison with Dejan.ai's implementation

Usage:
    uv run python benchmarks/run_full_comparison.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "dejan_baseline"))

import torch

from fused_turboquant.core.hadamard import DenseQRRotation, RHTRotation
from fused_turboquant.core.lloyd_max import LloydMaxQuantizer
from fused_turboquant.core.packing import pack_2bit, pack_nibbles, unpack_2bit, unpack_nibbles
from fused_turboquant.core.quantizer import CompressedTensor, TurboQuantMSE
from fused_turboquant.kernels.triton_rht import is_triton_available

QWEN35_FULL_ATTN_LAYERS = 8
QWEN35_NUM_KV_HEADS = 4
QWEN35_HEAD_DIM = 256
QWEN35_NUM_Q_HEADS = 16


class DenseQRTurboQuantMSE:
    """TurboQuant_MSE with Dense QR rotation (what Dejan.ai and all others use)."""

    def __init__(self, head_dim: int, bits: int = 4, seed: int = 42, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device
        self.rotation = DenseQRRotation(head_dim, seed=seed, device=device)
        self.quantizer = LloydMaxQuantizer(head_dim, bits=bits, device=device)

    def encode(self, x: torch.Tensor) -> CompressedTensor:
        x = x.float()
        rotated = self.rotation(x)
        norms = torch.norm(rotated, dim=-1, keepdim=True)
        normalized = rotated / (norms + 1e-8)
        indices = self.quantizer.quantize(normalized)
        if self.bits == 4:
            packed = pack_nibbles(indices)
        elif self.bits == 2:
            packed = pack_2bit(indices)
        else:
            packed = indices
        return CompressedTensor(
            indices=packed,
            norms=norms.squeeze(-1).float(),
            original_dim=self.head_dim,
            bits=self.bits,
        )

    def decode(self, compressed: CompressedTensor) -> torch.Tensor:
        if compressed.bits == 4:
            indices = unpack_nibbles(compressed.indices, compressed.original_dim)
        elif compressed.bits == 2:
            indices = unpack_2bit(compressed.indices, compressed.original_dim)
        else:
            indices = compressed.indices
        reconstructed = self.quantizer.dequantize(indices)
        reconstructed = reconstructed * compressed.norms.unsqueeze(-1)
        return self.rotation.inverse(reconstructed)

    def roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def _timer(fn, warmup=3, repeats=20, use_cuda=False):
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if use_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def fmt_bytes(b: float) -> str:
    if b < 1024:
        return f"{b:.0f} B"
    elif b < 1024**2:
        return f"{b / 1024:.1f} KB"
    elif b < 1024**3:
        return f"{b / 1024**2:.1f} MB"
    else:
        return f"{b / 1024**3:.2f} GB"


# ---- BENCHMARK 1: Quality Fairness ----
def bench_quality_fairness(device: str):
    print("\n" + "=" * 90)
    print("  BENCHMARK 1: QUANTIZATION QUALITY -- RHT (Triton) vs Dense QR (Dejan.ai style)")
    print("  Both use identical Lloyd-Max codebook; only the rotation method differs.")
    print("=" * 90)

    dim = QWEN35_HEAD_DIM
    n = 4096
    x = torch.randn(n, dim, device=device, dtype=torch.float32)

    rht_q = TurboQuantMSE(head_dim=dim, bits=4, device=device)
    print(f"\n  {n} random vectors, dim={dim}")
    print(f"  RHTRotation backend: {rht_q.rotation.extra_repr()}")
    print()

    header = (
        f"  {'Bits':>4} | {'Method':<24} | {'Cosine Sim':>11} | {'MSE':>11} | "
        f"{'IP Corr':>9} | {'Compress':>9}"
    )
    print(header)
    print(f"  {'----':>4}-+-{'----':<24}-+-{'----':>11}-+-{'----':>11}-+-{'----':>9}-+-{'----':>9}")

    for bits in [4, 3, 2]:
        for name, cls in [
            ("RHT Triton (ours)", TurboQuantMSE),
            ("Dense QR matmul (others)", DenseQRTurboQuantMSE),
        ]:
            tq = cls(head_dim=dim, bits=bits, device=device)
            compressed = tq.encode(x)
            x_hat = tq.decode(compressed)

            mse = torch.mean((x - x_hat) ** 2).item()
            x_n = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
            xh_n = x_hat / (torch.norm(x_hat, dim=-1, keepdim=True) + 1e-8)
            cos = torch.mean(torch.sum(x_n * xh_n, dim=-1)).item()
            ip_a = torch.sum(x[: n // 2] * x[n // 2 :], dim=-1)
            ip_b = torch.sum(x_hat[: n // 2] * x_hat[n // 2 :], dim=-1)
            ip_corr = torch.corrcoef(torch.stack([ip_a, ip_b]))[0, 1].item()
            ratio = compressed.compression_ratio

            print(
                f"  {bits:>4} | {name:<24} | {cos:>11.6f} | {mse:>11.6f} | "
                f"{ip_corr:>9.6f} | {ratio:>8.1f}x"
            )
        print(
            f"  {'----':>4}-+-{'----':<24}-+-{'----':>11}-+-{'----':>11}-+-{'----':>9}-+-{'----':>9}"
        )


# ---- BENCHMARK 2: Attention Accuracy ----
def bench_attention_accuracy(device: str):
    print("\n" + "=" * 90)
    print("  BENCHMARK 2: ATTENTION SCORE ACCURACY (Q . K^T)")
    print("  FP32 baseline vs quantized keys at each bit-width.")
    print("=" * 90)

    dim = QWEN35_HEAD_DIM
    num_queries = 16
    seq_len = 512
    queries = torch.randn(num_queries, dim, device=device, dtype=torch.float32)
    keys = torch.randn(seq_len, dim, device=device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(dim)
    scores_fp32 = (queries @ keys.T) * scale

    print(f"\n  {num_queries} queries x {seq_len} keys, dim={dim}")
    print()
    print(
        f"  {'Bits':>4} | {'Score Cosine':>13} | {'Score MAE':>11} | {'Max Err':>9} | {'Softmax KL':>11}"
    )
    print(f"  {'----':>4}-+-{'----':>13}-+-{'----':>11}-+-{'----':>9}-+-{'----':>11}")

    for bits in [4, 3, 2]:
        tq = TurboQuantMSE(head_dim=dim, bits=bits, device=device)
        keys_hat = tq.roundtrip(keys)
        scores_q = (queries @ keys_hat.T) * scale
        flat_fp = scores_fp32.flatten()
        flat_q = scores_q.flatten()
        cos = torch.nn.functional.cosine_similarity(
            flat_fp.unsqueeze(0), flat_q.unsqueeze(0)
        ).item()
        mae = torch.mean(torch.abs(flat_fp - flat_q)).item()
        max_err = torch.max(torch.abs(flat_fp - flat_q)).item()
        probs_fp = torch.softmax(scores_fp32, dim=-1)
        probs_q = torch.softmax(scores_q, dim=-1)
        kl = torch.sum(probs_fp * (torch.log(probs_fp + 1e-10) - torch.log(probs_q + 1e-10))).item()
        kl_per_query = kl / num_queries
        print(
            f"  {bits:>4} | {cos:>13.6f} | {mae:>11.6f} | {max_err:>9.4f} | {kl_per_query:>11.6f}"
        )


# ---- BENCHMARK 3: Rotation Latency (GPU) ----
def bench_rotation_gpu(device: str):
    print("\n" + "=" * 90)
    print("  BENCHMARK 3: ROTATION LATENCY ON GPU")
    print("  Triton fused RHT (single launch) vs Dense QR matmul (cuBLAS)")
    print("=" * 90)

    if device != "cuda":
        print("\n  [SKIPPED] GPU not available")
        return

    dim = QWEN35_HEAD_DIM
    use_cuda = True

    rht = RHTRotation(dim, device=device)
    dense = DenseQRRotation(dim, device=device)

    triton_enabled = rht._use_triton
    print(
        f"\n  head_dim={dim}, RHT backend={'Triton fused' if triton_enabled else 'PyTorch batched'}"
    )
    print()

    print(
        f"  {'Batch':>8} | {'RHT (ms)':>10} | {'Dense QR (ms)':>14} | {'Speedup':>8} | {'Winner':>12}"
    )
    print(
        f"  {'--------':>8}-+-{'----------':>10}-+-{'--------------':>14}-+-{'--------':>8}-+-{'------------':>12}"
    )

    for batch in [64, 256, 1024, 4096, 16384]:
        x = torch.randn(batch, dim, device=device, dtype=torch.float32)
        t_rht = _timer(lambda: rht(x), use_cuda=use_cuda)
        t_dense = _timer(lambda: dense(x), use_cuda=use_cuda)
        speedup = t_dense / t_rht if t_rht > 0 else float("inf")
        winner = "RHT" if speedup > 1.0 else "Dense QR"
        print(f"  {batch:>8} | {t_rht:>10.3f} | {t_dense:>14.3f} | {speedup:>7.2f}x | {winner:>12}")


# ---- BENCHMARK 4: Memory Analysis ----
def bench_memory():
    print("\n" + "=" * 90)
    print("  BENCHMARK 4: KV CACHE MEMORY -- Qwen3.5-9B on RTX 5070 Ti (16 GB)")
    print("  8 of 32 layers use traditional KV cache (full_attention).")
    print("=" * 90)

    head_dim = QWEN35_HEAD_DIM
    num_kv_heads = QWEN35_NUM_KV_HEADS
    num_attn_layers = QWEN35_FULL_ATTN_LAYERS

    fp16_per_token_all = num_kv_heads * head_dim * 2 * 2 * num_attn_layers

    print(
        f"\n  {'Context':>10} | {'FP16 KV':>10} | {'TQ4 KV':>10} | {'TQ3 KV':>10} | "
        f"{'TQ2 KV':>10} | {'FP16->TQ4':>10}"
    )
    print(
        f"  {'----------':>10}-+-{'----------':>10}-+-{'----------':>10}-+-{'----------':>10}-+-"
        f"{'----------':>10}-+-{'----------':>10}"
    )

    for ctx in [1024, 4096, 16384, 32768, 65536, 131072, 262144]:
        fp16_bytes = fp16_per_token_all * ctx
        results = {}
        for bits in [4, 3, 2]:
            packed_per_coord = {4: 0.5, 3: 1.0, 2: 0.25}[bits]
            indices = num_kv_heads * head_dim * 2 * packed_per_coord
            norms = num_kv_heads * 2 * 4
            results[bits] = (indices + norms) * num_attn_layers * ctx
        ratio_4 = fp16_bytes / results[4] if results[4] > 0 else 0
        print(
            f"  {ctx:>10,} | {fmt_bytes(fp16_bytes):>10} | {fmt_bytes(results[4]):>10} | "
            f"{fmt_bytes(results[3]):>10} | {fmt_bytes(results[2]):>10} | {ratio_4:>9.1f}x"
        )

    available = 9.5 * 1024**3
    max_ctx_fp16 = int(available / fp16_per_token_all)
    tq4_per_token = (
        QWEN35_NUM_KV_HEADS * QWEN35_HEAD_DIM * 2 * 0.5 + QWEN35_NUM_KV_HEADS * 2 * 4
    ) * QWEN35_FULL_ATTN_LAYERS
    max_ctx_tq4 = int(available / tq4_per_token)

    print("\n  RTX 5070 Ti: AWQ 4-bit model ~5GB, overhead ~1.5GB, KV budget ~9.5GB")
    print(f"  Max context FP16:    {max_ctx_fp16:>10,} tokens")
    print(
        f"  Max context TQ4-bit: {max_ctx_tq4:>10,} tokens  ({max_ctx_tq4 / max_ctx_fp16:.1f}x extension)"
    )


# ---- BENCHMARK 5: Rotation Storage ----
def bench_rotation_storage():
    print("\n" + "=" * 90)
    print("  BENCHMARK 5: ROTATION PARAMETER STORAGE OVERHEAD")
    print("=" * 90)

    dim = QWEN35_HEAD_DIM
    num_layers = QWEN35_FULL_ATTN_LAYERS
    rht_total = dim * 4 * 2 * num_layers
    dense_total = dim * dim * 4 * 2 * num_layers

    print(f"\n  head_dim={dim}, {num_layers} attention layers, separate K/V rotations")
    print()
    print(f"  {'Scope':>30} | {'RHT':>14} | {'Dense QR':>14} | {'Ratio':>8}")
    print(f"  {'------------------------------':>30}-+-{'----':>14}-+-{'----':>14}-+-{'----':>8}")
    print(
        f"  {'Per rotation':>30} | {fmt_bytes(dim * 4):>14} | {fmt_bytes(dim * dim * 4):>14} | {dim:>7}x"
    )
    print(
        f"  {'Per layer (K+V)':>30} | {fmt_bytes(dim * 4 * 2):>14} | {fmt_bytes(dim * dim * 4 * 2):>14} | {dim:>7}x"
    )
    print(
        f"  {'All ' + str(num_layers) + ' layers':>30} | {fmt_bytes(rht_total):>14} | {fmt_bytes(dense_total):>14} | {dim:>7}x"
    )


# ---- BENCHMARK 6: Throughput ----
def bench_throughput(device: str):
    print("\n" + "=" * 90)
    print("  BENCHMARK 6: ENCODE/DECODE THROUGHPUT")
    print("=" * 90)

    dim = QWEN35_HEAD_DIM
    use_cuda = device == "cuda"

    print(f"\n  batch=1, kv_heads={QWEN35_NUM_KV_HEADS}, head_dim={dim}")
    print()
    print(
        f"  {'Method':<28} | {'Seq Len':>8} | {'Encode (ms)':>12} | {'Decode (ms)':>12} | {'tok/s enc':>10}"
    )
    print(
        f"  {'----------------------------':<28}-+-{'--------':>8}-+-{'------------':>12}-+-{'------------':>12}-+-{'----------':>10}"
    )

    for seq_len in [128, 512, 2048]:
        x = torch.randn(1, QWEN35_NUM_KV_HEADS, seq_len, dim, device=device, dtype=torch.float32)
        for name, cls in [
            ("TQ4-RHT-Triton (ours)", TurboQuantMSE),
            ("TQ4-DenseQR-matmul (others)", DenseQRTurboQuantMSE),
        ]:
            tq = cls(head_dim=dim, bits=4, device=device)
            t_enc = _timer(lambda: tq.encode(x), use_cuda=use_cuda)
            compressed = tq.encode(x)
            t_dec = _timer(lambda: tq.decode(compressed), use_cuda=use_cuda)
            tps = seq_len / (t_enc / 1000)
            print(f"  {name:<28} | {seq_len:>8} | {t_enc:>12.3f} | {t_dec:>12.3f} | {tps:>10,.0f}")


# ---- BENCHMARK 7: Dejan.ai Direct Comparison ----
def bench_dejan_comparison(device: str):
    print("\n" + "=" * 90)
    print("  BENCHMARK 7: DIRECT COMPARISON WITH DEJAN.AI IMPLEMENTATION")
    print("  Using their downloaded code with identical test vectors.")
    print("=" * 90)

    dim = QWEN35_HEAD_DIM

    try:
        from turboquant_core import TurboQuantMSE as DejanTQ
    except ImportError:
        print("\n  [SKIPPED] Dejan.ai code not found in benchmarks/dejan_baseline/")
        return

    n = 2048
    torch.manual_seed(42)
    x = torch.randn(n, dim, device=device, dtype=torch.float32)

    print(f"\n  {n} random vectors, dim={dim}, seed=42")
    print()

    print(
        f"  {'Bits':>4} | {'Impl':<28} | {'Cosine Sim':>11} | {'MSE':>11} | {'IP Corr':>9} | {'Rotation':>12}"
    )
    print(
        f"  {'----':>4}-+-{'----------------------------':<28}-+-{'----':>11}-+-{'----':>11}-+-{'----':>9}-+-{'----':>12}"
    )

    for bits in [4, 3, 2]:
        # Ours (RHT Triton)
        ours = TurboQuantMSE(head_dim=dim, bits=bits, device=device)
        c_ours = ours.encode(x)
        x_ours = ours.decode(c_ours)

        mse_ours = torch.mean((x - x_ours) ** 2).item()
        xn = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
        xhn = x_ours / (torch.norm(x_ours, dim=-1, keepdim=True) + 1e-8)
        cos_ours = torch.mean(torch.sum(xn * xhn, dim=-1)).item()
        ip_a = torch.sum(x[: n // 2] * x[n // 2 :], dim=-1)
        ip_b = torch.sum(x_ours[: n // 2] * x_ours[n // 2 :], dim=-1)
        ipc_ours = torch.corrcoef(torch.stack([ip_a, ip_b]))[0, 1].item()

        print(
            f"  {bits:>4} | {'fused-turboquant (ours)':<28} | {cos_ours:>11.6f} | {mse_ours:>11.6f} | "
            f"{ipc_ours:>9.6f} | {'RHT Triton':>12}"
        )

        # Dejan.ai (Dense QR)
        dejan = DejanTQ(d=dim, bits=bits, device=device, rotation_seed=0)
        q_dejan = dejan.quantize(x)
        x_dejan = dejan.dequantize(q_dejan)

        mse_dejan = torch.mean((x - x_dejan) ** 2).item()
        xhn_d = x_dejan / (torch.norm(x_dejan, dim=-1, keepdim=True) + 1e-8)
        cos_dejan = torch.mean(torch.sum(xn * xhn_d, dim=-1)).item()
        ip_c = torch.sum(x_dejan[: n // 2] * x_dejan[n // 2 :], dim=-1)
        ipc_dejan = torch.corrcoef(torch.stack([ip_a, ip_c]))[0, 1].item()

        print(
            f"  {bits:>4} | {'Dejan.ai (their code)':<28} | {cos_dejan:>11.6f} | {mse_dejan:>11.6f} | "
            f"{ipc_dejan:>9.6f} | {'Dense QR':>12}"
        )
        print(
            f"  {'----':>4}-+-{'----------------------------':<28}-+-{'----':>11}-+-{'----':>11}-+-{'----':>9}-+-{'----':>12}"
        )

    # Architecture comparison table
    print("\n  ARCHITECTURAL DIFFERENCES:")
    print(f"  {'Feature':<35} | {'fused-turboquant (ours)':<30} | {'Dejan.ai':<30}")
    print(
        f"  {'-----------------------------------':<35}-+-{'------------------------------':<30}-+-{'------------------------------':<30}"
    )
    print(
        f"  {'Rotation method':<35} | {'RHT (O(d log d), Triton fused)':<30} | {'Dense QR (O(d^2), matmul)':<30}"
    )
    print(
        f"  {'Rotation storage per layer':<35} | {fmt_bytes(dim * 4) + ' (signs only)':<30} | {fmt_bytes(dim * dim * 4) + ' (d x d matrix)':<30}"
    )
    print(
        f"  {'Triton kernel purpose':<35} | {'Fused RHT butterfly':<30} | {'Fused Q.K^T from uint8 keys':<30}"
    )
    print(
        f"  {'Value compression':<35} | {'Yes (K + V both compressed)':<30} | {'No (K only compressed)':<30}"
    )
    print(
        f"  {'Nibble packing (4-bit)':<35} | {'Yes (2 indices per byte)':<30} | {'No (1 uint8 per index)':<30}"
    )
    print(
        f"  {'Sub-byte packing (2-bit)':<35} | {'Yes (4 indices per byte)':<30} | {'No (1 uint8 per index)':<30}"
    )
    print(f"  {'vLLM plugin':<35} | {'Yes (registered entry point)':<30} | {'No':<30}")
    print(
        f"  {'Target model':<35} | {'Qwen3.5-9B (hybrid 8/32)':<30} | {'Gemma 3 4B IT (27 layers)':<30}"
    )


# ---- MAIN ----
def main():
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("+" + "=" * 88 + "+")
    print("|" + " fused-turboquant: Comprehensive Benchmark Suite".center(88) + "|")
    print("+" + "=" * 88 + "+")

    if device == "cuda":
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n  GPU: {gpu} ({mem:.1f} GB)")
    else:
        print("\n  Device: CPU (GPU not available)")

    print(f"  Triton: {'AVAILABLE' if is_triton_available() else 'NOT AVAILABLE'}")
    print("  Target: Qwen3.5-9B (8 attention + 24 DeltaNet layers)")
    print(f"  head_dim={QWEN35_HEAD_DIM}, kv_heads={QWEN35_NUM_KV_HEADS}")

    bench_quality_fairness(device)
    bench_attention_accuracy(device)
    bench_rotation_gpu(device)
    bench_memory()
    bench_rotation_storage()
    bench_throughput(device)
    bench_dejan_comparison(device)

    print("\n" + "=" * 90)
    print("  BENCHMARK COMPLETE")
    print("=" * 90)
    print()
    print("  Fairness notes:")
    print("  - RHT and Dense QR use identical Lloyd-Max codebooks and packing.")
    print("  - Quality differences are only due to finite-precision rotation.")
    print("  - Memory analysis uses Qwen3.5-9B's real architecture (8/32 layers).")
    print("  - Attention accuracy uses FP32 Q.K^T as ground truth.")
    print("  - Dejan.ai comparison uses their downloaded code verbatim.")
    print()


if __name__ == "__main__":
    main()
