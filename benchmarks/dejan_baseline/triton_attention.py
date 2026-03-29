"""
Triton kernel for fused quantized attention scores.

Instead of: dequantize keys to fp16 → Q @ K^T  (loads fp16 keys from HBM)
We do:      Q_rotated @ gather(centroids, key_indices) * norms   (loads uint8 indices)

Memory bandwidth savings: ~4x (uint8 + small centroid table vs fp16 keys)

Key math:
    <q, R^T @ centroids[idx]> = <R @ q, centroids[idx]>
    
    So pre-rotate query once (one matmul), then the per-KV-position work
    is just: score[s] = norm[s] * sum_d(q_rot[d] * centroids[idx[s,d]]) / sqrt(d)
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================================
# Triton kernel: fused gather-dot for quantized Q@K^T
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": 32, "BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_S": 64, "BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_S": 128, "BLOCK_D": 64}, num_warps=8),
        triton.Config({"BLOCK_S": 64, "BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_S": 128, "BLOCK_D": 128}, num_warps=8),
    ],
    key=["seq_len", "head_dim"],
)
@triton.jit
def _fused_qk_scores_kernel(
    # Pre-rotated query: [BH_q, head_dim]  (BH_q = batch * n_q_heads)
    Q_ptr,
    # Compressed keys
    K_idx_ptr,    # [BH_kv, seq_len, head_dim] uint8
    K_norms_ptr,  # [BH_kv, seq_len] float16
    # Centroid table
    C_ptr,        # [n_levels] float32
    # Output scores
    Out_ptr,      # [BH_q, seq_len] float32
    # Dimensions
    seq_len,
    head_dim: tl.constexpr,
    n_q_heads,
    n_kv_heads,
    scale,        # 1/sqrt(head_dim)
    # Strides — Q: [BH_q, head_dim]
    stride_q_bh, stride_q_d,
    # Strides — K_idx: [BH_kv, seq_len, head_dim]
    stride_ki_bh, stride_ki_s, stride_ki_d,
    # Strides — K_norms: [BH_kv, seq_len]
    stride_kn_bh, stride_kn_s,
    # Strides — Out: [BH_q, seq_len]
    stride_o_bh, stride_o_s,
    # Block sizes (autotuned)
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute attention scores from pre-rotated queries and quantized keys.

    For each (query_head, kv_block):
        score[s] = key_norm[s] * sum_d(q_rot[d] * centroids[key_idx[s, d]]) * scale
    """
    pid_bh = tl.program_id(0)   # batch * query_head
    pid_s = tl.program_id(1)    # seq block

    # GQA: map query head → KV head
    batch_idx = pid_bh // n_q_heads
    q_head_idx = pid_bh % n_q_heads
    gqa_ratio = n_q_heads // n_kv_heads
    kv_head_idx = q_head_idx // gqa_ratio
    kv_bh = batch_idx * n_kv_heads + kv_head_idx

    # Sequence positions this program handles
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len

    # Accumulate dot product over head_dim in blocks
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    for d_start in range(0, head_dim, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < head_dim

        # Load query slice: Q[pid_bh, d_offs]
        q_ptrs = Q_ptr + pid_bh * stride_q_bh + d_offs * stride_q_d
        q_vals = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        # Load key indices: K_idx[kv_bh, s_offs, d_offs] → [BLOCK_S, BLOCK_D]
        ki_ptrs = (K_idx_ptr
                   + kv_bh * stride_ki_bh
                   + s_offs[:, None] * stride_ki_s
                   + d_offs[None, :] * stride_ki_d)
        combined_mask = s_mask[:, None] & d_mask[None, :]
        k_idx = tl.load(ki_ptrs, mask=combined_mask, other=0).to(tl.int32)

        # Gather centroids: C[k_idx] → [BLOCK_S, BLOCK_D]
        k_vals = tl.load(C_ptr + k_idx, mask=combined_mask, other=0.0).to(tl.float32)

        # Partial dot product: sum over D block
        acc += tl.sum(k_vals * q_vals[None, :], axis=1)

    # Load key norms: K_norms[kv_bh, s_offs]
    kn_ptrs = K_norms_ptr + kv_bh * stride_kn_bh + s_offs * stride_kn_s
    norms = tl.load(kn_ptrs, mask=s_mask, other=0.0).to(tl.float32)

    # Final score = norm * dot_product * scale
    scores = norms * acc * scale

    # Store
    o_ptrs = Out_ptr + pid_bh * stride_o_bh + s_offs * stride_o_s
    tl.store(o_ptrs, scores, mask=s_mask)


# ============================================================================
# Python wrapper
# ============================================================================

def fused_qk_scores(
    q_rotated: torch.Tensor,     # [batch, n_q_heads, q_len, head_dim] — pre-rotated
    key_indices: torch.Tensor,   # [batch, n_kv_heads, kv_len, head_dim] uint8
    key_norms: torch.Tensor,     # [batch, n_kv_heads, kv_len] float16
    centroids: torch.Tensor,     # [n_levels] float32
    scale: float,                # 1/sqrt(head_dim)
) -> torch.Tensor:
    """Compute attention scores Q @ K^T using compressed keys.

    Args:
        q_rotated: Query vectors pre-multiplied by rotation matrix Q^T.
                   Shape [batch, n_q_heads, q_len, head_dim]
        key_indices: Quantized key indices. [batch, n_kv_heads, kv_len, head_dim]
        key_norms: Key vector norms. [batch, n_kv_heads, kv_len]
        centroids: Lloyd-Max centroid values. [n_levels]
        scale: Attention scale factor (1/sqrt(head_dim))

    Returns:
        Attention scores [batch, n_q_heads, q_len, kv_len]
    """
    batch, n_q_heads, q_len, head_dim = q_rotated.shape
    _, n_kv_heads, kv_len, _ = key_indices.shape

    # For q_len > 1 (prefill), handle each query position
    # Reshape to [batch * n_q_heads * q_len, head_dim] for the kernel
    q_flat = q_rotated.reshape(batch * n_q_heads * q_len, head_dim).contiguous()
    ki_flat = key_indices.reshape(batch * n_kv_heads, kv_len, head_dim).contiguous()
    kn_flat = key_norms.reshape(batch * n_kv_heads, kv_len).contiguous()
    centroids = centroids.contiguous().float()

    # Output: [batch * n_q_heads * q_len, kv_len]
    out = torch.empty(batch * n_q_heads * q_len, kv_len,
                      device=q_rotated.device, dtype=torch.float32)

    # For the kernel, we treat each query position as a separate "head"
    # But GQA mapping needs to account for q_len grouping
    effective_q_heads = n_q_heads * q_len

    # Grid
    grid = (batch * effective_q_heads, triton.cdiv(kv_len, 64))  # 64 is a safe default

    _fused_qk_scores_kernel[grid](
        q_flat,
        ki_flat, kn_flat,
        centroids,
        out,
        kv_len,
        head_dim,
        effective_q_heads,
        n_kv_heads,
        scale,
        # Strides Q
        q_flat.stride(0), q_flat.stride(1),
        # Strides K_idx
        ki_flat.stride(0), ki_flat.stride(1), ki_flat.stride(2),
        # Strides K_norms
        kn_flat.stride(0), kn_flat.stride(1),
        # Strides Out
        out.stride(0), out.stride(1),
    )

    return out.reshape(batch, n_q_heads, q_len, kv_len)


# ============================================================================
# Self-test: verify Triton kernel matches PyTorch reference
# ============================================================================

def test_fused_kernel():
    """Compare fused Triton kernel against explicit dequantize + matmul."""
    import sys
    from turboquant_core import TurboQuantMSE

    torch.manual_seed(42)

    batch, n_q_heads, n_kv_heads = 1, 8, 4
    q_len, kv_len, head_dim = 1, 128, 256
    bits = 4

    # Create quantizer
    tq = TurboQuantMSE(d=head_dim, bits=bits, device="cuda")

    # Random Q and K
    q = torch.randn(batch, n_q_heads, q_len, head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn(batch, n_kv_heads, kv_len, head_dim, device="cuda", dtype=torch.float32)

    # Quantize K
    k_q = tq.quantize(k)
    k_indices = k_q["idx"]       # uint8
    k_norms = k_q["norms"]       # fp16

    # --- Reference: explicit dequantize + matmul ---
    k_deq = tq.dequantize(k_q)   # [batch, n_kv, kv_len, head_dim]
    # GQA expand
    gqa_ratio = n_q_heads // n_kv_heads
    k_expanded = k_deq.repeat_interleave(gqa_ratio, dim=1)
    scale = 1.0 / math.sqrt(head_dim)
    ref_scores = torch.matmul(q, k_expanded.transpose(2, 3)) * scale

    # --- Fused: pre-rotate query, then Triton kernel ---
    # Pre-rotate query: q_rot = q @ Q_T  (Q_T is the rotation matrix transpose)
    q_rot = q @ tq.Q_T.unsqueeze(0).unsqueeze(0)  # broadcast over batch, heads
    fused_scores = fused_qk_scores(q_rot, k_indices, k_norms, tq.centroids, scale)

    # Compare
    max_diff = (ref_scores - fused_scores).abs().max().item()
    mean_diff = (ref_scores - fused_scores).abs().mean().item()
    cos = torch.nn.functional.cosine_similarity(
        ref_scores.flatten().unsqueeze(0),
        fused_scores.flatten().unsqueeze(0)
    ).item()

    print(f"Fused kernel test (batch={batch}, q_heads={n_q_heads}, kv_heads={n_kv_heads}, "
          f"q_len={q_len}, kv_len={kv_len}, d={head_dim}, bits={bits}):")
    print(f"  Max diff:   {max_diff:.6f}")
    print(f"  Mean diff:  {mean_diff:.6f}")
    print(f"  Cosine sim: {cos:.6f}")
    print(f"  {'PASS' if cos > 0.999 else 'FAIL'}")
    print()
    return cos > 0.999


def benchmark_fused_vs_standard():
    """Benchmark fused kernel vs standard dequantize+matmul."""
    from turboquant_core import TurboQuantMSE

    torch.manual_seed(42)
    batch, n_q_heads, n_kv_heads = 1, 8, 4
    head_dim, bits = 256, 4
    scale = 1.0 / math.sqrt(head_dim)

    tq = TurboQuantMSE(d=head_dim, bits=bits, device="cuda")
    gqa_ratio = n_q_heads // n_kv_heads

    for kv_len in [128, 512, 1024, 2048, 4096]:
        q = torch.randn(batch, n_q_heads, 1, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, n_kv_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)

        # Quantize
        k_q = tq.quantize(k.float())
        k_deq = tq.dequantize(k_q).half()
        k_indices = k_q["idx"]
        k_norms = k_q["norms"]

        # Pre-rotate query
        q_rot = (q.float() @ tq.Q_T.unsqueeze(0).unsqueeze(0)).contiguous()

        # Warm up
        for _ in range(5):
            k_exp = k_deq.repeat_interleave(gqa_ratio, dim=1)
            _ = torch.matmul(q, k_exp.transpose(2, 3)) * scale
            _ = fused_qk_scores(q_rot, k_indices, k_norms, tq.centroids, scale)
        torch.cuda.synchronize()

        # Benchmark standard
        import time
        n_runs = 100
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            k_exp = k_deq.repeat_interleave(gqa_ratio, dim=1)
            _ = torch.matmul(q, k_exp.transpose(2, 3)) * scale
        torch.cuda.synchronize()
        t_std = (time.perf_counter() - t0) / n_runs * 1000

        # Benchmark fused
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = fused_qk_scores(q_rot, k_indices, k_norms, tq.centroids, scale)
        torch.cuda.synchronize()
        t_fused = (time.perf_counter() - t0) / n_runs * 1000

        speedup = t_std / t_fused
        print(f"  kv_len={kv_len:5d}  standard={t_std:.3f}ms  fused={t_fused:.3f}ms  "
              f"speedup={speedup:.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("Triton fused attention kernel tests")
    print("=" * 60)

    ok = test_fused_kernel()
    if not ok:
        print("Kernel test FAILED — skipping benchmark")
        exit(1)

    print("Benchmarking fused vs standard attention scores:")
    benchmark_fused_vs_standard()
