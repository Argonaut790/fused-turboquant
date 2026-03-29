"""
Fused Triton kernel for quantized attention scores with RHT.

Instead of: dequantize keys to fp16 -> Q @ K^T  (loads fp16 keys from HBM)
We do:      RHT(Q) -> gather(centroids, key_indices) * norms  (loads uint8 indices)

Key math identity (since RHT is orthogonal):
    <q, RHT_inv(centroids[idx])> = <RHT(q), centroids[idx]>

So pre-rotate the query once with a single RHT call (O(d log d)),
then per-KV-position work is just: score[s] = norm[s] * sum_d(q_rot[d] * C[idx[s,d]]) * scale

Compared to Dejan.ai's fused kernel which uses Dense QR for query rotation (O(d^2)),
ours uses the Triton fused RHT kernel for query rotation (O(d log d)).
"""

from __future__ import annotations

import math
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:

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
        Q_ptr,          # pre-rotated query: [BH_q, head_dim]
        K_idx_ptr,      # compressed key indices: [BH_kv, seq_len, head_dim] uint8
        K_norms_ptr,    # key norms: [BH_kv, seq_len] float32
        C_ptr,          # centroid table: [n_levels] float32
        Out_ptr,        # output scores: [BH_q, seq_len] float32
        seq_len,
        head_dim: tl.constexpr,
        n_q_heads,
        n_kv_heads,
        scale,
        stride_q_bh, stride_q_d,
        stride_ki_bh, stride_ki_s, stride_ki_d,
        stride_kn_bh, stride_kn_s,
        stride_o_bh, stride_o_s,
        BLOCK_S: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_s = tl.program_id(1)

        batch_idx = pid_bh // n_q_heads
        q_head_idx = pid_bh % n_q_heads
        gqa_ratio = n_q_heads // n_kv_heads
        kv_head_idx = q_head_idx // gqa_ratio
        kv_bh = batch_idx * n_kv_heads + kv_head_idx

        s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        s_mask = s_offs < seq_len

        acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

        for d_start in range(0, head_dim, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < head_dim

            q_ptrs = Q_ptr + pid_bh * stride_q_bh + d_offs * stride_q_d
            q_vals = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

            ki_ptrs = (K_idx_ptr
                       + kv_bh * stride_ki_bh
                       + s_offs[:, None] * stride_ki_s
                       + d_offs[None, :] * stride_ki_d)
            combined_mask = s_mask[:, None] & d_mask[None, :]
            k_idx = tl.load(ki_ptrs, mask=combined_mask, other=0).to(tl.int32)

            k_vals = tl.load(C_ptr + k_idx, mask=combined_mask, other=0.0).to(tl.float32)

            acc += tl.sum(k_vals * q_vals[None, :], axis=1)

        kn_ptrs = K_norms_ptr + kv_bh * stride_kn_bh + s_offs * stride_kn_s
        norms = tl.load(kn_ptrs, mask=s_mask, other=0.0).to(tl.float32)

        scores = norms * acc * scale

        o_ptrs = Out_ptr + pid_bh * stride_o_bh + s_offs * stride_o_s
        tl.store(o_ptrs, scores, mask=s_mask)


def fused_qk_scores_rht(
    q_rotated: torch.Tensor,     # [batch, n_q_heads, q_len, head_dim] pre-rotated via RHT
    key_indices: torch.Tensor,   # [batch, n_kv_heads, kv_len, head_dim] uint8
    key_norms: torch.Tensor,     # [batch, n_kv_heads, kv_len] float32
    centroids: torch.Tensor,     # [n_levels] float32
    scale: float,
) -> torch.Tensor:
    """
    Compute attention scores Q @ K^T directly from compressed keys.

    The query is pre-rotated via RHT (O(d log d)) instead of Dense QR matmul (O(d^2)).
    The kernel loads uint8 indices and gathers from a small centroid table in L1 cache.

    Returns: attention scores [batch, n_q_heads, q_len, kv_len]
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for fused attention kernel")

    batch, n_q_heads, q_len, head_dim = q_rotated.shape
    _, n_kv_heads, kv_len, _ = key_indices.shape

    q_flat = q_rotated.reshape(batch * n_q_heads * q_len, head_dim).contiguous()
    ki_flat = key_indices.reshape(batch * n_kv_heads, kv_len, head_dim).contiguous()
    kn_flat = key_norms.reshape(batch * n_kv_heads, kv_len).contiguous()
    centroids = centroids.contiguous().float()

    out = torch.empty(batch * n_q_heads * q_len, kv_len,
                      device=q_rotated.device, dtype=torch.float32)

    effective_q_heads = n_q_heads * q_len

    grid = (batch * effective_q_heads, triton.cdiv(kv_len, 64))

    _fused_qk_scores_kernel[grid](
        q_flat, ki_flat, kn_flat, centroids, out,
        kv_len, head_dim,
        effective_q_heads, n_kv_heads,
        scale,
        q_flat.stride(0), q_flat.stride(1),
        ki_flat.stride(0), ki_flat.stride(1), ki_flat.stride(2),
        kn_flat.stride(0), kn_flat.stride(1),
        out.stride(0), out.stride(1),
    )

    return out.reshape(batch, n_q_heads, q_len, kv_len)
