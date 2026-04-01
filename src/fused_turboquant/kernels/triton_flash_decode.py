"""
Fused FlashDecode-style kernel: QK scores + online softmax + V attention in one pass.

Combines Tier 2c (fused V) and Tier 2d (online softmax) into a single kernel
that tiles over the KV sequence with a running max/sum, eliminating:
  1. The full [batch, heads, seq_len] score tensor (2d)
  2. The full [batch, heads, seq_len, head_dim] dense V tensor (2c)

Uses the "online softmax" trick (Milakov & Gimelshein, 2018):
  - Maintain running max m and running sum of exp(score - m)
  - When a new block has a larger max, rescale the running accumulator
  - Final output = accumulated_value / sum_exp

Both K and V are read from packed compressed storage and unpacked inline.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _flash_decode_kernel(
        Q_ptr,  # pre-rotated query: [BH_q, head_dim] float
        K_idx_ptr,  # packed K indices: [BH_kv, seq_len, packed_dim_k] uint8
        K_norms_ptr,  # K norms: [BH_kv, seq_len] fp16
        V_idx_ptr,  # packed V indices: [BH_kv, seq_len, packed_dim_v] uint8
        V_norms_ptr,  # V norms: [BH_kv, seq_len] fp16
        C_K_ptr,  # K centroid table: [n_levels_k] float32
        C_V_ptr,  # V centroid table: [n_levels_v] float32
        Out_ptr,  # output: [BH_q, head_dim] float32 (in rotated space)
        seq_len,
        head_dim: tl.constexpr,
        n_q_heads,
        n_kv_heads,
        scale,
        stride_q_bh,
        stride_q_d,
        stride_ki_bh,
        stride_ki_s,
        stride_ki_d,
        stride_kn_bh,
        stride_kn_s,
        stride_vi_bh,
        stride_vi_s,
        stride_vi_d,
        stride_vn_bh,
        stride_vn_s,
        stride_o_bh,
        stride_o_d,
        BITS_K: tl.constexpr,
        BITS_V: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        """FlashDecode: online softmax over KV blocks, output in rotated space."""
        pid_bh = tl.program_id(0)

        batch_idx = pid_bh // n_q_heads
        q_head_idx = pid_bh % n_q_heads
        gqa_ratio = n_q_heads // n_kv_heads
        kv_head_idx = q_head_idx // gqa_ratio
        kv_bh = batch_idx * n_kv_heads + kv_head_idx

        d_idx = tl.arange(0, head_dim)

        q_vals = tl.load(
            Q_ptr + pid_bh * stride_q_bh + d_idx * stride_q_d,
            mask=d_idx < head_dim,
            other=0.0,
        ).to(tl.float32)

        m_prev = float("-inf")
        l_prev = 0.0
        acc = tl.zeros((head_dim,), dtype=tl.float32)

        for s_start in range(0, seq_len, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            s_mask = s_offs < seq_len

            # --- QK score for this block of positions ---
            # Unpack K indices
            if BITS_K == 4:
                pack_dk = d_idx // 2
                ki_ptrs = (
                    K_idx_ptr
                    + kv_bh * stride_ki_bh
                    + s_offs[:, None] * stride_ki_s
                    + pack_dk[None, :] * stride_ki_d
                )
                cm = s_mask[:, None] & (d_idx < head_dim)[None, :]
                pv = tl.load(ki_ptrs, mask=cm, other=0).to(tl.int32)
                is_low = (d_idx % 2) == 0
                k_idx = tl.where(is_low[None, :], pv & 0xF, (pv >> 4) & 0xF)
            elif BITS_K == 3:
                bit_off = d_idx * 3
                byte_idx = bit_off >> 3
                bit_shift = bit_off & 7
                packed_total_k = head_dim * 3 // 8
                cm = s_mask[:, None] & (d_idx < head_dim)[None, :]
                ki_base = K_idx_ptr + kv_bh * stride_ki_bh
                k_off = s_offs[:, None] * stride_ki_s
                b0_ptr = ki_base + k_off + byte_idx[None, :] * stride_ki_d
                b0 = tl.load(b0_ptr, mask=cm, other=0).to(tl.int32)
                b1v = (byte_idx + 1) < packed_total_k
                b1_ptr = ki_base + k_off + (byte_idx + 1)[None, :] * stride_ki_d
                b1 = tl.load(b1_ptr, mask=cm & b1v[None, :], other=0).to(tl.int32)
                k_idx = ((b0 | (b1 << 8)) >> bit_shift[None, :]) & 0x7
            elif BITS_K == 2:
                pack_dk = d_idx // 4
                ki_ptrs = (
                    K_idx_ptr
                    + kv_bh * stride_ki_bh
                    + s_offs[:, None] * stride_ki_s
                    + pack_dk[None, :] * stride_ki_d
                )
                cm = s_mask[:, None] & (d_idx < head_dim)[None, :]
                pv = tl.load(ki_ptrs, mask=cm, other=0).to(tl.int32)
                shift = (d_idx % 4) * 2
                k_idx = (pv >> shift[None, :]) & 0x3
            else:
                ki_ptrs = (
                    K_idx_ptr
                    + kv_bh * stride_ki_bh
                    + s_offs[:, None] * stride_ki_s
                    + d_idx[None, :] * stride_ki_d
                )
                cm = s_mask[:, None] & (d_idx < head_dim)[None, :]
                k_idx = tl.load(ki_ptrs, mask=cm, other=0).to(tl.int32)

            k_centroids = tl.load(C_K_ptr + k_idx, mask=cm, other=0.0).to(tl.float32)
            k_norms = tl.load(
                K_norms_ptr + kv_bh * stride_kn_bh + s_offs * stride_kn_s,
                mask=s_mask,
                other=0,
            ).to(tl.float32)

            dot = tl.sum(k_centroids * q_vals[None, :], axis=1)  # [BLOCK_S]
            scores = k_norms * dot * scale
            scores = tl.where(s_mask, scores, float("-inf"))

            # --- Online softmax update ---
            m_block = tl.max(scores, axis=0)
            m_new = tl.maximum(m_prev, m_block)
            correction = tl.exp(m_prev - m_new)
            exp_scores = tl.exp(scores - m_new)
            l_new = l_prev * correction + tl.sum(exp_scores, axis=0)
            acc = acc * correction

            # --- Accumulate V weighted by exp_scores ---
            if BITS_V == 4:
                pack_dv = d_idx // 2
                vi_ptrs = (
                    V_idx_ptr
                    + kv_bh * stride_vi_bh
                    + s_offs[:, None] * stride_vi_s
                    + pack_dv[None, :] * stride_vi_d
                )
                pv2 = tl.load(vi_ptrs, mask=cm, other=0).to(tl.int32)
                is_low_v = (d_idx % 2) == 0
                v_idx = tl.where(is_low_v[None, :], pv2 & 0xF, (pv2 >> 4) & 0xF)
            elif BITS_V == 3:
                bit_off_v = d_idx * 3
                byte_idx_v = bit_off_v >> 3
                bit_shift_v = bit_off_v & 7
                packed_total_v = head_dim * 3 // 8
                vi_base = V_idx_ptr + kv_bh * stride_vi_bh
                v_off = s_offs[:, None] * stride_vi_s
                b0v_ptr = vi_base + v_off + byte_idx_v[None, :] * stride_vi_d
                b0v = tl.load(b0v_ptr, mask=cm, other=0).to(tl.int32)
                b1v2 = (byte_idx_v + 1) < packed_total_v
                b1v_ptr = vi_base + v_off + (byte_idx_v + 1)[None, :] * stride_vi_d
                b1v_val = tl.load(b1v_ptr, mask=cm & b1v2[None, :], other=0).to(tl.int32)
                v_idx = ((b0v | (b1v_val << 8)) >> bit_shift_v[None, :]) & 0x7
            elif BITS_V == 2:
                pack_dv = d_idx // 4
                vi_ptrs = (
                    V_idx_ptr
                    + kv_bh * stride_vi_bh
                    + s_offs[:, None] * stride_vi_s
                    + pack_dv[None, :] * stride_vi_d
                )
                pv2 = tl.load(vi_ptrs, mask=cm, other=0).to(tl.int32)
                shift_v = (d_idx % 4) * 2
                v_idx = (pv2 >> shift_v[None, :]) & 0x3
            else:
                vi_ptrs = (
                    V_idx_ptr
                    + kv_bh * stride_vi_bh
                    + s_offs[:, None] * stride_vi_s
                    + d_idx[None, :] * stride_vi_d
                )
                v_idx = tl.load(vi_ptrs, mask=cm, other=0).to(tl.int32)

            v_centroids = tl.load(C_V_ptr + v_idx, mask=cm, other=0.0).to(tl.float32)
            v_norms_block = tl.load(
                V_norms_ptr + kv_bh * stride_vn_bh + s_offs * stride_vn_s,
                mask=s_mask,
                other=0,
            ).to(tl.float32)

            weighted_v = (exp_scores * v_norms_block)[:, None] * v_centroids  # [BLOCK_S, head_dim]
            acc += tl.sum(weighted_v, axis=0)

            m_prev = m_new
            l_prev = l_new

        # Normalize by softmax denominator
        result = acc / (l_prev + 1e-8)

        o_ptrs = Out_ptr + pid_bh * stride_o_bh + d_idx * stride_o_d
        tl.store(o_ptrs, result, mask=d_idx < head_dim)


def flash_decode_compressed(
    q_rotated: torch.Tensor,  # [batch, n_q_heads, 1, head_dim] pre-rotated
    k_indices: torch.Tensor,  # [batch, n_kv_heads, kv_len, packed_dim_k] uint8
    k_norms: torch.Tensor,  # [batch, n_kv_heads, kv_len] fp16
    v_indices: torch.Tensor,  # [batch, n_kv_heads, kv_len, packed_dim_v] uint8
    v_norms: torch.Tensor,  # [batch, n_kv_heads, kv_len] fp16
    k_centroids: torch.Tensor,  # [n_levels_k] float32
    v_centroids: torch.Tensor,  # [n_levels_v] float32
    signs: torch.Tensor,  # [head_dim] float32 RHT signs
    scale: float,
    bits_k: int = 4,
    bits_v: int = 4,
    seq_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fully fused decode attention: QK + online softmax + V in one pass.

    Eliminates both the score tensor and the dense V tensor.
    Output is in rotated space — caller must apply inverse RHT.

    Args:
        q_rotated: pre-rotated queries [batch, n_q_heads, 1, head_dim].
        k_indices, k_norms: compressed K cache.
        v_indices, v_norms: compressed V cache.
        k_centroids, v_centroids: Lloyd-Max centroid tables.
        signs: RHT sign vector for inverse RHT.
        scale: attention scale (1/sqrt(head_dim)).
        bits_k, bits_v: quantization bits for K and V.
        seq_lens: [batch] actual sequence lengths (for masking).

    Returns:
        Attention output [batch, n_q_heads, 1, head_dim] in original space.
    """
    if not HAS_TRITON or not q_rotated.is_cuda:
        return _flash_decode_fallback(
            q_rotated,
            k_indices,
            k_norms,
            v_indices,
            v_norms,
            k_centroids,
            v_centroids,
            signs,
            scale,
            bits_k,
            bits_v,
        )

    batch, n_q_heads, q_len, head_dim = q_rotated.shape
    _, n_kv_heads, kv_len, packed_dim_k = k_indices.shape
    packed_dim_v = v_indices.shape[-1]

    q_flat = q_rotated.reshape(batch * n_q_heads * q_len, head_dim).contiguous()
    ki_flat = k_indices.reshape(batch * n_kv_heads, kv_len, packed_dim_k).contiguous()
    kn_flat = k_norms.reshape(batch * n_kv_heads, kv_len).contiguous()
    vi_flat = v_indices.reshape(batch * n_kv_heads, kv_len, packed_dim_v).contiguous()
    vn_flat = v_norms.reshape(batch * n_kv_heads, kv_len).contiguous()

    total_q = batch * n_q_heads * q_len
    out_rotated = torch.empty(total_q, head_dim, device=q_rotated.device, dtype=torch.float32)

    effective_q_heads = n_q_heads * q_len

    grid = (batch * effective_q_heads,)

    _flash_decode_kernel[grid](
        q_flat,
        ki_flat,
        kn_flat,
        vi_flat,
        vn_flat,
        k_centroids.contiguous().float(),
        v_centroids.contiguous().float(),
        out_rotated,
        kv_len,
        head_dim,
        effective_q_heads,
        n_kv_heads,
        scale,
        q_flat.stride(0),
        q_flat.stride(1),
        ki_flat.stride(0),
        ki_flat.stride(1),
        ki_flat.stride(2),
        kn_flat.stride(0),
        kn_flat.stride(1),
        vi_flat.stride(0),
        vi_flat.stride(1),
        vi_flat.stride(2),
        vn_flat.stride(0),
        vn_flat.stride(1),
        out_rotated.stride(0),
        out_rotated.stride(1),
        BITS_K=bits_k,
        BITS_V=bits_v,
        BLOCK_S=64,
    )

    from fused_turboquant.core.hadamard import inverse_randomized_hadamard

    result = inverse_randomized_hadamard(out_rotated, signs)
    return result.reshape(batch, n_q_heads, q_len, head_dim)


def _flash_decode_fallback(
    q_rotated,
    k_indices,
    k_norms,
    v_indices,
    v_norms,
    k_centroids,
    v_centroids,
    signs,
    scale,
    bits_k,
    bits_v,
):
    """CPU fallback: use separate QK + softmax + V matmul."""
    from fused_turboquant.kernels.triton_attention import fused_qk_scores_rht
    from fused_turboquant.kernels.triton_v_attention import _fused_v_attention_fallback

    scores = fused_qk_scores_rht(
        q_rotated,
        k_indices,
        k_norms,
        k_centroids,
        scale,
        bits=bits_k,
    )
    attn_weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
    n_kv_groups = q_rotated.shape[1] // v_indices.shape[1]
    return _fused_v_attention_fallback(
        attn_weights,
        v_indices,
        v_norms,
        v_centroids,
        signs,
        q_rotated.shape[-1],
        bits_v,
        n_kv_groups,
    )
