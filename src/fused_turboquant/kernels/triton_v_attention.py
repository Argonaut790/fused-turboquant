"""
Fused Triton kernel for attention-weighted value sum from compressed V cache.

Instead of: decompress ALL V to fp16 -> attn_weights @ V  (O(seq_len * head_dim) memory)
We do:      for each output dim: unpack V on-the-fly, multiply by attn_weight, accumulate

This eliminates the O(seq_len * head_dim) intermediate dense V tensor, reducing
both memory bandwidth and peak memory during decode.
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

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_S": 32, "BLOCK_D": 64}, num_warps=4),
            triton.Config({"BLOCK_S": 64, "BLOCK_D": 64}, num_warps=4),
            triton.Config({"BLOCK_S": 128, "BLOCK_D": 64}, num_warps=8),
            triton.Config({"BLOCK_S": 64, "BLOCK_D": 128}, num_warps=4),
            triton.Config({"BLOCK_S": 128, "BLOCK_D": 128}, num_warps=8),
        ],
        key=["head_dim"],
    )
    @triton.jit
    def _fused_v_attention_kernel(
        W_ptr,  # attn weights: [BH_q, seq_len] float
        V_idx_ptr,  # packed V indices: [BH_kv, seq_len, packed_dim] uint8
        V_norms_ptr,  # V norms: [BH_kv, seq_len] fp16
        C_ptr,  # centroid table: [n_levels] float32
        Signs_ptr,  # RHT signs: [head_dim] float32
        Out_ptr,  # output: [BH_q, head_dim] float32
        Scratch_ptr,  # scratch for iFFWT: [BH_q, head_dim] float32
        seq_len,
        head_dim: tl.constexpr,
        LOG2_D: tl.constexpr,
        n_q_heads,
        n_kv_heads,
        stride_w_bh,
        stride_w_s,
        stride_vi_bh,
        stride_vi_s,
        stride_vi_d,
        stride_vn_bh,
        stride_vn_s,
        stride_o_bh,
        stride_o_d,
        BITS: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Compute attn_weights @ V directly from compressed V, with inverse RHT."""
        pid_bh = tl.program_id(0)
        pid_d = tl.program_id(1)

        batch_idx = pid_bh // n_q_heads
        q_head_idx = pid_bh % n_q_heads
        gqa_ratio = n_q_heads // n_kv_heads
        kv_head_idx = q_head_idx // gqa_ratio
        kv_bh = batch_idx * n_kv_heads + kv_head_idx

        d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offs < head_dim

        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for s_start in range(0, seq_len, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            s_mask = s_offs < seq_len

            w_ptrs = W_ptr + pid_bh * stride_w_bh + s_offs * stride_w_s
            w_vals = tl.load(w_ptrs, mask=s_mask, other=0.0).to(tl.float32)

            vn_ptrs = V_norms_ptr + kv_bh * stride_vn_bh + s_offs * stride_vn_s
            v_norms = tl.load(vn_ptrs, mask=s_mask, other=0).to(tl.float32)

            if BITS == 4:
                pack_d = d_offs // 2
                vi_ptrs = (
                    V_idx_ptr
                    + kv_bh * stride_vi_bh
                    + s_offs[:, None] * stride_vi_s
                    + pack_d[None, :] * stride_vi_d
                )
                combined_mask = s_mask[:, None] & d_mask[None, :]
                packed_val = tl.load(vi_ptrs, mask=combined_mask, other=0).to(tl.int32)
                is_low = (d_offs % 2) == 0
                v_idx = tl.where(is_low[None, :], packed_val & 0xF, (packed_val >> 4) & 0xF)
            elif BITS == 3:
                bit_off = d_offs * 3
                byte_idx = bit_off >> 3
                bit_shift = bit_off & 7
                packed_total = head_dim * 3 // 8
                combined_mask = s_mask[:, None] & d_mask[None, :]
                vi_base = V_idx_ptr + kv_bh * stride_vi_bh
                b0_ptrs = vi_base + s_offs[:, None] * stride_vi_s + byte_idx[None, :] * stride_vi_d
                b1_ptrs = (
                    vi_base + s_offs[:, None] * stride_vi_s + (byte_idx + 1)[None, :] * stride_vi_d
                )
                b0 = tl.load(b0_ptrs, mask=combined_mask, other=0).to(tl.int32)
                b1_valid = (byte_idx + 1) < packed_total
                b1 = tl.load(
                    b1_ptrs,
                    mask=combined_mask & b1_valid[None, :],
                    other=0,
                ).to(tl.int32)
                v_idx = ((b0 | (b1 << 8)) >> bit_shift[None, :]) & 0x7
            elif BITS == 2:
                pack_d = d_offs // 4
                vi_ptrs = (
                    V_idx_ptr
                    + kv_bh * stride_vi_bh
                    + s_offs[:, None] * stride_vi_s
                    + pack_d[None, :] * stride_vi_d
                )
                combined_mask = s_mask[:, None] & d_mask[None, :]
                packed_val = tl.load(vi_ptrs, mask=combined_mask, other=0).to(tl.int32)
                shift = (d_offs % 4) * 2
                v_idx = (packed_val >> shift[None, :]) & 0x3
            else:
                vi_ptrs = (
                    V_idx_ptr
                    + kv_bh * stride_vi_bh
                    + s_offs[:, None] * stride_vi_s
                    + d_offs[None, :] * stride_vi_d
                )
                combined_mask = s_mask[:, None] & d_mask[None, :]
                v_idx = tl.load(vi_ptrs, mask=combined_mask, other=0).to(tl.int32)

            v_centroids = tl.load(C_ptr + v_idx, mask=combined_mask, other=0.0).to(tl.float32)

            weighted = (w_vals * v_norms)[:, None] * v_centroids  # [BLOCK_S, BLOCK_D]
            acc += tl.sum(weighted, axis=0)  # [BLOCK_D]

        # acc is now the weighted sum in rotated space, need inverse RHT
        # Store to scratch, do iFFWT, then sign flip
        scratch_base = pid_bh * head_dim
        tl.store(Scratch_ptr + scratch_base + d_offs, acc, mask=d_mask)

    def _inverse_rht_from_scratch(
        scratch: torch.Tensor,
        signs: torch.Tensor,
        head_dim: int,
    ) -> torch.Tensor:
        """Apply inverse RHT to scratch buffer rows."""
        from fused_turboquant.core.hadamard import inverse_randomized_hadamard

        return inverse_randomized_hadamard(scratch, signs)


def fused_v_attention(
    attn_weights: torch.Tensor,  # [batch, n_q_heads, 1, kv_len] post-softmax
    v_indices: torch.Tensor,  # [batch, n_kv_heads, kv_len, packed_dim] uint8
    v_norms: torch.Tensor,  # [batch, n_kv_heads, kv_len] fp16
    centroids: torch.Tensor,  # [n_levels] float32
    signs: torch.Tensor,  # [head_dim] float32 RHT signs
    head_dim: int,
    bits: int = 4,
) -> torch.Tensor:
    """Compute attention output from compressed V without full decompression.

    Instead of materializing the full dense V matrix [batch, heads, seq_len, head_dim],
    this computes the weighted centroid sum in rotated space, then applies inverse RHT.

    Fallback (non-Triton) path provided for CPU/testing.

    Returns: [batch, n_q_heads, 1, head_dim] attention output.
    """
    batch, n_q_heads, q_len, kv_len = attn_weights.shape
    _, n_kv_heads, _, packed_dim = v_indices.shape
    n_kv_groups = n_q_heads // n_kv_heads

    w_flat = attn_weights.reshape(batch * n_q_heads * q_len, kv_len)

    if not HAS_TRITON or not attn_weights.is_cuda:
        return _fused_v_attention_fallback(
            attn_weights,
            v_indices,
            v_norms,
            centroids,
            signs,
            head_dim,
            bits,
            n_kv_groups,
        )

    vi_flat = v_indices.reshape(batch * n_kv_heads, kv_len, packed_dim).contiguous()
    vn_flat = v_norms.reshape(batch * n_kv_heads, kv_len).contiguous()
    centroids = centroids.contiguous().float()

    total_q = batch * n_q_heads * q_len
    scratch = torch.zeros(total_q, head_dim, device=attn_weights.device, dtype=torch.float32)
    out = torch.empty(total_q, head_dim, device=attn_weights.device, dtype=torch.float32)

    log2_d = (
        head_dim.bit_length() - 1 if isinstance(head_dim, int) else int(head_dim).bit_length() - 1
    )
    effective_q_heads = n_q_heads * q_len

    def grid(meta):
        import triton

        return (batch * effective_q_heads, triton.cdiv(head_dim, meta["BLOCK_D"]))

    _fused_v_attention_kernel[grid](
        w_flat,
        vi_flat,
        vn_flat,
        centroids,
        signs,
        out,
        scratch,
        kv_len,
        head_dim,
        log2_d,
        effective_q_heads,
        n_kv_heads,
        w_flat.stride(0),
        w_flat.stride(1),
        vi_flat.stride(0),
        vi_flat.stride(1),
        vi_flat.stride(2),
        vn_flat.stride(0),
        vn_flat.stride(1),
        out.stride(0),
        out.stride(1),
        BITS=bits,
    )

    # The kernel wrote the weighted sum in rotated space to scratch.
    # Apply inverse RHT to get back to original space.
    result = _inverse_rht_from_scratch(scratch, signs, head_dim)
    return result.reshape(batch, n_q_heads, q_len, head_dim)


def _fused_v_attention_fallback(
    attn_weights: torch.Tensor,
    v_indices: torch.Tensor,
    v_norms: torch.Tensor,
    centroids: torch.Tensor,
    signs: torch.Tensor,
    head_dim: int,
    bits: int,
    n_kv_groups: int,
) -> torch.Tensor:
    """CPU fallback: decode V fully then matmul."""
    from fused_turboquant.core.quantizer import CompressedTensor, TurboQuantMSE

    batch, n_kv_heads, kv_len, packed_dim = v_indices.shape
    n_q_heads = attn_weights.shape[1]

    ct = CompressedTensor(
        indices=v_indices,
        norms=v_norms,
        original_dim=head_dim,
        bits=bits,
    )
    tq = TurboQuantMSE(head_dim=head_dim, bits=bits)
    decoded_v = tq.decode(ct)

    if n_kv_groups > 1:
        decoded_v = (
            decoded_v[:, :, None, :, :]
            .expand(
                batch,
                n_kv_heads,
                n_kv_groups,
                kv_len,
                head_dim,
            )
            .reshape(batch, n_q_heads, kv_len, head_dim)
        )

    return torch.matmul(attn_weights, decoded_v.to(attn_weights.dtype))
