from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def flash_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        L_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_ob,
        stride_oq,
        stride_od,
        stride_lb,
        stride_lq,
        N_QUERIES,
        N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        q = tl.load(Q_block_ptr)
        m = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
        normalizer = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        output_accum = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

        for k_start in range(0, N_KEYS, K_TILE_SIZE):
            k = tl.load(K_block_ptr)
            v = tl.load(V_block_ptr)

            scores = tl.dot(q, tl.trans(k)) * scale
            if is_causal:
                k_offsets = k_start + tl.arange(0, K_TILE_SIZE)
                scores = tl.where(q_offsets[:, None] >= k_offsets[None, :], scores, -1e6)

            block_m = tl.max(scores, axis=1)
            m_new = tl.maximum(m, block_m)
            exp_scale = tl.exp(m - m_new)
            p = tl.exp(scores - m_new[:, None])
            normalizer_new = exp_scale * normalizer + tl.sum(p, axis=1)

            output_accum = exp_scale[:, None] * output_accum + tl.dot(p.to(v.dtype), v)
            m = m_new
            normalizer = normalizer_new
            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        output = output_accum / normalizer[:, None]
        tl.store(O_block_ptr, output.to(O_block_ptr.type.element_ty))
        tl.store(L_ptr + batch_index * stride_lb + q_offsets * stride_lq, m + tl.log(normalizer))

else:
    flash_fwd_kernel = None


class FlashAttention2PyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[-2]
        scale = 1.0 / math.sqrt(d)

        block_q = 16
        block_k = 16
        output = torch.empty_like(Q)
        L = torch.empty((*Q.shape[:-1],), dtype=torch.float32, device=Q.device)

        for q_start in range(0, n_queries, block_q):
            q_end = q_start + block_q
            q = Q[:, q_start:q_end, :]

            m = torch.full((batch_size, block_q), float("-inf"), dtype=torch.float32, device=Q.device)
            normalizer = torch.zeros((batch_size, block_q), dtype=torch.float32, device=Q.device)
            output_accum = torch.zeros((batch_size, block_q, V.shape[-1]), dtype=torch.float32, device=Q.device)

            for k_start in range(0, n_keys, block_k):
                k_end = k_start + block_k
                k = K[:, k_start:k_end, :]
                v = V[:, k_start:k_end, :]

                scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
                if is_causal:
                    q_offsets = torch.arange(q_start, q_end, device=Q.device)[:, None]
                    k_offsets = torch.arange(k_start, k_end, device=Q.device)[None, :]
                    scores = scores.masked_fill(q_offsets < k_offsets, -1e6)

                block_m = scores.max(dim=-1).values
                m_new = torch.maximum(m, block_m)
                exp_scale = torch.exp(m - m_new)
                p = torch.exp(scores - m_new[..., None])
                normalizer_new = exp_scale * normalizer + p.sum(dim=-1)

                output_accum = exp_scale[..., None] * output_accum + torch.matmul(p, v.float())
                m = m_new
                normalizer = normalizer_new

            output[:, q_start:q_end, :] = (output_accum / normalizer[..., None]).to(dtype=Q.dtype)
            L[:, q_start:q_end] = m + torch.log(normalizer)

        ctx.save_for_backward(L, Q, K, V, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        raise NotImplementedError


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        if triton is None or flash_fwd_kernel is None:
            raise RuntimeError("Triton is not available in this environment.")
        if not Q.is_cuda:
            raise RuntimeError("FlashAttention2Triton requires CUDA tensors.")

        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[-2]
        block_q = 16
        block_k = 16

        output = torch.empty_like(Q)
        L = torch.empty((*Q.shape[:-1],), dtype=torch.float32, device=Q.device)
        grid = (triton.cdiv(n_queries, block_q), batch_size)

        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            output,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            L.stride(0),
            L.stride(1),
            n_queries,
            n_keys,
            1.0 / math.sqrt(d),
            D=d,
            Q_TILE_SIZE=block_q,
            K_TILE_SIZE=block_k,
            is_causal=is_causal,
        )

        ctx.save_for_backward(L, Q, K, V, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        raise NotImplementedError
