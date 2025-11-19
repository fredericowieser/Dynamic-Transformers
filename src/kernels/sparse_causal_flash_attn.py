import torch
import triton
import triton.language as tl

@triton.jit
def _sparse_fwd_kernel(
    Q, K, V, sm_scale,
    M, Out,
    Real_Indices,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_ib, stride_im,
    stride_mb, stride_mh, stride_mm,
    n_heads, context_len, n_selected,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    off_b = off_hz // n_heads
    off_h = off_hz % n_heads

    # -- 1. Initialize Pointers --
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    
    Q_ptr = (Q + (off_b * stride_qb + off_h * stride_qh).to(tl.int64) 
             + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk))
    
    # Fix: Ensure proper stride math for Indices
    Idx_ptr = Real_Indices + (off_b * stride_ib).to(tl.int64) + (offs_m * stride_im)
    
    K_base = K + (off_b * stride_kb + off_h * stride_kh).to(tl.int64)
    V_base = V + (off_b * stride_vb + off_h * stride_vh).to(tl.int64)
    
    # -- 2. Load Q and Indices --
    q_real_pos = tl.load(Idx_ptr, mask=offs_m < n_selected, other=-1)
    q = tl.load(Q_ptr, mask=offs_m[:, None] < n_selected, other=0.0)

    # -- 3. Initialize Accumulators --
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504

    # -- 4. Loop over K/V History --
    for start_n in range(0, context_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Load K (Transposed conceptually: [D, BLOCK_N])
        K_ptr = K_base + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk)
        k = tl.load(K_ptr, mask=offs_n[None, :] < context_len, other=0.0)
        
        # Compute QK^T
        qk = tl.dot(q, k)
        
        # Apply Sparse Causal Mask
        mask = q_real_pos[:, None] >= offs_n[None, :]
        qk = qk * qk_scale
        qk = tl.where(mask & (offs_n[None, :] < context_len), qk, float("-inf"))

        # Online Softmax
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])

        acc = acc * alpha[:, None]
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

        # Load V
        V_ptr = V_base + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        v = tl.load(V_ptr, mask=offs_n[:, None] < context_len, other=0.0)
        
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)

    # -- 5. Epilogue --
    l_i_reciprocal = 1.0 / l_i
    acc = acc * l_i_reciprocal[:, None]
    m_i += tl.math.log2(l_i)
    
    Out_ptr = (Out + (off_b * stride_ob + off_h * stride_oh).to(tl.int64) 
               + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok))
    
    # Fix: Ensure proper stride math for M
    M_ptr = M + (off_b * stride_mb + off_h * stride_mh).to(tl.int64) + (offs_m * stride_mm)
    
    tl.store(Out_ptr, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < n_selected)
    tl.store(M_ptr, m_i, mask=offs_m < n_selected)


@triton.jit
def _sparse_bwd_preprocess(
    Out, dO, D,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_db, stride_dh, stride_dm,
    n_heads, n_selected,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_b = off_hz // n_heads
    off_h = off_hz % n_heads
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    O_ptr = (Out + (off_b * stride_ob + off_h * stride_oh).to(tl.int64)
             + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok))
    dO_ptr = (dO + (off_b * stride_dob + off_h * stride_doh).to(tl.int64)
              + (offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok))
    
    # Fix: Stride for D
    D_ptr = D + (off_b * stride_db + off_h * stride_dh).to(tl.int64) + (offs_m * stride_dm)

    o = tl.load(O_ptr, mask=offs_m[:, None] < n_selected, other=0.0).to(tl.float32)
    do = tl.load(dO_ptr, mask=offs_m[:, None] < n_selected, other=0.0).to(tl.float32)
    
    d = tl.sum(o * do, axis=1)
    tl.store(D_ptr, d, mask=offs_m < n_selected)


@triton.jit
def _sparse_bwd_kernel(
    Q, K, V, sm_scale, dO, M, D, Real_Indices,
    dQ, dK, dV,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_mb, stride_mh, stride_mm,
    stride_db, stride_dh, stride_dm,
    stride_ib, stride_im,
    n_heads, context_len, n_selected,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    # -- PARALLELIZATION FIX --
    # We now parallelize over Q blocks (start_m) AND Batch/Heads
    start_m = tl.program_id(0) 
    off_hz = tl.program_id(1)
    
    off_b = off_hz // n_heads
    off_h = off_hz % n_heads

    # Base Pointers (Batch/Head offset)
    K_base = K + (off_b * stride_kb + off_h * stride_kh).to(tl.int64)
    V_base = V + (off_b * stride_vb + off_h * stride_vh).to(tl.int64)
    dK_base = dK + (off_b * stride_kb + off_h * stride_kh).to(tl.int64)
    dV_base = dV + (off_b * stride_vb + off_h * stride_vh).to(tl.int64)

    Q_base = Q + (off_b * stride_qb + off_h * stride_qh).to(tl.int64)
    dO_base = dO + (off_b * stride_dob + off_h * stride_doh).to(tl.int64)
    dQ_base = dQ + (off_b * stride_qb + off_h * stride_qh).to(tl.int64)
    M_base = M + (off_b * stride_mb + off_h * stride_mh).to(tl.int64)
    D_base = D + (off_b * stride_db + off_h * stride_dh).to(tl.int64)
    Idx_base = Real_Indices + (off_b * stride_ib).to(tl.int64)

    qk_scale = sm_scale * 1.44269504

    # -- BLOCK SETUP (No outer loop over M anymore) --
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Load Q, dO, M, D, Indices
    q_ptr = Q_base + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    do_ptr = dO_base + (offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok)
    
    # FIX: Added strides to m, d, and idx pointers
    m_ptr = M_base + offs_m * stride_mm
    d_ptr = D_base + offs_m * stride_dm
    idx_ptr = Idx_base + offs_m * stride_im

    q = tl.load(q_ptr, mask=offs_m[:, None] < n_selected, other=0.0)
    do = tl.load(do_ptr, mask=offs_m[:, None] < n_selected, other=0.0)
    m = tl.load(m_ptr, mask=offs_m < n_selected, other=0.0)
    d = tl.load(d_ptr, mask=offs_m < n_selected, other=0.0)
    q_real_pos = tl.load(idx_ptr, mask=offs_m < n_selected, other=-1)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Loop over K/V blocks
    for start_n in range(0, context_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Load K, V
        k_ptr = K_base + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk)
        v_ptr = V_base + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        
        k = tl.load(k_ptr, mask=offs_n[None, :] < context_len, other=0.0)
        v = tl.load(v_ptr, mask=offs_n[:, None] < context_len, other=0.0)

        # Recompute Attention
        qk = tl.dot(q, k)
        mask = q_real_pos[:, None] >= offs_n[None, :]
        qk = tl.where(mask & (offs_n[None, :] < context_len), qk, float("-inf"))
        
        p = tl.math.exp2(qk * qk_scale - m[:, None])
        p = tl.where(mask & (offs_n[None, :] < context_len), p, 0.0)

        # Compute dV and dK
        p = p.to(do.dtype)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - d[:, None])
        ds = ds.to(do.dtype)
        
        # dV Accumulation (Atomic)
        dv = tl.dot(tl.trans(p), do)
        dv_ptr = dV_base + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        tl.atomic_add(dv_ptr, dv, mask=offs_n[:, None] < context_len)
        
        # dK Accumulation (Atomic)
        dk = tl.dot(tl.trans(ds), q)
        dk_ptr = dK_base + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        tl.atomic_add(dk_ptr, dk, mask=offs_n[:, None] < context_len)

        # dQ Accumulation (Register)
        dq += tl.dot(ds, tl.trans(k))

    # Write back dQ
    dq_ptr = dQ_base + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    tl.store(dq_ptr, dq, mask=offs_m[:, None] < n_selected)


class SparseCausalAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_selected, k_full, v_full, real_indices, sm_scale):
        # Ensure contiguity for safety with Triton, though we pass strides
        q = q_selected.transpose(1, 2).contiguous()
        k = k_full.transpose(1, 2).contiguous()
        v = v_full.transpose(1, 2).contiguous()
        real_indices = real_indices.contiguous()

        B, H, N_SEL, D = q.shape
        _, _, N_CTX, _ = k.shape
        
        o = torch.empty_like(q)
        M = torch.empty((B, H, N_SEL), device=q.device, dtype=torch.float32)

        grid = (triton.cdiv(N_SEL, 128), B * H)
        
        _sparse_fwd_kernel[grid](
            q, k, v, sm_scale,
            M, o,
            real_indices,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            real_indices.stride(0), real_indices.stride(1),
            M.stride(0), M.stride(1), M.stride(2),
            H, N_CTX, N_SEL,
            BLOCK_M=128, BLOCK_N=64, HEAD_DIM=D,
            num_warps=4, num_stages=2
        )

        ctx.save_for_backward(q, k, v, o, M, real_indices)
        ctx.sm_scale = sm_scale
        return o.transpose(1, 2)

    @staticmethod
    def backward(ctx, do_selected):
        do = do_selected.transpose(1, 2).contiguous()
        q, k, v, o, M, real_indices = ctx.saved_tensors
        
        B, H, N_SEL, D = q.shape
        _, _, N_CTX, _ = k.shape

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        D_pre = torch.empty((B, H, N_SEL), device=q.device, dtype=torch.float32)

        # Preprocess D
        grid_pre = (triton.cdiv(N_SEL, 128), B * H)
        _sparse_bwd_preprocess[grid_pre](
            o, do, D_pre,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            D_pre.stride(0), D_pre.stride(1), D_pre.stride(2),
            H, N_SEL,
            BLOCK_M=128, HEAD_DIM=D
        )

        # Backward Kernel
        # FIX: Grid must be (N_SEL // BLOCK_M, B * H)
        grid_bwd = (triton.cdiv(N_SEL, 64), B * H)
        
        _sparse_bwd_kernel[grid_bwd](
            q, k, v, ctx.sm_scale, do, M, D_pre, real_indices,
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            M.stride(0), M.stride(1), M.stride(2),
            D_pre.stride(0), D_pre.stride(1), D_pre.stride(2),
            real_indices.stride(0), real_indices.stride(1),
            H, N_CTX, N_SEL,
            BLOCK_M=64, BLOCK_N=64, HEAD_DIM=D,
            num_warps=4, num_stages=1
        )
        
        return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2), None, None

sparse_causal_attention = SparseCausalAttention.apply
