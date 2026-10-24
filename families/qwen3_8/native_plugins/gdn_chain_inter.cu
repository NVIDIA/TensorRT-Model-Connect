// Qwen3.8 GDN prefill: state chain over the chunks (the sequential part of the delta rule).
//
// One CTA per (value head, 64-wide V tile). The state is kept transposed (S^T: 64 V rows x 128 K
// columns) in mma accumulators; each of the 4 warps owns 16 V rows, so warps never exchange data.
// The operands of the next chunk are prefetched with cp.async into a second shared-memory stage
// while the current chunk is computed. Matmul operands are fp16 with fp32 accumulation, like the
// cuTile kernels. Compiled with NVRTC for the device's architecture when the plan is built.
// GDN prefill state chain for one (value head, 64-wide V tile) per CTA.
// State is kept transposed (S^T: 64 V rows x 128 K cols) in mma accumulators; each of the 4 warps
// owns 16 V rows, so warps never exchange data. Operands of the next chunk are prefetched with
// cp.async into a second shared-memory stage while the current chunk is computed.
#define CS 32
#define KD 128
#define VD 128
#define BV 64
#define LDKQ 136   // halfs per KQ row in smem (KD + 8: conflict-free fragment loads)
#define LDW 40     // halfs per K_w^T / QK_mask row in smem (CS + 8)
#define LDV 68     // floats per V_corr row in smem (BV + 4)

typedef unsigned short u16;
typedef unsigned int u32;

#ifndef OUT_BF16
#define OUT_BF16 0
#endif

__device__ __forceinline__ u32 pack2(float lo, float hi) {
    u32 d;
    asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(d) : "f"(lo), "f"(hi));
    return d;
}
__device__ __forceinline__ void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(void* smem_dst, const void* gsrc) {
    u32 s = (u32)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gsrc));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
// Output activation dtype: fp16, or bf16 when OUT_BF16 is set.
__device__ __forceinline__ u16 to_half(float x) {
    u16 r;
#if OUT_BF16
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
#else
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(x));
#endif
    return r;
}

#define STAGE_KQ (64 * LDKQ * 2)
#define STAGE_KW (KD * LDW * 2)
#define STAGE_QM (CS * LDW * 2)
#define STAGE_VC (CS * LDV * 4)
#define STAGE_G (CS * 4)
#define STAGE_BYTES (STAGE_KQ + STAGE_KW + STAGE_QM + STAGE_VC + STAGE_G)

__device__ __forceinline__ void prefetch(unsigned char* stage, int tid, int h, int pv, int c, int NC,
                                         const u16* KQ, const u16* KwT, const u16* QKM, const float* Vc,
                                         const float* Gc) {
    u16* kqs = (u16*)stage;
    u16* kws = (u16*)(stage + STAGE_KQ);
    u16* qms = (u16*)(stage + STAGE_KQ + STAGE_KW);
    float* vcs = (float*)(stage + STAGE_KQ + STAGE_KW + STAGE_QM);
    float* gs = (float*)(stage + STAGE_KQ + STAGE_KW + STAGE_QM + STAGE_VC);
    size_t hc = (size_t)h * NC + c;
    for (int i = tid; i < 64 * 16; i += 128) {
        int row = i >> 4, cc = i & 15;
        cp16(kqs + row * LDKQ + cc * 8, KQ + (hc * 64 + row) * KD + cc * 8);
    }
    for (int i = tid; i < KD * 4; i += 128) {
        int row = i >> 2, cc = i & 3;
        cp16(kws + row * LDW + cc * 8, KwT + (hc * KD + row) * CS + cc * 8);
    }
    for (int i = tid; i < CS * 4; i += 128) {
        int row = i >> 2, cc = i & 3;
        cp16(qms + row * LDW + cc * 8, QKM + (hc * CS + row) * CS + cc * 8);
    }
    for (int i = tid; i < CS * 16; i += 128) {
        int row = i >> 4, cc = i & 15;
        cp16(vcs + row * LDV + cc * 4, Vc + (hc * CS + row) * VD + pv * BV + cc * 4);
    }
    if (tid < 8) cp16(gs + tid * 4, Gc + hc * CS + tid * 4);
}

extern "C" __global__ void __launch_bounds__(128, 1)
gdn_inter(const u16* __restrict__ KQ,   // [H, NC, 2*CS, KD] fp16: rows 0..CS-1 k_cumdecay, CS.. q*exp(g)
          const u16* __restrict__ KwT,  // [H, NC, KD, CS] fp16: (k*exp(g_last-g))^T
          const u16* __restrict__ QKM,  // [H, NC, CS, CS] fp16: causal decayed q k^T
          const float* __restrict__ Vc, // [H, NC, CS, VD] fp32 corrected values
          const float* __restrict__ Gc, // [H, NC, CS] fp32 cumulative decay
          u16* __restrict__ Out,        // [T, H, VD] fp16 or bf16 (OUT_BF16)
          const float* __restrict__ S0, // [H, KD, VD] fp32
          float* __restrict__ S1,       // [H, KD, VD] fp32
          int NC, int T, int H) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int h = blockIdx.x, pv = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
    const int vbase = pv * BV + warp * 16;  // first V row of this warp

    float acc[16][4];  // S^T[16 V rows][128 K cols] in C-fragment layout, 16 n-tiles
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            int v = vbase + g + 8 * (e >> 1), k = j * 8 + 2 * tig + (e & 1);
            acc[j][e] = S0[((size_t)h * KD + k) * VD + v];
        }

    prefetch(smem, tid, h, pv, 0, NC, KQ, KwT, QKM, Vc, Gc);
    commit();

    for (int c = 0; c < NC; ++c) {
        unsigned char* cur = smem + (c & 1) * STAGE_BYTES;
        if (c + 1 < NC) {
            prefetch(smem + ((c + 1) & 1) * STAGE_BYTES, tid, h, pv, c + 1, NC, KQ, KwT, QKM, Vc, Gc);
            commit();
            wait_group<1>();
        } else {
            wait_group<0>();
        }
        __syncthreads();
        const u16* kqs = (const u16*)cur;
        const u16* kws = (const u16*)(cur + STAGE_KQ);
        const u16* qms = (const u16*)(cur + STAGE_KQ + STAGE_KW);
        const float* vcs = (const float*)(cur + STAGE_KQ + STAGE_KW + STAGE_QM);
        const float* gs = (const float*)(cur + STAGE_KQ + STAGE_KW + STAGE_QM + STAGE_VC);

        // sa^T[16 V][64] = S^T[16][128] @ [k_cumdecay; q*exp(g)]^T
        float sa[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) sa[j][e] = 0.f;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            u32 a0 = pack2(acc[2 * ks][0], acc[2 * ks][1]);
            u32 a1 = pack2(acc[2 * ks][2], acc[2 * ks][3]);
            u32 a2 = pack2(acc[2 * ks + 1][0], acc[2 * ks + 1][1]);
            u32 a3 = pack2(acc[2 * ks + 1][2], acc[2 * ks + 1][3]);
#pragma unroll
            for (int jn = 0; jn < 8; ++jn) {
                const u16* bp = kqs + (jn * 8 + g) * LDKQ + ks * 16 + 2 * tig;
                u32 b0 = *(const u32*)bp, b1 = *(const u32*)(bp + 8);
                mma16816(sa[jn], a0, a1, a2, a3, b0, b1);
            }
        }
        // v_new^T = v_corr^T - (k_cumdecay S)^T  (tiles 0..3 of sa)
        float vn[4][4];
#pragma unroll
        for (int jn = 0; jn < 4; ++jn)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                int vl = warp * 16 + g + 8 * (e >> 1), t = jn * 8 + 2 * tig + (e & 1);
                vn[jn][e] = vcs[t * LDV + vl] - sa[jn][e];
            }
        u32 va[2][4];
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            va[s][0] = pack2(vn[2 * s][0], vn[2 * s][1]);
            va[s][1] = pack2(vn[2 * s][2], vn[2 * s][3]);
            va[s][2] = pack2(vn[2 * s + 1][0], vn[2 * s + 1][1]);
            va[s][3] = pack2(vn[2 * s + 1][2], vn[2 * s + 1][3]);
        }
        // out^T = (q*exp(g) S)^T + v_new^T @ QK_mask^T
        float oc[4][4];
#pragma unroll
        for (int jn = 0; jn < 4; ++jn)
#pragma unroll
            for (int e = 0; e < 4; ++e) oc[jn][e] = sa[4 + jn][e];
#pragma unroll
        for (int s = 0; s < 2; ++s)
#pragma unroll
            for (int jn = 0; jn < 4; ++jn) {
                const u16* bp = qms + (jn * 8 + g) * LDW + s * 16 + 2 * tig;
                mma16816(oc[jn], va[s][0], va[s][1], va[s][2], va[s][3], *(const u32*)bp, *(const u32*)(bp + 8));
            }
#pragma unroll
        for (int jn = 0; jn < 4; ++jn)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                int v = vbase + g + 8 * (e >> 1), t = c * CS + jn * 8 + 2 * tig + (e & 1);
                if (t < T) Out[((size_t)t * H + h) * VD + v] = to_half(oc[jn][e]);
            }
        // S^T = S^T * exp(g_last) + v_new^T @ K_w
        const float eg = expf(gs[CS - 1]);
#pragma unroll
        for (int j = 0; j < 16; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[j][e] *= eg;
#pragma unroll
        for (int s = 0; s < 2; ++s)
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const u16* bp = kws + (j * 8 + g) * LDW + s * 16 + 2 * tig;
                mma16816(acc[j], va[s][0], va[s][1], va[s][2], va[s][3], *(const u32*)bp, *(const u32*)(bp + 8));
            }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            int v = vbase + g + 8 * (e >> 1), k = j * 8 + 2 * tig + (e & 1);
            S1[((size_t)h * KD + k) * VD + v] = acc[j][e];
        }
}
