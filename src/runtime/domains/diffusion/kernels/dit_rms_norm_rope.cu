// SPDX-License-Identifier: Apache-2.0
//
// Per-tensor fused RMSNorm + RoPE for DiT attention.
//
// Adapted from TensorRT-LLM PR #13052 (fusedDiTQKNormRopeKernel.cu, commit
// f7fadf1b84a1bbda53d0900ec6eff39980384fff). Upstream targets a packed QKV
// layout with dual-stream text/image norm. We simplify to a per-tensor
// launcher matching our FLUX.2 builder, which holds Q/K/V as separate
// ITensors and processes image and text branches independently.
//
// Per-warp algorithm (one warp processes one (token, head)):
//   1. Vectorized half-precision load + per-thread sum-of-squares
//   2. Warp-shuffle reduce → RMS scale → per-head weight multiply
//   3. RoPE (interleaved pairing, fp32 cos/sin)
//   4. Vectorized half-precision store
//
// Two TVM-FFI exports — bf16 and fp16. The FLUX.2 builder picks the one
// matching its `_CAST_DTYPE` (fp16 is the default; bf16 is opt-in).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>

namespace trtmc::diffusion::kernels {

// Inlined from TensorRT-LLM common/reduceKernelUtils.cuh (Apache-2.0).
// Specialized to float since the only caller reduces a fp32 accumulator.
__forceinline__ __device__ float warpReduceSumF32(float val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask, 32);
    }
    return val;
}

// Maps (uint, vecSize) → uint / uint2 / uint4 for vectorized I/O.
// Inlined from TensorRT-LLM common::packed_as.
template <int N> struct PackedUint;
template <> struct PackedUint<1> { using type = unsigned int; };
template <> struct PackedUint<2> { using type = uint2; };
template <> struct PackedUint<4> { using type = uint4; };

// Per-dtype intrinsics. Each specialization defines the 2-wide packed type
// + the float2 unpack/pack helpers + the scalar to-float helper.
template <typename T> struct DTypeTraits;

template <> struct DTypeTraits<__nv_bfloat16> {
    using T2 = __nv_bfloat162;
    __forceinline__ __device__ static float2 to_float2(T2 v) {
        return __bfloat1622float2(v);
    }
    __forceinline__ __device__ static T2 from_float2(float2 v) {
        return __float22bfloat162_rn(v);
    }
    __forceinline__ __device__ static float to_float(__nv_bfloat16 v) {
        return __bfloat162float(v);
    }
};

template <> struct DTypeTraits<__half> {
    using T2 = __half2;
    __forceinline__ __device__ static float2 to_float2(T2 v) {
        return __half22float2(v);
    }
    __forceinline__ __device__ static T2 from_float2(float2 v) {
        return __float22half2_rn(v);
    }
    __forceinline__ __device__ static float to_float(__half v) {
        return __half2float(v);
    }
};

// One warp per (token, head). x_in and x_out are [num_tokens, num_heads * head_dim].
template <typename T, int head_dim>
__global__ void perTensorRMSNormRopeKernel(
    T const* __restrict__ x_in,
    T* __restrict__ x_out,
    T const* __restrict__ weight,                // [head_dim]
    float const* __restrict__ cos_emb,           // [num_tokens, head_dim]
    float const* __restrict__ sin_emb,           // [num_tokens, head_dim]
    int num_tokens, int num_heads, float eps)
{
    using Traits = DTypeTraits<T>;
    using T2 = typename Traits::T2;
    constexpr int warpSize = 32;
    int const warpsPerBlock = blockDim.x / warpSize;
    int const warpId = threadIdx.x / warpSize;
    int const laneId = threadIdx.x % warpSize;
    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

    int const tokenIdx = globalWarpIdx / num_heads;
    int const headIdx = globalWarpIdx % num_heads;
    if (tokenIdx >= num_tokens) return;

    static_assert(head_dim % 64 == 0,
                  "head_dim must be divisible by 64 (each warp lane processes ≥2 elems)");
    constexpr int numElemsPerThread = head_dim / warpSize;
    constexpr int elemSizeBytes = numElemsPerThread * sizeof(T);
    static_assert(elemSizeBytes % 4 == 0, "elemSizeBytes must be a multiple of 4");
    constexpr int vecSize = elemSizeBytes / 4;
    using vec_T = typename PackedUint<vecSize>::type;

    int64_t const offsetWarp =
        static_cast<int64_t>(tokenIdx) * num_heads * head_dim + headIdx * head_dim;
    int64_t const offsetThread = offsetWarp + laneId * numElemsPerThread;

    float elements[numElemsPerThread];

    // Step 1: vectorized load + sum-of-squares
    float sumSq = 0.f;
    {
        vec_T vec = *reinterpret_cast<vec_T const*>(&x_in[offsetThread]);
        unsigned int* vecAsUint = reinterpret_cast<unsigned int*>(&vec);
        for (int i = 0; i < vecSize; i++) {
            float2 v = Traits::to_float2(*reinterpret_cast<T2*>(vecAsUint + i));
            sumSq += v.x * v.x + v.y * v.y;
            elements[2 * i] = v.x;
            elements[2 * i + 1] = v.y;
        }
    }

    // Step 2: RMS normalize + per-head weight
    sumSq = warpReduceSumF32(sumSq);
    float const rms_rcp = rsqrtf(sumSq / static_cast<float>(head_dim) + eps);
    for (int i = 0; i < numElemsPerThread; i++) {
        int const dim = laneId * numElemsPerThread + i;
        elements[i] *= rms_rcp * Traits::to_float(weight[dim]);
    }

    // Step 3: interleaved RoPE
    int64_t const embOffset = static_cast<int64_t>(tokenIdx) * head_dim;
    for (int i = 0; i < numElemsPerThread; i += 2) {
        int const dim = laneId * numElemsPerThread + i;
        float const cos0 = cos_emb[embOffset + dim];
        float const sin0 = sin_emb[embOffset + dim];
        float const cos1 = cos_emb[embOffset + dim + 1];
        float const sin1 = sin_emb[embOffset + dim + 1];
        float const x = elements[i];
        float const y = elements[i + 1];
        elements[i]     = x * cos0 - y * sin0;
        elements[i + 1] = y * cos1 + x * sin1;
    }

    // Step 4: vectorized store
    {
        vec_T vec;
        unsigned int* vecAsUint = reinterpret_cast<unsigned int*>(&vec);
        for (int i = 0; i < vecSize; i++) {
            T2 v = Traits::from_float2(
                make_float2(elements[2 * i], elements[2 * i + 1]));
            *reinterpret_cast<T2*>(vecAsUint + i) = v;
        }
        *reinterpret_cast<vec_T*>(&x_out[offsetThread]) = vec;
    }
}

template <typename T>
static void launchPerTensorRMSNormRope(
    T const* x_in, T* x_out, T const* weight,
    float const* cos_emb, float const* sin_emb,
    int num_tokens, int num_heads, int head_dim, float eps,
    cudaStream_t stream)
{
    constexpr int blockSize = 256;
    int const warpsPerBlock = blockSize / 32;
    int const totalWarps = num_tokens * num_heads;
    int const gridSize = (totalWarps + warpsPerBlock - 1) / warpsPerBlock;
    dim3 const gridDim(gridSize), blockDim(blockSize);

#define LAUNCH(HEAD_DIM)                                                            \
    perTensorRMSNormRopeKernel<T, HEAD_DIM>                                         \
        <<<gridDim, blockDim, 0, stream>>>(x_in, x_out, weight, cos_emb, sin_emb,   \
                                            num_tokens, num_heads, eps);

    switch (head_dim) {
        case 64:  LAUNCH(64);  break;
        case 128: LAUNCH(128); break;
        case 256: LAUNCH(256); break;
        default: /* Unsupported; silent no-op. Callers must guard. */ break;
    }
#undef LAUNCH
}

} // namespace trtmc::diffusion::kernels

// ---------------------------------------------------------------------------
// TVM-FFI export — one symbol per dtype
// ---------------------------------------------------------------------------
//
// Arg layout (destination-passing, matching tvm_ffi_kernel_plugin.cpp's
// "[inputs..., outputs..., extras...]" convention):
//   args[0] = x_in DLTensor    [num_tokens, num_heads * head_dim] T
//   args[1] = weight DLTensor  [head_dim]                          T
//   args[2] = cos_emb DLTensor [num_tokens, head_dim]              fp32
//   args[3] = sin_emb DLTensor [num_tokens, head_dim]              fp32
//   args[4] = x_out DLTensor   [num_tokens, num_heads * head_dim] T  (preallocated)
//   args[5] = num_heads (int)
//   args[6] = head_dim  (int)
//   args[7] = eps (float; stored via v_float64 by the plugin)
//
// T is determined by which global name is looked up:
//   "trtmc.dit_rms_norm_rope_bf16" → __nv_bfloat16
//   "trtmc.dit_rms_norm_rope_fp16" → __half

// Extract a DLTensor* from either:
//   - the C-plugin convention (type_index = kTVMFFIDLTensorPtr, v_ptr is DLTensor*)
//   - the Python/tvm-ffi convention (type_index = kTVMFFITensor, v_obj is a
//     Tensor object whose DLTensor sub-object lives at +sizeof(TVMFFIObject)).
// TVMFFITensorGetDLTensorPtr is the inline helper provided by tvm/ffi/c_api.h.
static inline DLTensor* get_dl_tensor(TVMFFIAny const& any) {
    if (any.type_index == kTVMFFITensor) {
        return TVMFFITensorGetDLTensorPtr(any.v_obj);
    }
    return static_cast<DLTensor*>(any.v_ptr);
}

template <typename T>
static int trtmc_dit_rms_norm_rope_impl_typed(
    TVMFFIAny const* args, int32_t num_args, TVMFFIAny* result)
{
    if (num_args < 8) {
        std::fprintf(stderr,
            "[dit_rms_norm_rope] ERROR: expected 8 args, got %d\n", num_args);
        return -1;
    }

    auto* x_in_t   = get_dl_tensor(args[0]);
    auto* weight_t = get_dl_tensor(args[1]);
    auto* cos_t    = get_dl_tensor(args[2]);
    auto* sin_t    = get_dl_tensor(args[3]);
    auto* x_out_t  = get_dl_tensor(args[4]);
    int const num_heads = static_cast<int>(args[5].v_int64);
    int const head_dim  = static_cast<int>(args[6].v_int64);
    float const eps = static_cast<float>(args[7].v_float64);

    int const num_tokens = static_cast<int>(x_in_t->shape[0]);

    // TVM-FFI v0.1.x exposes per-device current stream. kDLCUDA = 2.
    auto stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(kDLCUDA, x_in_t->device.device_id));

    trtmc::diffusion::kernels::launchPerTensorRMSNormRope<T>(
        static_cast<T const*>(x_in_t->data),
        static_cast<T*>(x_out_t->data),
        static_cast<T const*>(weight_t->data),
        static_cast<float const*>(cos_t->data),
        static_cast<float const*>(sin_t->data),
        num_tokens, num_heads, head_dim, eps, stream);

    result->type_index = kTVMFFINone;
    return 0;
}

extern "C" int trtmc_dit_rms_norm_rope_bf16_impl(
    void* /*self*/, TVMFFIAny const* args, int32_t num_args, TVMFFIAny* result)
{
    return trtmc_dit_rms_norm_rope_impl_typed<__nv_bfloat16>(args, num_args, result);
}

extern "C" int trtmc_dit_rms_norm_rope_fp16_impl(
    void* /*self*/, TVMFFIAny const* args, int32_t num_args, TVMFFIAny* result)
{
    return trtmc_dit_rms_norm_rope_impl_typed<__half>(args, num_args, result);
}

static void register_global_func(char const* name_str, TVMFFISafeCallType impl) {
    TVMFFIByteArray name;
    name.data = name_str;
    name.size = static_cast<int64_t>(std::strlen(name_str));
    TVMFFIObjectHandle h = nullptr;
    TVMFFIFunctionCreate(nullptr, impl, nullptr, &h);
    TVMFFIFunctionSetGlobal(&name, h, /*allow_override=*/1);
}

// Run at module load.
namespace {
struct _AutoRegister {
    _AutoRegister() {
        register_global_func("trtmc.dit_rms_norm_rope_bf16",
                             trtmc_dit_rms_norm_rope_bf16_impl);
        register_global_func("trtmc.dit_rms_norm_rope_fp16",
                             trtmc_dit_rms_norm_rope_fp16_impl);
    }
};
static _AutoRegister _auto_register_instance;
} // namespace
