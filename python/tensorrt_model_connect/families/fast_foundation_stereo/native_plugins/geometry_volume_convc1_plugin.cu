/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "geometry_volume_convc1_plugin.h"

#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <mma.h>

namespace trtmc {
namespace {

namespace wmma = nvcuda::wmma;

constexpr int32_t kBatch = 1;
constexpr int32_t kGeometryChannels = 28;
constexpr int32_t kVolumeChannelPitch = 32;
constexpr int32_t kLevels = 2;
constexpr int32_t kRadius = 4;
constexpr int32_t kSamples = 2 * kRadius + 1;
constexpr int32_t kInterpolationEndpoints = kSamples + 1;
constexpr int32_t kHeight = 176;
constexpr int32_t kWidth = 176;
constexpr int32_t kPixels = kBatch * kHeight * kWidth;
constexpr int32_t kGeometryWidth0 = 48;
constexpr int32_t kGeometryWidth1 = 24;
constexpr int32_t kCorrelationWidth0 = 176;
constexpr int32_t kCorrelationWidth1 = 88;
constexpr int32_t kSourcesPerLevel = kGeometryChannels + 1;
constexpr int32_t kSourceGroups = kLevels * kSourcesPerLevel;
constexpr int32_t kSampledChannelsPerLevel = kSourcesPerLevel * kSamples;
constexpr int32_t kSampledChannels = kLevels * kSampledChannelsPerLevel;
constexpr int32_t kSampledChannelPitch = 528;
constexpr int32_t kSourceGroupsPerPhase = 16;
constexpr int32_t kPhaseSampledChannelPitch = kSourceGroupsPerPhase * kSamples;
constexpr int32_t kPhases = 4;
constexpr int32_t kTailSourceGroups = kSourceGroups - 3 * kSourceGroupsPerPhase;
constexpr int32_t kTailSampledChannels = kTailSourceGroups * kSamples;
constexpr int32_t kTailMmaChannels = 96;
constexpr int32_t kOutputChannels = 56;
constexpr int32_t kOutputChannelPitch = 56;
constexpr int32_t kWeightOutputPitch = 64;
constexpr int32_t kPixelsPerBlock = 32;
constexpr int32_t kThreads = 256;
constexpr int32_t kWarpSize = 32;
constexpr int32_t kWarpsPerBlock = kThreads / kWarpSize;
constexpr int32_t kHalfWarp = 16;
constexpr int32_t kHalfWarpsPerBlock = kThreads / kHalfWarp;
constexpr int32_t kMmaM = 16;
constexpr int32_t kMmaN = 16;
constexpr int32_t kMmaK = 16;
constexpr int32_t kCoordinateKinds = kLevels * 2;

static_assert(kGeometryChannels < kVolumeChannelPitch);
static_assert(kVolumeChannelPitch % 8 == 0);
static_assert(kSampledChannels == 522);
static_assert(kSampledChannelPitch == 528);
static_assert(kSampledChannelPitch % kMmaK == 0);
static_assert(kSourceGroups == 58);
static_assert(kPhaseSampledChannelPitch == 144);
static_assert(kPhaseSampledChannelPitch % kMmaK == 0);
static_assert(kTailSourceGroups == 10);
static_assert(kTailSampledChannels == 90);
static_assert(kTailMmaChannels == 96);
static_assert(kTailMmaChannels % kMmaK == 0);
static_assert(3 * kPhaseSampledChannelPitch + kTailMmaChannels == kSampledChannelPitch);
static_assert(kOutputChannels == 56);
static_assert(kOutputChannelPitch == kOutputChannels);
static_assert(kOutputChannels % 8 == 0);
static_assert(kWeightOutputPitch == 64);
static_assert(kWeightOutputPitch % kMmaN == 0);
static_assert(kPixels % kPixelsPerBlock == 0);
static_assert(kWarpsPerBlock == 8);

struct SamplingSharedStorage {
    float coordinate_fractions[kPixelsPerBlock][kCoordinateKinds];
    int32_t coordinate_starts[kPixelsPerBlock][kCoordinateKinds];
    int32_t coordinate_is_finite[kPixelsPerBlock][kCoordinateKinds];
    __half sampled[kPixelsPerBlock][kPhaseSampledChannelPitch];
};

union __align__(32) SharedStorage {
    SamplingSharedStorage sampling;
    float accumulators[kWarpsPerBlock][kMmaM][kMmaN];
};

static_assert(sizeof(SamplingSharedStorage) == 10752);
static_assert(sizeof(SharedStorage) == 10752);

__device__ __forceinline__ float load_geometry(const __half* volume, int32_t level, int32_t pixel,
                                               int32_t source_channel, int32_t source_index) {
    // TensorRT kDHWC8 for logical [N,C,D,H,W] is physically NDHWC8. The
    // distilled 28-channel volume therefore has a 32-half channel pitch.
    const int32_t disparity0 = level == 0 ? source_index : 2 * source_index;
    const int32_t index0 = (disparity0 * kPixels + pixel) * kVolumeChannelPitch + source_channel;
    const float value0 = __half2float(volume[index0]);
    if (level == 0)
        return value0;

    // Match avg-pool(window=2, stride=2) after the original HALF->FLOAT cast:
    // first round the two converted values' sum in FP32, then multiply by the
    // exactly representable reciprocal. There is no padding at this level.
    const int32_t index1 = index0 + kPixels * kVolumeChannelPitch;
    const float value1 = __half2float(volume[index1]);
    return __fmul_rn(__fadd_rn(value0, value1), 0.5F);
}

__device__ __forceinline__ float load_source(const __half* volume, const float* correlation0,
                                             const float* correlation1, int32_t level,
                                             bool is_correlation, int32_t pixel,
                                             int32_t source_channel, int32_t source_index) {
    if (is_correlation) {
        const float* correlation = level == 0 ? correlation0 : correlation1;
        const int32_t width = level == 0 ? kCorrelationWidth0 : kCorrelationWidth1;
        return correlation[pixel * width + source_index];
    }
    return load_geometry(volume, level, pixel, source_channel, source_index);
}

__global__ __launch_bounds__(kThreads, 6) void geometry_volume_convc1_hwc8_wmma_kernel(
    const float* disparity, const __half* volume, const float* correlation0,
    const float* correlation1, const __half* packed_weight, const __half* packed_bias,
    __half* output) {
    __shared__ SharedStorage shared;

    const int32_t block_pixel = static_cast<int32_t>(blockIdx.x) * kPixelsPerBlock;
    if (threadIdx.x < kPixelsPerBlock * kCoordinateKinds) {
        const int32_t local_pixel = threadIdx.x / kCoordinateKinds;
        const int32_t coordinate_kind = threadIdx.x % kCoordinateKinds;
        const int32_t level = coordinate_kind / 2;
        const bool is_correlation = coordinate_kind % 2 != 0;
        const int32_t pixel = block_pixel + local_pixel;
        const int32_t width_index = pixel % kWidth;
        const float inverse_scale = level == 0 ? 1.0F : 0.5F;
        const float disparity_level = disparity[pixel] * inverse_scale;
        const float coordinate =
            is_correlation ? static_cast<float>(width_index) * inverse_scale - disparity_level
                           : disparity_level;
        const int32_t source_width = is_correlation
                                         ? (level == 0 ? kCorrelationWidth0 : kCorrelationWidth1)
                                         : (level == 0 ? kGeometryWidth0 : kGeometryWidth1);
        // Across the nine radius-four samples, no source index can contribute
        // outside this bounded interval. Reject it before float-to-int conversion
        // so even a finite FLT_MAX disparity cannot overflow an int32_t.
        const bool finite_coordinate = isfinite(coordinate) && coordinate >= -5.0F &&
                                       coordinate < static_cast<float>(source_width + 4);
        const float coordinate_floor = finite_coordinate ? floorf(coordinate) : 0.0F;
        shared.sampling.coordinate_starts[local_pixel][coordinate_kind] =
            static_cast<int32_t>(coordinate_floor) - kRadius;
        shared.sampling.coordinate_fractions[local_pixel][coordinate_kind] =
            finite_coordinate ? coordinate - coordinate_floor : 0.0F;
        shared.sampling.coordinate_is_finite[local_pixel][coordinate_kind] =
            finite_coordinate ? 1 : 0;
    }
    __syncthreads();

    const int32_t warp_index = threadIdx.x / kWarpSize;
    const int32_t warp_lane = threadIdx.x % kWarpSize;
    const int32_t half_warp_index = threadIdx.x / kHalfWarp;
    const int32_t half_warp_lane = threadIdx.x % kHalfWarp;
    const int32_t pixel_tile = warp_index / (kWeightOutputPitch / kMmaN);
    const int32_t output_tile = warp_index % (kWeightOutputPitch / kMmaN);
    const int32_t local_pixel_base = pixel_tile * kMmaM;
    const int32_t output_channel_base = output_tile * kMmaN;

    wmma::fragment<wmma::accumulator, kMmaM, kMmaN, kMmaK, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0F);

    // Keep the exact [level][geometry-or-correlation source][nine samples]
    // channel order used by the standalone sampler and its 1x1 convolution.
    for (int32_t phase = 0; phase < kPhases; ++phase) {
        const int32_t source_group_base = phase * kSourceGroupsPerPhase;
        const int32_t phase_source_groups =
            phase == kPhases - 1 ? kTailSourceGroups : kSourceGroupsPerPhase;
        const int32_t phase_mma_channels =
            phase == kPhases - 1 ? kTailMmaChannels : kPhaseSampledChannelPitch;
        // Correlation is laid out [pixel, source-width]. A single half-warp
        // lane loading all ten interpolation endpoints serially would execute
        // ten warp-wide global-load instructions. Instead, ten adjacent lanes
        // issue one coalesced load and exchange the right endpoint with a
        // width-16 shuffle. This preserves the exact FP32 interpolation and
        // HALF rounding boundary.
        const bool phase_has_correlation = phase == 1 || phase == kPhases - 1;
        if (phase_has_correlation) {
            const int32_t level = phase == 1 ? 0 : 1;
            const int32_t coordinate_kind = level * 2 + 1;
            const int32_t correlation_source_group = level * kSourcesPerLevel + kGeometryChannels;
            const int32_t phase_correlation_source_group =
                correlation_source_group - source_group_base;
            const float* correlation = level == 0 ? correlation0 : correlation1;
            const int32_t source_width = level == 0 ? kCorrelationWidth0 : kCorrelationWidth1;
            for (int32_t local_pixel = half_warp_index; local_pixel < kPixelsPerBlock;
                 local_pixel += kHalfWarpsPerBlock) {
                const bool finite_coordinate =
                    shared.sampling.coordinate_is_finite[local_pixel][coordinate_kind] != 0;
                const float fraction =
                    shared.sampling.coordinate_fractions[local_pixel][coordinate_kind];
                const int32_t coordinate_start =
                    shared.sampling.coordinate_starts[local_pixel][coordinate_kind];
                const int32_t source_index = coordinate_start + half_warp_lane;
                float value = 0.0F;
                if (half_warp_lane < kInterpolationEndpoints && finite_coordinate &&
                    source_index >= 0 && source_index < source_width) {
                    const int32_t pixel = block_pixel + local_pixel;
                    value = correlation[pixel * source_width + source_index];
                }
                const float next_value = __shfl_down_sync(0xFFFFFFFFU, value, 1, kHalfWarp);
                if (half_warp_lane < kSamples) {
                    const float sampled = value * (1.0F - fraction) + next_value * fraction;
                    const int32_t phase_sampled_channel =
                        phase_correlation_source_group * kSamples + half_warp_lane;
                    shared.sampling.sampled[local_pixel][phase_sampled_channel] =
                        __float2half_rn(sampled);
                }
            }
        }
        // A half-warp owns one pixel and maps its lanes to sixteen adjacent
        // source groups. That mapping turns the NDHWC8 geometry loads into
        // channel-contiguous 32-byte transactions instead of striding adjacent
        // lanes across complete disparity planes. Each half-warp handles two
        // pixels, and each lane reuses the right interpolation endpoint as the
        // next sample's left endpoint.
        for (int32_t local_pixel = half_warp_index; local_pixel < kPixelsPerBlock;
             local_pixel += kHalfWarpsPerBlock) {
            const int32_t phase_source_group = half_warp_lane;
            if (phase_source_group < phase_source_groups) {
                const int32_t source_group = source_group_base + phase_source_group;
                const int32_t level = source_group / kSourcesPerLevel;
                const int32_t source_channel = source_group % kSourcesPerLevel;
                const bool is_correlation = source_channel == kGeometryChannels;
                if (is_correlation)
                    continue;
                const int32_t coordinate_kind = level * 2 + static_cast<int32_t>(is_correlation);
                const bool finite_coordinate =
                    shared.sampling.coordinate_is_finite[local_pixel][coordinate_kind] != 0;
                const float fraction =
                    shared.sampling.coordinate_fractions[local_pixel][coordinate_kind];
                const int32_t coordinate_start =
                    shared.sampling.coordinate_starts[local_pixel][coordinate_kind];
                const int32_t pixel = block_pixel + local_pixel;

                const int32_t source_width =
                    is_correlation ? (level == 0 ? kCorrelationWidth0 : kCorrelationWidth1)
                                   : (level == 0 ? kGeometryWidth0 : kGeometryWidth1);
                float value = 0.0F;
                if (finite_coordinate && coordinate_start >= 0 && coordinate_start < source_width) {
                    value = load_source(volume, correlation0, correlation1, level, is_correlation,
                                        pixel, source_channel, coordinate_start);
                }

#pragma unroll
                for (int32_t sample = 0; sample < kSamples; ++sample) {
                    const int32_t next_source_index = coordinate_start + sample + 1;
                    float next_value = 0.0F;
                    if (finite_coordinate && next_source_index >= 0 &&
                        next_source_index < source_width) {
                        next_value =
                            load_source(volume, correlation0, correlation1, level, is_correlation,
                                        pixel, source_channel, next_source_index);
                    }

                    const float sampled = value * (1.0F - fraction) + next_value * fraction;
                    const int32_t phase_sampled_channel = phase_source_group * kSamples + sample;
                    // Preserve the sampler -> TensorRT convolution boundary.
                    shared.sampling.sampled[local_pixel][phase_sampled_channel] =
                        __float2half_rn(sampled);
                    value = next_value;
                }
            }
        }

        if (phase == kPhases - 1) {
            constexpr int32_t kTailPaddingChannels = kTailMmaChannels - kTailSampledChannels;
            constexpr int32_t kTailPaddingValues = kPixelsPerBlock * kTailPaddingChannels;
            if (threadIdx.x < kTailPaddingValues) {
                const int32_t local_pixel = threadIdx.x / kTailPaddingChannels;
                const int32_t padding_channel = threadIdx.x % kTailPaddingChannels;
                shared.sampling.sampled[local_pixel][kTailSampledChannels + padding_channel] =
                    __float2half_rn(0.0F);
            }
        }
        __syncthreads();

        const int32_t sampled_channel_base = source_group_base * kSamples;
        for (int32_t phase_sampled_channel = 0; phase_sampled_channel < phase_mma_channels;
             phase_sampled_channel += kMmaK) {
            wmma::fragment<wmma::matrix_a, kMmaM, kMmaN, kMmaK, __half, wmma::row_major>
                sampled_fragment;
            wmma::fragment<wmma::matrix_b, kMmaM, kMmaN, kMmaK, __half, wmma::col_major>
                weight_fragment;
            wmma::load_matrix_sync(
                sampled_fragment, &shared.sampling.sampled[local_pixel_base][phase_sampled_channel],
                kPhaseSampledChannelPitch);
            wmma::load_matrix_sync(weight_fragment,
                                   packed_weight + output_channel_base * kSampledChannelPitch +
                                       sampled_channel_base + phase_sampled_channel,
                                   kSampledChannelPitch);
            wmma::mma_sync(accumulator, sampled_fragment, weight_fragment, accumulator);
        }

        if (phase != kPhases - 1)
            __syncthreads();
    }
    // All warps must finish their final shared-A load before the accumulator
    // store reuses the sampling storage. The union lowers shared memory enough
    // for six resident blocks (48 warps) on SM89.
    __syncthreads();
    wmma::store_matrix_sync(&shared.accumulators[warp_index][0][0], accumulator, kMmaN,
                            wmma::mem_row_major);
    __syncwarp();

    for (int32_t output_index = warp_lane; output_index < kMmaM * kMmaN;
         output_index += kWarpSize) {
        const int32_t pixel_in_tile = output_index / kMmaN;
        const int32_t channel_in_tile = output_index % kMmaN;
        const int32_t output_channel = output_channel_base + channel_in_tile;
        if (output_channel < kOutputChannels) {
            float value = shared.accumulators[warp_index][pixel_in_tile][channel_in_tile] +
                          __half2float(packed_bias[output_channel]);
            value = value < 0.0F ? 0.0F : value;
            const int32_t pixel = block_pixel + local_pixel_base + pixel_in_tile;
            output[pixel * kOutputChannelPitch + output_channel] = __float2half_rn(value);
        }
    }
}

bool is_exact_disparity(nvinfer1::PluginTensorDesc const& desc) noexcept {
    return desc.type == nvinfer1::DataType::kFLOAT &&
           desc.format == nvinfer1::TensorFormat::kLINEAR && desc.dims.nbDims == 4 &&
           desc.dims.d[0] == kBatch && desc.dims.d[1] == 1 && desc.dims.d[2] == kHeight &&
           desc.dims.d[3] == kWidth;
}

bool is_exact_volume(nvinfer1::PluginTensorDesc const& desc) noexcept {
    return desc.type == nvinfer1::DataType::kHALF &&
           desc.format == nvinfer1::TensorFormat::kDHWC8 && desc.dims.nbDims == 5 &&
           desc.dims.d[0] == kBatch && desc.dims.d[1] == kGeometryChannels &&
           desc.dims.d[2] == kGeometryWidth0 && desc.dims.d[3] == kHeight &&
           desc.dims.d[4] == kWidth;
}

bool is_exact_correlation(nvinfer1::PluginTensorDesc const& desc, int32_t width) noexcept {
    return desc.type == nvinfer1::DataType::kFLOAT &&
           desc.format == nvinfer1::TensorFormat::kLINEAR && desc.dims.nbDims == 4 &&
           desc.dims.d[0] == kPixels && desc.dims.d[1] == 1 && desc.dims.d[2] == 1 &&
           desc.dims.d[3] == width;
}

bool is_exact_weight(nvinfer1::PluginTensorDesc const& desc) noexcept {
    return desc.type == nvinfer1::DataType::kHALF &&
           desc.format == nvinfer1::TensorFormat::kLINEAR && desc.dims.nbDims == 2 &&
           desc.dims.d[0] == kWeightOutputPitch && desc.dims.d[1] == kSampledChannelPitch;
}

bool is_exact_bias(nvinfer1::PluginTensorDesc const& desc) noexcept {
    return desc.type == nvinfer1::DataType::kHALF &&
           desc.format == nvinfer1::TensorFormat::kLINEAR && desc.dims.nbDims == 1 &&
           desc.dims.d[0] == kWeightOutputPitch;
}

bool is_exact_output(nvinfer1::PluginTensorDesc const& desc) noexcept {
    return desc.type == nvinfer1::DataType::kHALF && desc.format == nvinfer1::TensorFormat::kHWC8 &&
           desc.dims.nbDims == 4 && desc.dims.d[0] == kBatch && desc.dims.d[1] == kOutputChannels &&
           desc.dims.d[2] == kHeight && desc.dims.d[3] == kWidth;
}

} // namespace

FastFoundationStereoGeometryVolumeConvc1Plugin::FastFoundationStereoGeometryVolumeConvc1Plugin(
    const void*, std::size_t length)
    : valid_(length == 0) {}

char const* FastFoundationStereoGeometryVolumeConvc1Plugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* FastFoundationStereoGeometryVolumeConvc1Plugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t FastFoundationStereoGeometryVolumeConvc1Plugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t FastFoundationStereoGeometryVolumeConvc1Plugin::initialize() noexcept {
    return valid_ ? 0 : 1;
}

void FastFoundationStereoGeometryVolumeConvc1Plugin::terminate() noexcept {}

void FastFoundationStereoGeometryVolumeConvc1Plugin::destroy() noexcept {
    delete this;
}

std::size_t FastFoundationStereoGeometryVolumeConvc1Plugin::getSerializationSize() const noexcept {
    return 0;
}

void FastFoundationStereoGeometryVolumeConvc1Plugin::serialize(void*) const noexcept {}

void FastFoundationStereoGeometryVolumeConvc1Plugin::setPluginNamespace(
    char const* plugin_namespace) noexcept {
    namespace_ = plugin_namespace != nullptr ? plugin_namespace : "";
}

char const* FastFoundationStereoGeometryVolumeConvc1Plugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType FastFoundationStereoGeometryVolumeConvc1Plugin::getOutputDataType(
    int32_t, nvinfer1::DataType const*, int32_t) const noexcept {
    return nvinfer1::DataType::kHALF;
}

FastFoundationStereoGeometryVolumeConvc1Plugin*
FastFoundationStereoGeometryVolumeConvc1Plugin::clone() const noexcept {
    auto* plugin = new FastFoundationStereoGeometryVolumeConvc1Plugin();
    plugin->namespace_ = namespace_;
    plugin->valid_ = valid_;
    return plugin;
}

nvinfer1::DimsExprs FastFoundationStereoGeometryVolumeConvc1Plugin::getOutputDimensions(
    int32_t output_index, nvinfer1::DimsExprs const* inputs, int32_t input_count,
    nvinfer1::IExprBuilder& expr_builder) noexcept {
    nvinfer1::DimsExprs output{};
    if (output_index != 0 || inputs == nullptr || input_count != 6)
        return output;
    output.nbDims = 4;
    output.d[0] = inputs[0].d[0];
    output.d[1] = expr_builder.constant(kOutputChannels);
    output.d[2] = inputs[0].d[2];
    output.d[3] = inputs[0].d[3];
    return output;
}

bool FastFoundationStereoGeometryVolumeConvc1Plugin::supportsFormatCombination(
    int32_t position, nvinfer1::PluginTensorDesc const* input_output, int32_t input_count,
    int32_t output_count) noexcept {
    if (input_output == nullptr || input_count != 6 || output_count != 1 || position < 0 ||
        position >= 7)
        return false;
    switch (position) {
    case 0:
        return is_exact_disparity(input_output[position]);
    case 1:
        return is_exact_volume(input_output[position]);
    case 2:
        return is_exact_correlation(input_output[position], kCorrelationWidth0);
    case 3:
        return is_exact_correlation(input_output[position], kCorrelationWidth1);
    case 4:
        return is_exact_weight(input_output[position]);
    case 5:
        return is_exact_bias(input_output[position]);
    case 6:
        return is_exact_output(input_output[position]);
    default:
        return false;
    }
}

void FastFoundationStereoGeometryVolumeConvc1Plugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const*, int32_t, nvinfer1::DynamicPluginTensorDesc const*,
    int32_t) noexcept {}

std::size_t FastFoundationStereoGeometryVolumeConvc1Plugin::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const*, int32_t, nvinfer1::PluginTensorDesc const*,
    int32_t) const noexcept {
    return 0;
}

int32_t FastFoundationStereoGeometryVolumeConvc1Plugin::enqueue(
    nvinfer1::PluginTensorDesc const* input_desc, nvinfer1::PluginTensorDesc const* output_desc,
    void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept {
    if (input_desc == nullptr || output_desc == nullptr || inputs == nullptr ||
        outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr ||
        inputs[2] == nullptr || inputs[3] == nullptr || inputs[4] == nullptr ||
        inputs[5] == nullptr || outputs[0] == nullptr || !is_exact_disparity(input_desc[0]) ||
        !is_exact_volume(input_desc[1]) ||
        !is_exact_correlation(input_desc[2], kCorrelationWidth0) ||
        !is_exact_correlation(input_desc[3], kCorrelationWidth1) ||
        !is_exact_weight(input_desc[4]) || !is_exact_bias(input_desc[5]) ||
        !is_exact_output(output_desc[0])) {
        return -1;
    }

    constexpr int32_t kBlocks = kPixels / kPixelsPerBlock;
    geometry_volume_convc1_hwc8_wmma_kernel<<<kBlocks, kThreads, 0, stream>>>(
        static_cast<const float*>(inputs[0]), static_cast<const __half*>(inputs[1]),
        static_cast<const float*>(inputs[2]), static_cast<const float*>(inputs[3]),
        static_cast<const __half*>(inputs[4]), static_cast<const __half*>(inputs[5]),
        static_cast<__half*>(outputs[0]));
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

} // namespace trtmc
