/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host-only TensorRT adapter. All attention math is the unchanged NVIDIA kernel.
#include "native_attention_kernel.h"
#include "runtime/attention_metadata.h"

#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#define HSTU_CUDA_ERROR_CHECK(call)                                                                \
    do {                                                                                           \
        const cudaError_t status = (call);                                                         \
        if (status != cudaSuccess)                                                                 \
            throw std::runtime_error(cudaGetErrorString(status));                                  \
    } while (0)
#ifndef HSTU_DENSE_ATTENTION
#define HSTU_DENSE_ATTENTION 0
#endif

nvinfer1::IPluginCreatorInterface* hstu_linear_creator() noexcept;
nvinfer1::IPluginCreatorInterface* hstu_projection_barrier_creator() noexcept;

namespace {
#if HSTU_DENSE_ATTENTION
constexpr char kName[] = "HstuDenseAttention";
#else
constexpr char kName[] = "HstuPagedAttention";
#endif
constexpr char kVersion[] = "1";
#ifndef HSTU_PLUGIN_NAMESPACE
#error "The model builder must supply a specialization namespace"
#endif
constexpr char kNamespace[] = HSTU_PLUGIN_NAMESPACE;
constexpr int kHeads = 4, kHeadDim = 64, kScaleLength = 1024;
constexpr int kFieldWidth = kHeads * kHeadDim, kTokenStride = 4 * kFieldWidth;

std::size_t metadata_elements(std::size_t batch) {
    return 5 * 8 * (batch + 1);
}
struct MetadataViews {
    std::int32_t *q_offsets, *k_offsets, *targets, *page_indptrs, *page_ids, *last_page_lens;
    MetadataViews(void* data, std::size_t length) {
        q_offsets = static_cast<std::int32_t*>(data);
        k_offsets = q_offsets + 8 * length;
        targets = q_offsets + 16 * length;
        page_indptrs = q_offsets + 24 * length;
        page_ids = q_offsets + 32 * length;
        last_page_lens = targets + length;
    }
};
struct ProfileMetadata {
    int batch, queries;
    std::int32_t* host{nullptr};
    std::size_t elements;
    ProfileMetadata(int b, int q) : batch(b), queries(q), elements(metadata_elements(b)) {
        HSTU_CUDA_ERROR_CHECK(
            cudaMallocHost(reinterpret_cast<void**>(&host), elements * sizeof(std::int32_t)));
        std::fill(host, host + elements, 0);
        MetadataViews views(host, static_cast<std::size_t>(batch + 1));
        for (int user = 0; user < batch; ++user) {
            const int length = queries / batch + (user < queries % batch);
            views.q_offsets[user + 1] = views.q_offsets[user] + length;
            views.k_offsets[user + 1] = views.q_offsets[user + 1];
            views.targets[user] = length;
        }
#if !HSTU_DENSE_ATTENTION
        trtmc::hstu::fill_attention_page_lengths(host, static_cast<std::size_t>(batch), elements);
#endif
        // BUILD only: q=k, history0, zero page planes, every Q row a target.
        // The reserved target-plane tail lengths use the H0 sentinel128.
        // This safe profile underrepresents real historical attention work.
    }
    ~ProfileMetadata() {
        if (host)
            (void)cudaFreeHost(host);
    }
    ProfileMetadata(const ProfileMetadata&) = delete;
    ProfileMetadata& operator=(const ProfileMetadata&) = delete;
};

bool tensor_shape(const nvinfer1::Dims& dims, bool dynamic) {
    return dims.nbDims == 3 && (dims.d[0] > 0 || (dynamic && dims.d[0] == -1)) &&
           dims.d[0] <= std::numeric_limits<std::int32_t>::max() && dims.d[1] == kHeads &&
           dims.d[2] == kHeadDim;
}
bool descriptor(const nvinfer1::PluginTensorDesc& desc, int index, bool dynamic) {
    if (desc.format != nvinfer1::TensorFormat::kLINEAR ||
        desc.type != (index == 2 ? nvinfer1::DataType::kINT32 : nvinfer1::DataType::kBF16))
        return false;
    if (index == 3)
        return tensor_shape(desc.dims, dynamic);
    if (index == 2)
        return desc.dims.nbDims == 3 && desc.dims.d[0] == 5 && desc.dims.d[2] == 8 &&
               (desc.dims.d[1] >= 2 || (dynamic && desc.dims.d[1] == -1)) &&
               desc.dims.d[1] <= std::numeric_limits<std::int32_t>::max() / 8;
    const bool extent =
        desc.dims.nbDims > 0 && (desc.dims.d[0] > 0 || (dynamic && desc.dims.d[0] == -1));
    if (index == 0)
        return extent && desc.dims.nbDims == 2 && desc.dims.d[1] == kTokenStride &&
               desc.dims.d[0] <= std::numeric_limits<std::int32_t>::max();
#if HSTU_DENSE_ATTENTION
    // This one-element constant is a graph adapter placeholder, not page storage.
    return index == 1 && desc.dims.nbDims == 1 && desc.dims.d[0] == 1;
#else
    return index == 1 && extent && desc.dims.nbDims == 4 && desc.dims.d[1] == 2 &&
           desc.dims.d[2] == 128 && desc.dims.d[3] == kFieldWidth &&
           desc.dims.d[0] <= std::numeric_limits<std::int32_t>::max() / 256;
#endif
}

class Plugin final : public nvinfer1::IPluginV3,
                     public nvinfer1::IPluginV3OneCore,
                     public nvinfer1::IPluginV3OneBuild,
                     public nvinfer1::IPluginV3OneRuntime {
  public:
    explicit Plugin(bool build) : build_(build) {}
    nvinfer1::IPluginCapability*
    getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override {
        switch (type) {
        case nvinfer1::PluginCapabilityType::kCORE:
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        case nvinfer1::PluginCapabilityType::kBUILD:
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        case nvinfer1::PluginCapabilityType::kRUNTIME:
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }
    Plugin* clone() noexcept override { return new (std::nothrow) Plugin(build_); }
    const char* getPluginName() const noexcept override { return kName; }
    const char* getPluginVersion() const noexcept override { return kVersion; }
    const char* getPluginNamespace() const noexcept override { return kNamespace; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t getOutputDataTypes(nvinfer1::DataType* output, int32_t outputs,
                               const nvinfer1::DataType* input,
                               int32_t inputs) const noexcept override {
        if (!output || !input || inputs != 3 || outputs != 1)
            return 1;
        for (int i = 0; i < 3; ++i)
            if (input[i] != (i < 2 ? nvinfer1::DataType::kBF16 : nvinfer1::DataType::kINT32))
                return 1;
        output[0] = nvinfer1::DataType::kBF16;
        return 0;
    }
    int32_t getOutputShapes(const nvinfer1::DimsExprs* input, int32_t inputs,
                            const nvinfer1::DimsExprs*, int32_t shape_inputs,
                            nvinfer1::DimsExprs* output, int32_t outputs,
                            nvinfer1::IExprBuilder& builder) noexcept override {
        if (!input || !output || inputs != 3 || outputs != 1 || shape_inputs != 0 ||
            input[0].nbDims != 2)
            return 1;
        output[0].nbDims = 3;
        output[0].d[0] = input[0].d[0];
        output[0].d[1] = builder.constant(kHeads);
        output[0].d[2] = builder.constant(kHeadDim);
        return 0;
    }
    bool supportsFormatCombination(int32_t position, const nvinfer1::DynamicPluginTensorDesc* desc,
                                   int32_t inputs, int32_t outputs) noexcept override {
        return desc && inputs == 3 && outputs == 1 && position >= 0 && position < 4 &&
               descriptor(desc[position].desc, position, true);
    }
    int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* input, int32_t inputs,
                            const nvinfer1::DynamicPluginTensorDesc* output,
                            int32_t outputs) noexcept override {
        if (!input || !output || inputs != 3 || outputs != 1)
            return 1;
        for (int i = 0; i < inputs; ++i)
            if (!descriptor(input[i].desc, i, true))
                return 1;
        return descriptor(output[0].desc, 3, true) ? 0 : 1;
    }
    std::size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* input, int32_t inputs,
                                 const nvinfer1::DynamicPluginTensorDesc*,
                                 int32_t outputs) const noexcept override {
        if (!build_ || !input || inputs != 3 || outputs != 1 || input[2].max.nbDims != 3 ||
            input[2].max.d[0] != 5 || input[2].max.d[2] != 8 || input[2].max.d[1] < 2 ||
            input[2].max.d[1] > std::numeric_limits<std::int32_t>::max() / 8)
            return 0;
        return metadata_elements(static_cast<std::size_t>(input[2].max.d[1] - 1)) *
               sizeof(std::int32_t);
    }
    const char* getTimingCacheID() noexcept override {
#if HSTU_DENSE_ATTENTION
        return "original-generic-bf16-dense-m64n128w4-packed5x8-uqkv1024-h4-d64-scale1024-v2";
#else
        return "original-generic-bf16-nativepages-m64n128w4-packed5x8-uqkv1024-h4-d64-scale1024-v2";
#endif
    }
    const char* getMetadataString() noexcept override {
#if HSTU_DENSE_ATTENTION
        return "UQKV[T,1024],BF16;Q256,K512,V768,stride1024;ignored_dummy_BF16[1];outputTHD;"
               "INT32_metadata[5,B+1,8]:Qoffsets,Koffsets,targets,unused,unused;"
               "H4,D64,M64,N128,W4,alpha.125,scale1024;dense_QK_equal;finite_padding_independent_"
               "targets;"
               "RUNTIME_views_only_no_copy_no_GPU_read;dummy_not_passed_to_CUDA_ABI";
#else
        return "UQKV[T,1024],BF16;Q256,K512,V768,stride1024;pages[P,2,128,256];outputTHD;"
               "INT32_metadata[5,B+1,8]:Qoffsets,Koffsets,targets,pageIndptrs,pageIDs;"
               "lastPageLens=targets+(B+1);zero_referenced_tail_required;"
               "H4,D64,M64,N128,W4,alpha.125,scale1024;BUILD_history0_underrepresents_history_cost;"
               "RUNTIME_views_only_no_copy_no_GPU_read;pageIDs_capacity8xL_CPU_validated_used_end";
#endif
    }
    int32_t onShapeChange(const nvinfer1::PluginTensorDesc* input, int32_t inputs,
                          const nvinfer1::PluginTensorDesc* output,
                          int32_t outputs) noexcept override {
        try {
            ready_ = false;
            if (!input || !output || inputs != 3 || outputs != 1)
                return 1;
            for (int i = 0; i < inputs; ++i)
                if (!descriptor(input[i], i, false))
                    return 1;
            if (!descriptor(output[0], 3, false) || output[0].dims.d[0] != input[0].dims.d[0])
                return 1;
            const int batch = static_cast<int>(input[2].dims.d[1] - 1);
            const int queries = static_cast<int>(input[0].dims.d[0]);
            if (queries < batch ||
                (queries + static_cast<std::int64_t>(batch) - 1) / batch > kScaleLength)
                return 1;
            if (build_) {
                profile_ = nullptr;
                for (const auto& saved : profiles_)
                    if (saved->batch == batch && saved->queries == queries)
                        profile_ = saved.get();
                if (!profile_) {
                    profiles_.push_back(std::make_unique<ProfileMetadata>(batch, queries));
                    profile_ = profiles_.back().get();
                }
            }
            ready_ = true;
            return 0;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "%s shape: %s\n", kName, error.what());
            return 1;
        }
    }
    int32_t enqueue(const nvinfer1::PluginTensorDesc* desc, const nvinfer1::PluginTensorDesc*,
                    const void* const* input, void* const* output, void* workspace,
                    cudaStream_t stream) noexcept override {
        try {
            if (!ready_ || !desc || !input || !output)
                return 1;
            if (!input[0] || !input[2] || !output[0])
                return 1;
#if !HSTU_DENSE_ATTENTION
            if (!input[1])
                return 1;
#endif
            for (int index = 0; index < 3; ++index) {
                // The dense dummy is never read or described by the CUDA ABI;
                // its two-byte constant does not require a tensor-map alignment.
                if (HSTU_DENSE_ATTENTION && index == 1)
                    continue;
                if (reinterpret_cast<std::uintptr_t>(input[index]) % 16 != 0)
                    return 1;
            }
            if (reinterpret_cast<std::uintptr_t>(output[0]) % 16 != 0)
                return 1;
            const auto length = static_cast<std::size_t>(desc[2].dims.d[1]);
            // Runtime only forms host pointer/shape views. Values and the used
            // page-ID count must have been validated by the CPU owner before upload.
            MetadataViews metadata(const_cast<void*>(input[2]), length);
            if (build_) {
                if (!profile_ || !workspace)
                    return 1;
                HSTU_CUDA_ERROR_CHECK(cudaMemcpyAsync(workspace, profile_->host,
                                                      profile_->elements * sizeof(std::int32_t),
                                                      cudaMemcpyHostToDevice, stream));
                metadata = MetadataViews(workspace, length);
            }
            const auto* projection = static_cast<const std::uint16_t*>(input[0]);
            HstuAttentionArgs args{};
            args.q = const_cast<std::uint16_t*>(projection + kFieldWidth);
            args.k = const_cast<std::uint16_t*>(projection + 2 * kFieldWidth);
            args.v = const_cast<std::uint16_t*>(projection + 3 * kFieldWidth);
            args.output = output[0];
            args.q_offsets = metadata.q_offsets;
            args.k_offsets = metadata.k_offsets;
            args.targets = metadata.targets;
            args.batch = static_cast<std::int32_t>(length - 1);
            args.max_q = args.max_k = kScaleLength;
            args.q_row_stride = args.k_row_stride = args.v_row_stride = kTokenStride;
#if !HSTU_DENSE_ATTENTION
            args.pages = const_cast<void*>(input[1]);
            args.page_count = static_cast<std::int32_t>(desc[1].dims.d[0]);
            args.page_indptrs = metadata.page_indptrs;
            args.page_ids = metadata.page_ids;
            args.last_page_lens = metadata.last_page_lens;
#endif
            // Dense leaves every page field zero; its one-element graph dummy
            // never reaches the kernel. Both modes use the same fixed M64 body.
            return hstu_attention_forward(&args, stream) == 0 ? 0 : 1;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "%s enqueue: %s\n", kName, error.what());
            return 1;
        }
    }
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        return clone();
    }
    const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept override {
        return &fields_;
    }

  private:
    bool build_, ready_{false};
    nvinfer1::PluginFieldCollection fields_{0, nullptr};
    std::vector<std::unique_ptr<ProfileMetadata>> profiles_;
    ProfileMetadata* profile_{nullptr};
};
class Creator final : public nvinfer1::IPluginCreatorV3One {
  public:
    const char* getPluginName() const noexcept override { return kName; }
    const char* getPluginVersion() const noexcept override { return kVersion; }
    const char* getPluginNamespace() const noexcept override { return kNamespace; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV3* createPlugin(const char*, const nvinfer1::PluginFieldCollection* fields,
                                      nvinfer1::TensorRTPhase phase) noexcept override {
        if (fields && fields->nbFields != 0)
            return nullptr;
        return new (std::nothrow) Plugin(phase == nvinfer1::TensorRTPhase::kBUILD);
    }

  private:
    nvinfer1::PluginFieldCollection fields_{0, nullptr};
};
Creator creator;
// TensorRT loads this creator into a builder/runtime-local registry. There is
// no process-global registration and no LD_PRELOAD requirement.
} // namespace

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder*) noexcept {}
extern "C" nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& count) noexcept {
#if HSTU_DENSE_ATTENTION
    static nvinfer1::IPluginCreatorInterface* creators[] = {&creator, hstu_linear_creator()};
#else
    static nvinfer1::IPluginCreatorInterface* creators[] = {&creator, hstu_linear_creator(),
                                                            hstu_projection_barrier_creator()};
#endif
    count = static_cast<int32_t>(sizeof(creators) / sizeof(creators[0]));
    return creators;
}
