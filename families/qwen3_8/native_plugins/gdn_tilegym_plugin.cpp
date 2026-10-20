/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Single-token Gated DeltaNet decode for Qwen3.8 as four cuTile kernels
// (see ../cutile/gdn_tilegym.py): conv1d+SiLU, gate/beta, delta rule, gated RMSNorm.
//
// The kernels are compiled ahead of time to cubins. The cubins travel inside the plugin's
// serialized fields, so the engine is self-contained and the runtime needs no Python.
//
// Inputs : qkv[1,conv_dim] z[1,d_inner] a[1,H] b[1,H]            (fp16 or bf16, same type)
//          conv_state[conv_dim,4]                                 (same type as qkv)
//          conv_w[conv_dim,4] conv_b[conv_dim] A[H] dt_bias[H]
//          gamma[d_inner] ssm_state[H,D,D]                        (fp32)
// With the field fuse_ab=1 the plugin also computes the a and b projections: input 2 is the
// normalized hidden row x[1,hidden_size] (activation type) and input 3 is the weight
// w_ab[2H,hidden_size] (fp32, a rows then b rows) instead of a and b.
// Outputs: out[1,d_inner], conv_state'[conv_dim,4]  (same type as qkv)
//          ssm_state'[H,D,D]                                      (fp32)
// Workspace: y[conv_dim] (act), g[H] (fp32), beta[H] (act), o[d_inner] (act).

#include <NvInferRuntime.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <functional>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "cubin_launch.h"

namespace trtmc {
namespace {

constexpr int32_t kNumInputs = 11;
constexpr int32_t kNumOutputs = 3;
constexpr int32_t kConvKernel = 4;
constexpr int32_t kNumStages = 4;
constexpr char const* kStageField[kNumStages] = {"cubin_conv", "cubin_gate", "cubin_gdn",
                                                 "cubin_norm"};
constexpr char const* kStageSymbol[kNumStages] = {"gdn_conv_silu", "gdn_gate", "gdn_delta",
                                                  "gdn_norm"};

} // namespace

class Qwen38GdnDecodeTileGymPlugin final : public nvinfer1::IPluginV3,
                                           public nvinfer1::IPluginV3OneCore,
                                           public nvinfer1::IPluginV3OneBuild,
                                           public nvinfer1::IPluginV3OneRuntime {
  public:
    static constexpr nvinfer1::AsciiChar const* kPLUGIN_NAME = "Qwen38GdnDecodeTileGym";
    static constexpr nvinfer1::AsciiChar const* kPLUGIN_VERSION = "1";

    explicit Qwen38GdnDecodeTileGymPlugin(nvinfer1::PluginFieldCollection const& fc) noexcept {
        for (int32_t i = 0; i < fc.nbFields; ++i) {
            auto const& f = fc.fields[i];
            std::string const name = f.name;
            if (f.data == nullptr)
                continue;
            auto const as_int = [&]() { return *static_cast<int32_t const*>(f.data); };
            if (f.type == nvinfer1::PluginFieldType::kINT32) {
                if (name == "threads" && f.length == kNumStages)
                    for (int32_t s = 0; s < kNumStages; ++s)
                        threads_[s] = static_cast<int32_t const*>(f.data)[s];
                else if (name == "num_heads") num_heads_ = as_int();
                else if (name == "num_kv_heads") num_kv_heads_ = as_int();
                else if (name == "head_dim") head_dim_ = as_int();
                else if (name == "conv_dim") conv_dim_ = as_int();
                else if (name == "act_bytes") act_bytes_ = as_int();
                else if (name == "conv_block") conv_block_ = as_int();
                else if (name == "block_v") block_v_ = as_int();
                else if (name == "fuse_ab") fuse_ab_ = as_int();
                else if (name == "hidden_size") hidden_size_ = as_int();
            } else if (f.type == nvinfer1::PluginFieldType::kINT8) {
                for (int32_t s = 0; s < kNumStages; ++s)
                    if (name == kStageField[s])
                        cubins_[s].assign(static_cast<char const*>(f.data),
                                          static_cast<char const*>(f.data) + f.length);
            }
        }
        valid_ = num_heads_ > 0 && num_kv_heads_ > 0 && head_dim_ > 0 && conv_dim_ > 0 &&
                 (act_bytes_ == 2) && conv_block_ > 0 && block_v_ > 0 &&
                 head_dim_ % block_v_ == 0;
        for (int32_t s = 0; s < kNumStages; ++s)
            valid_ = valid_ && !cubins_[s].empty() && threads_[s] > 0;
    }
    Qwen38GdnDecodeTileGymPlugin(Qwen38GdnDecodeTileGymPlugin const&) = default;

    bool isValid() const noexcept { return valid_; }

    // ---- IPluginV3
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
    nvinfer1::IPluginV3* clone() noexcept override {
        return new (std::nothrow) Qwen38GdnDecodeTileGymPlugin(*this);
    }

    // ---- Core
    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return kPLUGIN_NAME; }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return kPLUGIN_VERSION;
    }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

    // ---- Build
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                            nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {
        return 0;
    }
    int32_t getOutputDataTypes(nvinfer1::DataType* types, int32_t nb_outputs,
                               nvinfer1::DataType const* input_types,
                               int32_t nb_inputs) const noexcept override {
        if (nb_outputs != kNumOutputs || nb_inputs != kNumInputs)
            return -1;
        types[0] = input_types[0];
        types[1] = input_types[0];
        types[2] = nvinfer1::DataType::kFLOAT;
        return 0;
    }
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nb_inputs,
                            nvinfer1::DimsExprs const*, int32_t, nvinfer1::DimsExprs* outputs,
                            int32_t nb_outputs, nvinfer1::IExprBuilder&) noexcept override {
        if (nb_outputs != kNumOutputs || nb_inputs != kNumInputs)
            return -1;
        outputs[0] = inputs[1];  // z      -> [1, d_inner]
        outputs[1] = inputs[4];  // conv   -> [conv_dim, 4]
        outputs[2] = inputs[10]; // ssm    -> [H, D, D]
        return 0;
    }
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* io,
                                   int32_t nb_inputs, int32_t nb_outputs) noexcept override {
        if (nb_inputs != kNumInputs || nb_outputs != kNumOutputs)
            return false;
        auto const& d = io[pos].desc;
        if (d.format != nvinfer1::TensorFormat::kLINEAR)
            return false;
        if (pos == 3 && fuse_ab_)
            return d.type == nvinfer1::DataType::kFLOAT;
        if (pos <= 3)
            return is_half_type(d.type) && d.type == io[0].desc.type;
        // conv state in/out and out use the activation type; the rest is fp32.
        if (pos == 4 || pos == 11 || pos == 12)
            return d.type == io[0].desc.type;
        return d.type == nvinfer1::DataType::kFLOAT;
    }
    int32_t getNbOutputs() const noexcept override { return kNumOutputs; }
    std::size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                 nvinfer1::DynamicPluginTensorDesc const*,
                                 int32_t) const noexcept override {
        return workspace_bytes();
    }
    char const* getTimingCacheID() noexcept override { return nullptr; }
    char const* getMetadataString() noexcept override { return nullptr; }

    // ---- Runtime
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const*, int32_t,
                          nvinfer1::PluginTensorDesc const*, int32_t) noexcept override {
        return 0;
    }
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        return clone();
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const*, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override {
        if (!valid_ || workspace == nullptr)
            return -1;
        std::array<LoadedKernel, kNumStages> kernels;
        for (int32_t s = 0; s < kNumStages; ++s)
            if (!get_kernel(cubins_[s], kStageSymbol[s], kernels[s]))
                return -1;

        int32_t const H = num_heads_, NK = num_kv_heads_, D = head_dim_, CD = conv_dim_;
        auto* const base = static_cast<char*>(workspace);
        void* const y = base;
        void* const g = base + align_up(static_cast<std::size_t>(CD) * act_bytes_);
        void* const beta = static_cast<char*>(g) + align_up(static_cast<std::size_t>(H) * 4);
        void* const o = static_cast<char*>(beta) + align_up(static_cast<std::size_t>(H) * act_bytes_);
        auto* const yb = static_cast<char*>(y);
        std::size_t const qk_bytes = static_cast<std::size_t>(NK) * D * act_bytes_;
        auto cast = [](void const* p) { return const_cast<void*>(p); };


        // 1. conv1d + SiLU
        std::array<Arr, 6> conv{{
            {cast(inputs[0]), 1, {CD, 0, 0, 0}},          // qkv as [CD]
            {cast(inputs[4]), 2, {CD, kConvKernel, 0, 0}}, // state in
            {cast(inputs[5]), 2, {CD, kConvKernel, 0, 0}}, // weight
            {cast(inputs[6]), 1, {CD, 0, 0, 0}},          // bias
            {y, 1, {CD, 0, 0, 0}},                         // y
            {outputs[1], 2, {CD, kConvKernel, 0, 0}},      // state out
        }};
        unsigned const conv_grid = static_cast<unsigned>((CD + conv_block_ - 1) / conv_block_);
        if (!launch(kernels[0], {conv_grid, 1, 1}, static_cast<unsigned>(threads_[0]), conv.data(), 6, stream))
            return -1;

        // 2. gate / beta (with the a and b projections when fuse_ab)
        int32_t const K = hidden_size_;
        std::array<Arr, 6> gate{{
            {cast(inputs[3]), 1, {H, 0, 0, 0}}, // b
            {cast(inputs[2]), 1, {H, 0, 0, 0}}, // a
            {cast(inputs[7]), 1, {H, 0, 0, 0}}, // A
            {cast(inputs[8]), 1, {H, 0, 0, 0}}, // dt_bias
            {beta, 1, {H, 0, 0, 0}},
            {g, 1, {H, 0, 0, 0}},
        }};
        if (fuse_ab_) {
            gate[0] = {cast(inputs[2]), 1, {K, 0, 0, 0}};     // x
            gate[1] = {cast(inputs[3]), 2, {2 * H, K, 0, 0}}; // w_ab
        }
        unsigned const gate_grid = fuse_ab_ ? static_cast<unsigned>(H) : 1u;
        if (!launch(kernels[1], {gate_grid, 1, 1}, static_cast<unsigned>(threads_[1]), gate.data(), 6, stream))
            return -1;

        // 3. delta rule; q, k, v are slices of y
        std::array<Arr, 8> gdn{{
            {yb, 4, {1, 1, NK, D}},
            {yb + qk_bytes, 4, {1, 1, NK, D}},
            {yb + 2 * qk_bytes, 4, {1, 1, H, D}},
            {g, 3, {1, 1, H, 0}},
            {beta, 3, {1, 1, H, 0}},
            {o, 4, {1, 1, H, D}},
            {cast(inputs[10]), 4, {1, H, D, D}},
            {outputs[2], 4, {1, H, D, D}},
        }};
        if (!launch(kernels[2], {static_cast<unsigned>(H), static_cast<unsigned>(D / block_v_), 1}, static_cast<unsigned>(threads_[2]),
                    gdn.data(), 8, stream))
            return -1;

        // 4. gated RMSNorm
        std::array<Arr, 4> norm{{
            {o, 2, {H, D, 0, 0}},
            {cast(inputs[1]), 2, {H, D, 0, 0}},
            {cast(inputs[9]), 2, {H, D, 0, 0}},
            {outputs[0], 2, {H, D, 0, 0}},
        }};
        if (!launch(kernels[3], {static_cast<unsigned>(H), 1, 1}, static_cast<unsigned>(threads_[3]), norm.data(), 4, stream))
            return -1;
        return 0;
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        fields_.clear();
        fields_.push_back({"num_heads", &num_heads_, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"num_kv_heads", &num_kv_heads_, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"head_dim", &head_dim_, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"conv_dim", &conv_dim_, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"act_bytes", &act_bytes_, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"conv_block", &conv_block_, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"block_v", &block_v_, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"fuse_ab", &fuse_ab_, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"hidden_size", &hidden_size_, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"threads", threads_.data(), nvinfer1::PluginFieldType::kINT32,
                           kNumStages});
        for (int32_t s = 0; s < kNumStages; ++s)
            fields_.push_back({kStageField[s], cubins_[s].data(), nvinfer1::PluginFieldType::kINT8,
                               static_cast<int32_t>(cubins_[s].size())});
        fc_.nbFields = static_cast<int32_t>(fields_.size());
        fc_.fields = fields_.data();
        return &fc_;
    }

  private:
    std::size_t workspace_bytes() const noexcept {
        std::size_t const H = num_heads_, D = head_dim_, CD = conv_dim_;
        return align_up(CD * act_bytes_) + align_up(H * 4) + align_up(H * act_bytes_) +
               align_up(H * D * act_bytes_);
    }

    int32_t num_heads_{0};
    int32_t num_kv_heads_{0};
    int32_t head_dim_{0};
    int32_t conv_dim_{0};
    int32_t act_bytes_{0};
    int32_t conv_block_{0};
    int32_t block_v_{0};
    int32_t fuse_ab_{0};
    int32_t hidden_size_{0};
    std::array<std::vector<char>, kNumStages> cubins_;
    std::array<int32_t, kNumStages> threads_{};
    bool valid_{false};
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
};

class Qwen38GdnDecodeTileGymCreator final : public nvinfer1::IPluginCreatorV3One {
  public:
    Qwen38GdnDecodeTileGymCreator() noexcept {
        fields_.nbFields = 0;
        fields_.fields = nullptr;
    }
    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const* fc,
                                      nvinfer1::TensorRTPhase) noexcept override {
        if (fc == nullptr)
            return nullptr;
        auto* plugin = new (std::nothrow) Qwen38GdnDecodeTileGymPlugin(*fc);
        if (plugin != nullptr && !plugin->isValid()) {
            delete plugin;
            return nullptr;
        }
        return plugin;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return Qwen38GdnDecodeTileGymPlugin::kPLUGIN_NAME;
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return Qwen38GdnDecodeTileGymPlugin::kPLUGIN_VERSION;
    }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

  private:
    nvinfer1::PluginFieldCollection fields_{};
};

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::Qwen38GdnDecodeTileGymCreator>
    pluginRegistrarQwen38GdnDecodeTileGym{};
