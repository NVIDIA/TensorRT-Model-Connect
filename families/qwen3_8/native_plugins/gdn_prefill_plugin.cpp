/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Gated DeltaNet prefill chunk for Qwen3.8 (batch 1, seq_len tokens per call) as six
// cuTile kernels (see ../cutile/gdn_prefill.py): conv1d+SiLU, conv state, gate/beta, chunked delta
// rule (intra-chunk, inter-chunk), gated RMSNorm.
//
// Static mode: exactly seq_len rows per call. Dynamic mode (field "dynamic" = 1): the row count T
// is read from the input shapes at run time (1 <= T <= seq_len, the profile maximum), so a prompt of
// up to seq_len tokens is one call and needs no padding.
// A prompt of N tokens runs as ceil(N / seq_len) calls. The last call may be padded: `valid_len`
// gives the number of real tokens. Padding does not change the states, and output rows at or
// after `valid_len` are zero.
//
// Inputs : qkv[T,conv_dim] z[T,d_inner] a[T,H] b[T,H]            (fp16 or bf16, same type)
//          conv_state[conv_dim,4]                                 (same type as qkv)
//          conv_w[conv_dim,4] conv_b[conv_dim] A[H] dt_bias[H]
//          gamma[d_inner] ssm_state[H,D,D]                        (fp32)
//          valid_len[1]                                           (int32, 1 <= valid_len <= T)
// Outputs: out[T,d_inner], conv_state'[conv_dim,4]  (same type as qkv)
//          ssm_state'[H,D,D]                                      (fp32)
// Workspace: y[T,conv_dim] (act), g[T,H] (fp32), beta[T,H] (act), o[T,H,D] (act),
//            q, k, v_corr, k_cumdecay [H,T,D] (fp32), g_cum [H,T] (fp32).

#include <NvInferRuntime.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <string>
#include <vector>

#include "cubin_launch.h"

namespace trtmc {
namespace {

constexpr int32_t kNumInputs = 12;
constexpr int32_t kNumOutputs = 3;
constexpr int32_t kConvKernel = 4;
constexpr int32_t kNumStages = 6;
constexpr char const* kStageField[kNumStages] = {"cubin_conv", "cubin_conv_state", "cubin_gate",
                                                 "cubin_intra", "cubin_inter", "cubin_norm"};
constexpr char const* kStageSymbol[kNumStages] = {"gdnp_conv", "gdnp_conv_state", "gdnp_gate",
                                                  "gdnp_intra", "gdnp_inter", "gdnp_norm"};

} // namespace

class Qwen38GdnPrefillTileGymPlugin final : public nvinfer1::IPluginV3,
                                            public nvinfer1::IPluginV3OneCore,
                                            public nvinfer1::IPluginV3OneBuild,
                                            public nvinfer1::IPluginV3OneRuntime {
  public:
    static constexpr nvinfer1::AsciiChar const* kPLUGIN_NAME = "Qwen38GdnPrefillTileGym";
    static constexpr nvinfer1::AsciiChar const* kPLUGIN_VERSION = "1";

    explicit Qwen38GdnPrefillTileGymPlugin(nvinfer1::PluginFieldCollection const& fc) noexcept {
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
                else if (name == "seq_len") seq_len_ = as_int();
                else if (name == "act_bytes") act_bytes_ = as_int();
                else if (name == "conv_block") conv_block_ = as_int();
                else if (name == "block_v") block_v_ = as_int();
                else if (name == "chunk") chunk_ = as_int();
                else if (name == "dynamic") dynamic_ = as_int();
            } else if (f.type == nvinfer1::PluginFieldType::kINT8) {
                for (int32_t s = 0; s < kNumStages; ++s)
                    if (name == kStageField[s])
                        cubins_[s].assign(static_cast<char const*>(f.data),
                                          static_cast<char const*>(f.data) + f.length);
            }
        }
        valid_ = num_heads_ > 0 && num_kv_heads_ > 0 && head_dim_ > 0 && conv_dim_ > 0 &&
                 seq_len_ > 0 && (act_bytes_ == 2) && conv_block_ > 0 && block_v_ > 0 &&
                 head_dim_ % block_v_ == 0 && chunk_ > 0 &&
                 (dynamic_ != 0 || seq_len_ % chunk_ == 0);
        for (int32_t s = 0; s < kNumStages; ++s)
            valid_ = valid_ && !cubins_[s].empty() && threads_[s] > 0;
    }
    Qwen38GdnPrefillTileGymPlugin(Qwen38GdnPrefillTileGymPlugin const&) = default;

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
        return new (std::nothrow) Qwen38GdnPrefillTileGymPlugin(*this);
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
        outputs[0] = inputs[1];  // z      -> [T, d_inner]
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
        if (pos <= 3)
            return is_half_type(d.type) && d.type == io[0].desc.type;
        if (pos == 11)
            return d.type == nvinfer1::DataType::kINT32;
        // conv state in/out and out use the activation type; the rest is fp32.
        if (pos == 4 || pos == 12 || pos == 13)
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

    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override {
        if (!valid_ || workspace == nullptr)
            return -1;
        int32_t const rows = dynamic_ != 0 ? static_cast<int32_t>(input_desc[0].dims.d[0]) : seq_len_;
        if (rows < 1 || rows > seq_len_)
            return -1;
        std::array<LoadedKernel, kNumStages> kernels;
        for (int32_t s = 0; s < kNumStages; ++s)
            if (!get_kernel(cubins_[s], kStageSymbol[s], kernels[s]))
                return -1;

        int32_t const T = rows, H = num_heads_, NK = num_kv_heads_, D = head_dim_,
                      CD = conv_dim_;
        // Buffers are laid out for the maximum row count so the carve-up does not depend on T.
        std::size_t const Tm = seq_len_;
        auto* const base = static_cast<char*>(workspace);
        void* const y = base;
        void* const g = base + align_up(Tm * CD * act_bytes_);
        void* const beta = static_cast<char*>(g) + align_up(Tm * H * 4);
        void* const o = static_cast<char*>(beta) + align_up(Tm * H * act_bytes_);
        std::size_t const hd_bytes = align_up(static_cast<std::size_t>(H) * Tm * D * 4);
        int32_t const CS = chunk_;
        int32_t const NC = (T + CS - 1) / CS;
        char* const wq = static_cast<char*>(o) + align_up(Tm * H * D * act_bytes_);
        void* const qn = wq;
        void* const kn = wq + hd_bytes;
        void* const vc = wq + 2 * hd_bytes;
        void* const kc = wq + 3 * hd_bytes;
        void* const gc = wq + 4 * hd_bytes;
        auto* const yb = static_cast<char*>(y);
        std::size_t const qk_bytes = static_cast<std::size_t>(NK) * D * act_bytes_;
        auto cast = [](void const* p) { return const_cast<void*>(p); };
        std::array<int32_t, kMaxRank> const row_strides{T * CD, CD, D, 1};
        unsigned const conv_tiles = static_cast<unsigned>((CD + conv_block_ - 1) / conv_block_);

        // 1. conv1d + SiLU over [old state tail, x]
        std::array<Arr, 5> conv{{
            {cast(inputs[0]), 2, {T, CD, 0, 0}},           // qkv
            {cast(inputs[4]), 2, {CD, kConvKernel, 0, 0}}, // state in
            {cast(inputs[5]), 2, {CD, kConvKernel, 0, 0}}, // weight
            {cast(inputs[6]), 1, {CD, 0, 0, 0}},           // bias
            {y, 2, {T, CD, 0, 0}},                         // y
        }};
        if (!launch(kernels[0], {conv_tiles, static_cast<unsigned>(T), 1}, static_cast<unsigned>(threads_[0]), conv.data(), 5, stream))
            return -1;

        // 2. new conv state = last 4 real inputs
        std::array<Arr, 4> conv_state{{
            {cast(inputs[0]), 2, {T, CD, 0, 0}},
            {cast(inputs[4]), 2, {CD, kConvKernel, 0, 0}},
            {cast(inputs[11]), 1, {1, 0, 0, 0}},           // valid_len
            {outputs[1], 2, {CD, kConvKernel, 0, 0}},
        }};
        if (!launch(kernels[1], {conv_tiles, 1, 1}, static_cast<unsigned>(threads_[1]), conv_state.data(), 4, stream))
            return -1;

        // 3. gate / beta for all T rows
        std::array<Arr, 7> gate{{
            {cast(inputs[3]), 2, {T, H, 0, 0}}, // b
            {cast(inputs[2]), 2, {T, H, 0, 0}}, // a
            {cast(inputs[7]), 1, {H, 0, 0, 0}}, // A
            {cast(inputs[8]), 1, {H, 0, 0, 0}}, // dt_bias
            {cast(inputs[11]), 1, {1, 0, 0, 0}}, // valid_len
            {beta, 2, {T, H, 0, 0}},
            {g, 2, {T, H, 0, 0}},
        }};
        if (!launch(kernels[2], {static_cast<unsigned>(T), 1, 1}, static_cast<unsigned>(threads_[2]), gate.data(), 7, stream))
            return -1;

        // 4a. per chunk: decay, triangular solve (WY form), corrected values. q, k, v are strided
        //     views of y.
        std::array<Arr, 10> intra{{
            {yb, 4, {1, T, NK, D}, row_strides},
            {yb + qk_bytes, 4, {1, T, NK, D}, row_strides},
            {yb + 2 * qk_bytes, 4, {1, T, H, D}, row_strides},
            {g, 3, {1, T, H, 0}},
            {beta, 3, {1, T, H, 0}},
            {qn, 4, {H, NC, CS, D}},
            {kn, 4, {H, NC, CS, D}},
            {vc, 4, {H, NC, CS, D}},
            {kc, 4, {H, NC, CS, D}},
            {gc, 3, {H, NC, CS, 0}},
        }};
        if (!launch(kernels[3], {static_cast<unsigned>(H), static_cast<unsigned>(NC), 1}, static_cast<unsigned>(threads_[3]),
                    intra.data(), 10, stream))
            return -1;

        // 4b. state passing over the chunks and the chunk outputs
        std::array<Arr, 8> inter{{
            {qn, 4, {H, NC, CS, D}},
            {kn, 4, {H, NC, CS, D}},
            {vc, 4, {H, NC, CS, D}},
            {kc, 4, {H, NC, CS, D}},
            {gc, 3, {H, NC, CS, 0}},
            {o, 4, {1, T, H, D}},
            {cast(inputs[10]), 4, {1, H, D, D}},
            {outputs[2], 4, {1, H, D, D}},
        }};
        if (!launch(kernels[4], {static_cast<unsigned>(H), static_cast<unsigned>(D / block_v_), 1}, static_cast<unsigned>(threads_[4]),
                    inter.data(), 8, stream))
            return -1;

        // 5. gated RMSNorm, zero for padding rows
        std::array<Arr, 5> norm{{
            {o, 2, {T * H, D, 0, 0}},
            {cast(inputs[1]), 2, {T * H, D, 0, 0}},
            {cast(inputs[9]), 2, {H, D, 0, 0}},
            {cast(inputs[11]), 1, {1, 0, 0, 0}},
            {outputs[0], 2, {T * H, D, 0, 0}},
        }};
        return launch(kernels[5], {static_cast<unsigned>(T * H), 1, 1}, static_cast<unsigned>(threads_[5]), norm.data(), 5, stream)
                   ? 0
                   : -1;
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        fields_.clear();
        fields_.push_back({"num_heads", &num_heads_, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"num_kv_heads", &num_kv_heads_, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"head_dim", &head_dim_, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"conv_dim", &conv_dim_, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"seq_len", &seq_len_, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"act_bytes", &act_bytes_, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"conv_block", &conv_block_, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"block_v", &block_v_, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"chunk", &chunk_, nvinfer1::PluginFieldType::kINT32, 1});
        fields_.push_back({"dynamic", &dynamic_, nvinfer1::PluginFieldType::kINT32, 1});
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
        std::size_t const T = seq_len_, H = num_heads_, D = head_dim_, CD = conv_dim_;
        return align_up(T * CD * act_bytes_) + align_up(T * H * 4) + align_up(T * H * act_bytes_) +
               align_up(T * H * D * act_bytes_) + 4 * align_up(H * T * D * 4) + align_up(H * T * 4);
    }

    int32_t num_heads_{0};
    int32_t num_kv_heads_{0};
    int32_t head_dim_{0};
    int32_t conv_dim_{0};
    int32_t seq_len_{0};
    int32_t act_bytes_{0};
    int32_t conv_block_{0};
    int32_t block_v_{0};
    int32_t chunk_{0};
    int32_t dynamic_{0};
    std::array<std::vector<char>, kNumStages> cubins_;
    std::array<int32_t, kNumStages> threads_{};
    bool valid_{false};
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
};

class Qwen38GdnPrefillTileGymCreator final : public nvinfer1::IPluginCreatorV3One {
  public:
    Qwen38GdnPrefillTileGymCreator() noexcept {
        fields_.nbFields = 0;
        fields_.fields = nullptr;
    }
    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const* fc,
                                      nvinfer1::TensorRTPhase) noexcept override {
        if (fc == nullptr)
            return nullptr;
        auto* plugin = new (std::nothrow) Qwen38GdnPrefillTileGymPlugin(*fc);
        if (plugin != nullptr && !plugin->isValid()) {
            delete plugin;
            return nullptr;
        }
        return plugin;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return Qwen38GdnPrefillTileGymPlugin::kPLUGIN_NAME;
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return Qwen38GdnPrefillTileGymPlugin::kPLUGIN_VERSION;
    }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

  private:
    nvinfer1::PluginFieldCollection fields_{};
};

} // namespace trtmc

static nvinfer1::PluginRegistrar<trtmc::Qwen38GdnPrefillTileGymCreator>
    pluginRegistrarQwen38GdnPrefillTileGym{};
