/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Reference validation only: replay token IDs through the production native
// decoder and state, exporting its logits without introducing another sampler.
#include "core/runtime/tensorrt/trt_module_impl.h"
#include "families/lfm2_moe/runtime/chat_templates.h"
#include "families/lfm2_moe/runtime/hybrid_state.h"
#include "families/lfm2_moe/runtime/plugin_helpers.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc {

// Read the context actually used by this test probe; never deserialize a second engine.
class TrtModuleImplTestPeer {
  public:
    static nvinfer1::ICudaEngine& engine(TrtModuleImpl& module) { return *module.engine_; }
    static nvinfer1::IExecutionContext& context(TrtModuleImpl& module) { return *module.ctx_; }
};

} // namespace trtmc

namespace {

using Json = nlohmann::json;
constexpr int kFirstPosition = 16, kLastPosition = 104;
constexpr int kHeads = 8, kHeadDim = 64, kConvWords = 2048 * 3;
constexpr std::array<int, 6> kAttentionLayers{2, 6, 10, 14, 18, 21};
constexpr std::array<int, 18> kConvLayers{0,  1,  3,  4,  5,  7,  8,  9,  11,
                                          12, 13, 15, 16, 17, 19, 20, 22, 23};

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void cuda_check(cudaError_t status) {
    require(status == cudaSuccess, "state observation CUDA transfer or synchronization failed");
}

std::vector<int64_t> dimensions(nvinfer1::Dims dims) {
    require(dims.nbDims >= 0, "state observation has unresolved dimensions");
    return {dims.d, dims.d + dims.nbDims};
}

template <typename T>
void write_values(const std::filesystem::path& path, const std::vector<T>& values) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    output.close();
    require(static_cast<bool>(output), "state observation payload write failed");
}

class StateCapture {
  public:
    StateCapture(const std::string& path, const Json& config) : root_(path) {
        metadata_ = {{"status", "capturing"},
                     {"rows_captured", 0},
                     {"positions", Json::array()},
                     {"rows", Json::array()},
                     {"attention_layers", kAttentionLayers},
                     {"conv_layers", kConvLayers},
                     {"prefix_checks", Json::array()},
                     {"prefix_anomaly", nullptr},
                     {"conv_prefix_checks", Json::array()},
                     {"conv_prefix_anomaly", nullptr},
                     {"present_anomaly", nullptr},
                     {"layout_admitted", false},
                     {"stream_admitted", false},
                     {"present_equal", true},
                     {"error_type", nullptr},
                     {"dtype", "BF16_uint16_bits"}};
        try {
            require(!std::filesystem::exists(root_), "state observation directory must be fresh");
            std::filesystem::create_directories(root_);
            can_write_ = true;
            const auto types = config.at("layer_types").get<std::vector<std::string>>();
            metadata_["layer_types"] = types;
            std::vector<int> attention, conv;
            for (size_t layer = 0; layer < types.size(); ++layer) {
                if (types[layer] == "full_attention")
                    attention.push_back(static_cast<int>(layer));
                else {
                    require(types[layer] == "conv", "state observation unknown layer type");
                    conv.push_back(static_cast<int>(layer));
                }
            }
            require(attention ==
                            std::vector<int>(kAttentionLayers.begin(), kAttentionLayers.end()) &&
                        conv == std::vector<int>(kConvLayers.begin(), kConvLayers.end()) &&
                        config.at("num_key_value_heads") == kHeads &&
                        config.at("head_dim") == kHeadDim && config.at("hidden_size") == 2048 &&
                        config.at("conv_L_cache") == 3,
                    "state observation runtime layer mapping differs");
        } catch (const std::exception& error) {
            fail(error);
        }
    }

    void step(trtmc::ITrtModule& module, trtmc::Lfm2MoeHybridState& state, int32_t token,
              int32_t phase) noexcept {
        try {
            if (failed_ || state.position() < kFirstPosition || state.position() > kLastPosition)
                return;
            capture(module, state, token, phase);
        } catch (const std::exception& error) {
            fail(error);
        }
    }

    void save() noexcept {
        if (!can_write_)
            return;
        try {
            metadata_["rows_captured"] = metadata_["positions"].size();
            metadata_["kv_compacted_length"] = compacted_length_;
            metadata_["kv_history_shape"] = {6, 2, kHeads, compacted_length_, kHeadDim};
            metadata_["conv_compacted_taps"] = conv_taps_;
            metadata_["conv_shape"] = {18, 2048, conv_taps_};
            if (!failed_)
                metadata_["status"] = metadata_["positions"].size() == 89 ? "captured" : "partial";
            write_values(root_ / "kv.u16", history_);
            write_values(root_ / "conv.u16", conv_history_);
            std::ofstream output(root_ / "metadata.json");
            output << metadata_.dump() << '\n';
            output.close();
            require(static_cast<bool>(output), "state observation metadata write failed");
        } catch (const std::exception& error) {
            fail(error);
        }
    }

  private:
    std::filesystem::path root_;
    Json metadata_;
    std::vector<uint16_t> history_, previous_kv_, conv_history_, previous_conv_;
    int compacted_length_{0};
    int conv_taps_{0};
    bool failed_{false};
    bool can_write_{false};

    void fail(const std::exception& error) noexcept {
        failed_ = true;
        try {
            metadata_["status"] = "unavailable";
            metadata_["error_type"] = "capture_error";
            metadata_["error_message"] = std::string(error.what()).substr(0, 256);
        } catch (...) {
        }
    }

    Json descriptor(trtmc::ITrtModule& module, const std::string& name, bool input,
                    const std::vector<int64_t>& shape, const std::vector<int64_t>& strides,
                    void* address) {
        auto* actual = dynamic_cast<trtmc::TrtModuleImpl*>(&module);
        require(actual != nullptr, "state observation requires the actual TensorRT backend");
        auto& engine = trtmc::TrtModuleImplTestPeer::engine(*actual);
        auto& context = trtmc::TrtModuleImplTestPeer::context(*actual);
        const auto profile = context.getOptimizationProfile();
        require(profile == module.profile_idx(), "state observation profile differs");
        const auto mode = input ? nvinfer1::TensorIOMode::kINPUT : nvinfer1::TensorIOMode::kOUTPUT;
        const int components = engine.getTensorComponentsPerElement(name.c_str(), profile);
        const int bytes = engine.getTensorBytesPerComponent(name.c_str(), profile);
        // Non-vectorized I/O can return -1 for vector-component queries. Actual
        // BF16 dtype and LINEAR strides establish the two-byte scalar storage.
        require(engine.getTensorIOMode(name.c_str()) == mode &&
                    engine.getTensorDataType(name.c_str()) == nvinfer1::DataType::kBF16 &&
                    engine.getTensorFormat(name.c_str(), profile) ==
                        nvinfer1::TensorFormat::kLINEAR &&
                    engine.getTensorVectorizedDim(name.c_str(), profile) == -1 &&
                    ((components == 1 && bytes == 2) || (components == -1 && bytes == -1)),
                "state observation requires BF16 scalar LINEAR I/O");
        const auto actual_shape = dimensions(context.getTensorShape(name.c_str()));
        const auto actual_strides = dimensions(context.getTensorStrides(name.c_str()));
        require(actual_shape == shape && actual_strides == strides,
                "state observation actual context layout differs");
        require(module.device_ptr(name) == address &&
                    context.getTensorAddress(name.c_str()) == address,
                "state observation actual context address differs");
        return {{"name", name},
                {"mode", input ? "INPUT" : "OUTPUT"},
                {"dtype", "BF16"},
                {"format", "LINEAR"},
                {"profile", profile},
                {"shape", actual_shape},
                {"strides", actual_strides},
                {"vectorized_dimension", -1},
                {"components_per_element", components},
                {"bytes_per_component", bytes},
                {"scalar_bytes", 2},
                {"address", reinterpret_cast<uintptr_t>(address)}};
    }

    void validate_tensor(const trtmc::DeviceTensor& tensor, cudaStream_t stream,
                         const std::vector<int64_t>& shape) {
        require(tensor.ok() && tensor.dtype() == trtmc::DType::kBFloat16 && tensor.shape() == shape,
                "state observation allocation differs");
        require(tensor.stream() == stream, "state observation stream differs");
    }

    void pack_kv(trtmc::ITrtModule& module, trtmc::Lfm2MoeHybridState& state, Json& descriptors,
                 std::vector<uint16_t>& packed) {
        const int capacity = state.max_length(), length = state.position();
        const auto stream = module.stream();
        const std::vector<int64_t> shape{1, kHeads, capacity, kHeadDim};
        const std::vector<int64_t> strides{int64_t(kHeads) * capacity * kHeadDim,
                                           int64_t(capacity) * kHeadDim, kHeadDim, 1};
        packed.resize(6 * 2 * kHeads * length * kHeadDim);
        for (int layer = 0; layer < 6; ++layer) {
            for (int kind = 0; kind < 2; ++kind) {
                auto& tensor =
                    kind == 0 ? state.kv_cache().cache_k(layer) : state.kv_cache().cache_v(layer);
                validate_tensor(tensor, stream, {capacity, kHeads * kHeadDim});
                const auto suffix = std::string(kind == 0 ? "k_" : "v_") + std::to_string(layer);
                descriptors.push_back(
                    descriptor(module, "cache_" + suffix, true, shape, strides, tensor.data()));
                descriptors.push_back(
                    descriptor(module, "present_" + suffix, false, shape, strides, tensor.data()));
                cuda_check(cudaMemcpy2DAsync(
                    packed.data() + (layer * 2 + kind) * kHeads * length * kHeadDim,
                    length * kHeadDim * sizeof(uint16_t), tensor.data(),
                    capacity * kHeadDim * sizeof(uint16_t), length * kHeadDim * sizeof(uint16_t),
                    kHeads, cudaMemcpyDeviceToHost, stream));
            }
        }
    }

    void compact_kv(const std::vector<uint16_t>& packed, int length) {
        if (!metadata_["prefix_anomaly"].is_null()) {
            metadata_["prefix_checks"].push_back(nullptr);
            return;
        }
        size_t differences = 0, first = 0;
        for (int row = 0; row < 6 * 2 * kHeads; ++row) {
            for (int word = 0; word < compacted_length_ * kHeadDim; ++word) {
                const size_t current = row * length * kHeadDim + word;
                if (packed[current] != previous_kv_[row * compacted_length_ * kHeadDim + word]) {
                    if (differences++ == 0)
                        first = current;
                }
            }
        }
        if (differences) {
            metadata_["prefix_checks"].push_back(false);
            const auto channel = first % kHeadDim;
            const auto position = first / kHeadDim % length;
            const auto head = first / (kHeadDim * length) % kHeads;
            const auto kind = first / (kHeadDim * length * kHeads) % 2;
            const auto layer = first / (kHeadDim * length * kHeads * 2);
            metadata_["prefix_anomaly"] = {{"row", length - kFirstPosition},
                                           {"position", length},
                                           {"first_index", {layer, kind, head, position, channel}},
                                           {"differing_words", differences},
                                           {"file", "kv-prefix-anomaly.u16"}};
            write_values(root_ / "kv-prefix-anomaly.u16", packed);
        } else {
            metadata_["prefix_checks"].push_back(compacted_length_ ? Json(true) : Json(nullptr));
            // packed is an independent logical BHND copy; replacing this bounded history
            // stores the initial 16 rows plus each observed append exactly once.
            history_ = packed;
            previous_kv_ = packed;
            compacted_length_ = length;
        }
    }

    void compact_conv(const std::vector<uint16_t>& values, int length) {
        if (!metadata_["conv_prefix_anomaly"].is_null()) {
            metadata_["conv_prefix_checks"].push_back(nullptr);
            return;
        }
        size_t differences = 0, first = 0;
        if (!previous_conv_.empty()) {
            for (int channel = 0; channel < 18 * 2048; ++channel) {
                for (int tap = 0; tap < 2; ++tap) {
                    const size_t index = channel * 3 + tap;
                    if (values[index] != previous_conv_[index + 1]) {
                        if (differences++ == 0)
                            first = index;
                    }
                }
            }
        }
        if (differences) {
            metadata_["conv_prefix_checks"].push_back(false);
            metadata_["conv_prefix_anomaly"] = {
                {"row", length - kFirstPosition},
                {"position", length},
                {"first_index", {first / kConvWords, first / 3 % 2048, first % 3}},
                {"differing_words", differences},
                {"file", "conv-shift-anomaly.u16"}};
            write_values(root_ / "conv-shift-anomaly.u16", values);
            return;
        }
        metadata_["conv_prefix_checks"].push_back(previous_conv_.empty() ? Json(nullptr)
                                                                         : Json(true));
        if (previous_conv_.empty()) {
            conv_history_ = values;
            conv_taps_ = 3;
        } else {
            std::vector<uint16_t> appended(18 * 2048 * (conv_taps_ + 1));
            for (int channel = 0; channel < 18 * 2048; ++channel) {
                std::copy_n(conv_history_.begin() + channel * conv_taps_, conv_taps_,
                            appended.begin() + channel * (conv_taps_ + 1));
                appended[channel * (conv_taps_ + 1) + conv_taps_] = values[channel * 3 + 2];
            }
            conv_history_ = std::move(appended);
            ++conv_taps_;
        }
        previous_conv_ = values;
    }

    void capture(trtmc::ITrtModule& module, trtmc::Lfm2MoeHybridState& state, int32_t token,
                 int32_t phase) {
        const int length = state.position();
        require(state.kv_cache().num_layers() == 6 && state.conv_state().num_layers() == 18 &&
                    state.max_length() >= kLastPosition,
                "state observation requires the declared MoE geometry");
        require(length == kFirstPosition + static_cast<int>(metadata_["positions"].size()) &&
                    phase == (length == kFirstPosition ? 0 : 1),
                "state observation position or phase differs");
        const auto stream = module.stream();
        // advance() enqueues convolution state D2D copies after module.sync().
        cuda_check(cudaStreamSynchronize(stream));
        Json descriptors = Json::array();
        std::vector<uint16_t> kv, conv(18 * kConvWords), present(18 * kConvWords);
        // Keep host destinations alive until any already queued copies finish,
        // including when a later metadata check fails.
        struct Drain {
            cudaStream_t stream;
            ~Drain() { cudaStreamSynchronize(stream); }
        } drain{stream};
        pack_kv(module, state, descriptors, kv);
        for (int layer = 0; layer < 18; ++layer) {
            auto& stable = state.conv_state().state(layer);
            auto& output = state.conv_state().present(layer);
            validate_tensor(stable, stream, {2048, 3});
            validate_tensor(output, stream, {2048, 3});
            require(stable.data() != output.data(), "state observation convolution buffers alias");
            const auto suffix = "_" + std::to_string(layer);
            descriptors.push_back(
                descriptor(module, "conv_state" + suffix, true, {2048, 3}, {3, 1}, stable.data()));
            descriptors.push_back(descriptor(module, "present_conv" + suffix, false, {2048, 3},
                                             {3, 1}, output.data()));
            cuda_check(cudaMemcpyAsync(conv.data() + layer * kConvWords, stable.data(),
                                       kConvWords * sizeof(uint16_t), cudaMemcpyDeviceToHost,
                                       stream));
            cuda_check(cudaMemcpyAsync(present.data() + layer * kConvWords, output.data(),
                                       kConvWords * sizeof(uint16_t), cudaMemcpyDeviceToHost,
                                       stream));
        }
        cuda_check(cudaStreamSynchronize(stream));
        if (metadata_.contains("descriptors"))
            require(metadata_["descriptors"] == descriptors,
                    "state observation I/O identity changed");
        else
            metadata_["descriptors"] = descriptors;
        if (conv != present && metadata_["present_anomaly"].is_null()) {
            metadata_["present_equal"] = false;
            metadata_["present_anomaly"] = {{"row", length - kFirstPosition},
                                            {"position", length},
                                            {"file", "conv-present-anomaly.u16"}};
            write_values(root_ / "conv-present-anomaly.u16", present);
        }
        compact_kv(kv, length);
        compact_conv(conv, length);
        metadata_["positions"].push_back(length);
        metadata_["rows"].push_back({{"position", length},
                                     {"write_index", length - 1},
                                     {"token", token},
                                     {"phase", phase},
                                     {"stream", reinterpret_cast<uintptr_t>(stream)},
                                     {"layout_rechecked", true}});
        metadata_["layout_admitted"] = true;
        metadata_["stream_admitted"] = true;
    }
};

} // namespace

int main(int argc, char** argv) {
    try {
        const bool encode_only = argc == 4 && std::string(argv[1]) == "--encode-prompt";
        if (argc != 2 && !encode_only)
            throw std::invalid_argument(
                "usage: lfm25_moe_logits_probe [--encode-prompt] bundle [prompt]");
        trtmc::BundleReader bundle(argv[encode_only ? 2 : 1]);
        const auto cfg =
            nlohmann::json::parse(trtmc::lfm2_moe::require_text_section(bundle, "runtime.json"));
        if (encode_only) {
            trtmc::Lfm2MoeTextGenConfig generation;
            generation.id_bos = cfg.at("bos_token_id");
            trtmc::lfm2_moe::apply_chat_template(bundle, generation);
            auto tokenizer = trtmc::lfm2_moe::create_tokenizer(bundle);
            std::string rendered = argv[3];
            if (!generation.chat_template_format.empty())
                rendered =
                    trtmc::lfm2_moe_apply_chat_template(generation.chat_template_format, rendered);
            auto ids = tokenizer->encode(rendered);
            if (ids.size() >= 2 && generation.id_bos >= 0 && ids[0] == generation.id_bos &&
                ids[1] == generation.id_bos)
                ids.erase(ids.begin());
            if (ids.empty() && generation.id_bos >= 0)
                ids.push_back(generation.id_bos);
            std::cout << nlohmann::json({{"prompt_ids", ids}}).dump() << std::endl;
            return 0;
        }
        std::unique_ptr<trtmc::IBackend, decltype(&trtmc_destroy_backend)> backend(
            trtmc_create_backend(), trtmc_destroy_backend);
        auto module = trtmc::lfm2_moe::load_module(
            *backend, trtmc::lfm2_moe::require_section(bundle, "engine.plan"));
        const auto stream = module->stream();
        const auto dtype = trtmc::lfm2_moe::state_dtype(cfg.at("precision"));
        auto kv = std::make_unique<trtmc::Lfm2MoeKvCache>(
            cfg.at("num_attention_layers"), cfg.at("max_cache_length"),
            cfg.at("num_key_value_heads"), cfg.at("head_dim"), stream, dtype,
            trtmc::lfm2_moe::kv_names(cfg.at("num_attention_layers")));
        auto conv = std::make_unique<trtmc::Lfm2MoeConvState>(
            cfg.at("num_conv_layers"), cfg.at("hidden_size"), cfg.at("conv_L_cache"), stream,
            dtype);
        trtmc::Lfm2MoeHybridState state(std::move(kv), std::move(conv));
        if (!state.ok())
            throw std::runtime_error("state allocation failed");
        state.bind_to(*module);
        const int vocab = cfg.at("vocab_size");
        std::vector<float> logits(vocab);
        std::unique_ptr<StateCapture> capture;
        std::cout << "{\"ready\":true}" << std::endl;
        for (std::string line; std::getline(std::cin, line);) {
            const auto request = nlohmann::json::parse(line);
            if (request.value("reset", false)) {
                capture.reset();
                state.reset();
                module->reset_execution_context();
                state.bind_to(*module);
            }
            if (request.contains("state_capture"))
                capture = std::make_unique<StateCapture>(
                    request.at("state_capture").get<std::string>(), cfg);
            const auto tokens = request.at("tokens").get<std::vector<int32_t>>();
            if (tokens.empty() ||
                tokens.size() > static_cast<size_t>(state.max_length() - state.position()))
                throw std::invalid_argument("empty token input or context capacity exceeded");
            for (int32_t token : tokens) {
                if (token < 0 || token >= vocab)
                    throw std::invalid_argument("token ID is outside vocabulary");
                trtmc::TensorMap inputs;
                inputs["token_id"] = trtmc::Tensor{&token, {1}, trtmc::DType::kInt32};
                int32_t phase = request.value("decode_phase", false) ? 1 : 0;
                if (module->has_input("decode_phase"))
                    inputs["decode_phase"] = trtmc::Tensor{&phase, {1, 1}, trtmc::DType::kInt32};
                state.prepare_step(inputs);
                module->forward_async(inputs);
                module->sync();
                state.advance();
                if (capture)
                    capture->step(*module, state, token, phase);
            }
            if (capture)
                capture->save();
            if (cudaMemcpy(logits.data(), module->device_ptr("logits"),
                           logits.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("logits transfer failed");
            std::ofstream output(request.at("output").get<std::string>(), std::ios::binary);
            output.write(reinterpret_cast<const char*>(logits.data()),
                         logits.size() * sizeof(float));
            output.close();
            if (!output)
                throw std::runtime_error("logits write failed");
            std::cout
                << nlohmann::json({{"position", state.position()}, {"vocab_size", vocab}}).dump()
                << std::endl;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
