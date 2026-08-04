/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-VL-CPP-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-FAC-01
// Intent:         LocateAnything native-KV text and VL prefill behavior with mock modules;
//                 grounding-token decode preservation; chunked prefill cache positions;
//                 capacity admission before enqueue; no unnecessary final decode launch
// Preconditions:  TRT + CUDA GPU available
// Postconditions: Pipeline prefills text and VL prompts through native KV only, writes each
//                 chunk at its logical cache position, and rejects capacity overflow
// =============================================================================

// =============================================================================
// Test suite: LocateAnythingPipeline — vision-language generation
// =============================================================================

#include "runtime/models/locateanything/kv_cache.h"
#include "runtime/models/locateanything/pipeline.h"
#include "runtime/models/locateanything/plugin_helpers.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

static int failures = 0;
static void check(bool c, const char* n) {
    if (!c) {
        std::cerr << "FAIL: " << n << '\n';
        ++failures;
    }
}

struct CountingTextStats {
    int32_t calls{0};
    std::unordered_map<std::string, std::vector<int64_t>> shapes;
    std::unordered_map<std::string, std::vector<float>> float_values;
    std::vector<int32_t> write_indices;
    std::vector<int32_t> kv_lengths;
};

class CountingTextModule final : public trtmc::ITrtModule {
  public:
    CountingTextModule(std::shared_ptr<CountingTextStats> stats, bool prefill, cudaStream_t stream,
                       int32_t profile_limit = 8)
        : stats_(std::move(stats)), prefill_(prefill), profile_limit_(profile_limit),
          stream_(stream) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++stats_->calls;
        stats_->shapes.clear();
        stats_->float_values.clear();
        for (const auto& [name, tensor] : inputs) {
            stats_->shapes[name] = tensor.shape;
            if (tensor.dtype == trtmc::DType::kFloat32) {
                const auto* begin = static_cast<const float*>(tensor.data);
                stats_->float_values[name] =
                    std::vector<float>(begin, begin + static_cast<std::ptrdiff_t>(tensor.numel()));
            }
        }
        stats_->write_indices.push_back(
            *static_cast<const int32_t*>(inputs.at("cache_write_indices").data));
        stats_->kv_lengths.push_back(
            *static_cast<const int32_t*>(inputs.at("key_value_lengths").data));
        return {{"logits", trtmc::Tensor{logits_.data(), {1, 4}, trtmc::DType::kFloat32}}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap& inputs) override { (void)forward(inputs); }
    void sync() override {}
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        if (name == "cache_write_indices" || name == "key_value_lengths")
            return true;
        return name == "token_id" || name == "position_id" || name == "input_embed" ||
               name == "use_input_embed" || name == "cache_k_0" || name == "cache_v_0";
    }
    bool has_output(const std::string& name) const override {
        return name == "logits" || name == "present_k_0" || name == "present_v_0";
    }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        if (name == "cache_write_indices" || name == "key_value_lengths" || name == "token_id" ||
            name == "position_id")
            return trtmc::DType::kInt32;
        return trtmc::DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "cache_write_indices" || name == "key_value_lengths")
            return {1};
        if (name == "cache_k_0" || name == "cache_v_0" || name == "present_k_0" ||
            name == "present_v_0")
            return {1, 1, 8, 4};
        return {};
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        if (name == "token_id")
            return {profile_limit_};
        return tensor_shape(name);
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& name) const override {
        const auto binding = bindings_.find(name);
        if (binding != bindings_.end())
            return binding->second;
        return nullptr;
    }
    void bind_external(const std::string& name, void* pointer) override {
        bindings_[name] = pointer;
        if (name.rfind("cache_", 0) == 0)
            bindings_["present_" + name.substr(6)] = pointer;
    }
    int32_t input_rank(const std::string& name) const override {
        return name == "token_id" || name == "position_id" || name == "cache_write_indices" ||
                       name == "key_value_lengths"
                   ? 1
                   : 2;
    }
    bool input_is_dynamic(const std::string&) const override { return prefill_; }
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    std::shared_ptr<CountingTextStats> stats_;
    bool prefill_{false};
    int32_t profile_limit_{8};
    cudaStream_t stream_{nullptr};
    mutable std::unordered_map<std::string, void*> bindings_;
    std::vector<float> logits_{0.1F, 0.2F, 0.9F, 0.3F};
};

class FakeSequenceVisionModule final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap&) override {
        return {{"image_features", trtmc::Tensor{features_.data(), {1, 4}, trtmc::DType::kFloat32}},
                {"deepstack_features_0",
                 trtmc::Tensor{deepstack_.data(), {1, 4}, trtmc::DType::kFloat32}}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override {
        return {{"pixel_values", {3, 4, 4}, trtmc::DType::kFloat32, true}};
    }
    std::vector<trtmc::TensorInfo> output_info() const override {
        return {{"image_features", {1, 4}, trtmc::DType::kFloat32, false}};
    }
    bool has_input(const std::string& name) const override { return name == "pixel_values"; }
    bool has_output(const std::string& name) const override {
        return name == "image_features" || name == "deepstack_features_0";
    }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    std::vector<float> features_{10.0F, 11.0F, 12.0F, 13.0F};
    std::vector<float> deepstack_{20.0F, 21.0F, 22.0F, 23.0F};
};

class RejectingVisionBackend final : public trtmc::IBackend {
  public:
    std::unique_ptr<trtmc::ITrtModule> create_module(const void*, size_t,
                                                     const trtmc::ModuleCreateOptions&) override {
        ++create_calls;
        return nullptr;
    }

    trtmc::BackendDualProfileModules
    create_dual_profile_modules(const void*, size_t, const trtmc::ModuleCreateOptions&) override {
        return {};
    }

    trtmc::BackendProfileModules create_profile_modules(const void*, size_t,
                                                        const trtmc::ModuleCreateOptions&,
                                                        const std::vector<int32_t>&) override {
        return {};
    }

    trtmc::BackendContextModules
    create_context_modules(const void*, size_t,
                           const std::vector<trtmc::ModuleCreateOptions>&) override {
        return {};
    }

    const char* name() const override { return "rejecting-test-backend"; }

    int32_t create_calls{0};
};

// ---------------------------------------------------------------------------
// Inline FixedTokenizer for string-based generate() tests
// ---------------------------------------------------------------------------
class VLFixedTokenizer : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string&) const override { return {1, 2, 3}; }
    std::string decode(const std::vector<int32_t>&) const override { return "out"; }
    int32_t id_for_token(std::string_view) const override { return 0; }
    std::string token_for_id(int32_t) const override { return ""; }
};

class GroundingTokenizer : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string&) const override { return {}; }

    std::string decode(const std::vector<int32_t>& ids) const override {
        std::string text;
        for (const int32_t id : ids) {
            if (id == 2)
                text += "red";
            else if (id == 3)
                text += " vehicle";
        }
        return text;
    }

    int32_t id_for_token(std::string_view) const override { return 0; }

    std::string token_for_id(int32_t id) const override {
        switch (id) {
        case 1:
            return "<ref>";
        case 4:
            return "</ref>";
        case 5:
            return "<box>";
        case 6:
            return "<304>";
        case 7:
            return "<267>";
        case 8:
            return "<828>";
        case 9:
            return "<708>";
        case 10:
            return "</box>";
        case 11:
            return "<|im_end|>";
        case 12:
            return "<-1>";
        default:
            return "";
        }
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_declared_vision_engine_fails_closed() {
    RejectingVisionBackend backend;
    trtmc::ModuleCreateOptions options;
    trtmc::BundleFile bundle;
    const std::string declared_config{R"({"has_vision_engine":true})"};

    bool missing_rejected = false;
    try {
        (void)trtmc::load_locateanything_vision_module(&backend, bundle, options, nullptr,
                                                       declared_config);
    } catch (const std::runtime_error& error) {
        missing_rejected = std::string(error.what()).find("Bundle missing vision_engine_plan") !=
                           std::string::npos;
    }
    check(missing_rejected && backend.create_calls == 0,
          "declared vision engine: missing section fails before deserialization");

    bundle.sections.push_back({"vision_engine_plan", {'p', 'l', 'a', 'n'}});
    bool deserialize_rejected = false;
    try {
        (void)trtmc::load_locateanything_vision_module(&backend, bundle, options, nullptr,
                                                       declared_config);
    } catch (const std::runtime_error& error) {
        deserialize_rejected =
            std::string(error.what()).find("Failed to create ITrtModule") != std::string::npos;
    }
    check(deserialize_rejected && backend.create_calls == 1,
          "declared vision engine: deserialization failure is fatal");

    bool present_rejected = false;
    try {
        (void)trtmc::load_locateanything_vision_module(&backend, bundle, options, nullptr, "{}");
    } catch (const std::runtime_error&) {
        present_rejected = true;
    }
    check(present_rejected && backend.create_calls == 2,
          "present vision section is required even when metadata is stale");
}

static void test_grounding_decode_preserves_semantic_special_tokens() {
    GroundingTokenizer tokenizer;
    const std::vector<int32_t> generated{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const std::string decoded = trtmc::locateanything_decode_generated_text(tokenizer, generated);
    check(decoded == "<ref>red vehicle</ref><box><304><267><828><708></box>",
          "grounding decode: preserves ref, box, and coordinate tokens only");
    check(trtmc::locateanything_decode_generated_text(tokenizer, {12}).empty(),
          "grounding decode: rejects negative coordinate syntax");
}

static void test_vl_sequence_prefill_uses_one_text_launch() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decode_stats = std::make_shared<CountingTextStats>();
    auto prefill_stats = std::make_shared<CountingTextStats>();
    auto decoder = std::make_unique<CountingTextModule>(decode_stats, false, stream, 1);
    auto prefill = std::make_unique<CountingTextModule>(prefill_stats, true, stream, 8);
    auto vision = std::make_unique<FakeSequenceVisionModule>();
    auto cache = std::make_unique<trtmc::LocateanythingKvCache>(1, 8, 4, stream);

    trtmc::LocateAnythingConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.image_token_id = 1;
    cfg.vision_output_dim = 4;
    cfg.num_layers = 1;
    cfg.prefill_max_length = 8;

    trtmc::LocateAnythingPreprocessConfig vl_pp;
    vl_pp.preprocessor_type = "simple_chw";
    vl_pp.fixed_image_size = 4;
    vl_pp.in_channels = 3;

    auto tokenizer = std::make_shared<VLFixedTokenizer>();
    trtmc::LocateAnythingPipeline pipeline(std::move(decoder), std::move(vision), std::move(cache),
                                           cfg, vl_pp, stream, tokenizer, "", nullptr,
                                           std::move(prefill));

    float pixels[2 * 2 * 3] = {0.5F, 0.5F, 0.5F, 0.4F, 0.4F, 0.4F,
                               0.3F, 0.3F, 0.3F, 0.2F, 0.2F, 0.2F};
    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;
    gen_cfg.eos_token_id = 2;
    auto result = pipeline.generate("test", pixels, 2, 2, gen_cfg);

    check(result.token_ids == std::vector<int32_t>{2}, "sequence prefill: output remains correct");
    check(prefill_stats->calls == 1, "sequence prefill: one prefill launch");
    check(decode_stats->calls == 0, "sequence prefill: no prompt-linear decode launches");
    check(prefill_stats->shapes["token_id"] == std::vector<int64_t>{3},
          "sequence prefill: token shape");
    check(prefill_stats->shapes["input_embed"] == std::vector<int64_t>({3, 4}),
          "sequence prefill: input embed shape");
    check(prefill_stats->shapes["use_input_embed"] == std::vector<int64_t>({3, 1}),
          "sequence prefill: embed selector shape");
    check(prefill_stats->float_values["use_input_embed"] == std::vector<float>({1.0F, 0.0F, 0.0F}),
          "sequence prefill: image embedding selected by position");
    check(prefill_stats->shapes.count("attention_mask") == 0,
          "sequence prefill: native KV does not materialize an attention mask");

    cudaStreamDestroy(stream);
}

static void test_native_kv_text_prefill_chunks_without_copy() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decode_stats = std::make_shared<CountingTextStats>();
    auto prefill_stats = std::make_shared<CountingTextStats>();
    auto decoder = std::make_unique<CountingTextModule>(decode_stats, false, stream, 1);
    auto prefill = std::make_unique<CountingTextModule>(prefill_stats, true, stream, 2);
    auto cache =
        std::make_unique<trtmc::LocateanythingKvCache>(1, 8, 4, stream, trtmc::DType::kFloat32);

    trtmc::LocateAnythingConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 99;
    cfg.vision_output_dim = 4;
    cfg.num_layers = 1;
    cfg.prefill_max_length = 2;

    trtmc::LocateAnythingPreprocessConfig vl_pp;
    trtmc::LocateAnythingPipeline pipeline(std::move(decoder), nullptr, std::move(cache), cfg,
                                           vl_pp, stream, nullptr, "", nullptr, std::move(prefill));

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;
    const auto result = pipeline.generate_ids({0, 1, 2, 3, 0}, gen_cfg);

    check(result.token_ids.size() == 6, "native prefill: generates one token");
    check(prefill_stats->calls == 3, "native prefill: runs 2/2/1 chunks");
    check(prefill_stats->write_indices == std::vector<int32_t>({0, 2, 4}),
          "native prefill: writes each chunk at the logical cache position");
    check(prefill_stats->kv_lengths == std::vector<int32_t>({2, 4, 5}),
          "native prefill: attends only over the active cache prefix");
    check(prefill_stats->shapes.count("attention_mask") == 0,
          "native prefill: does not materialize a dense attention mask");
    check(decode_stats->calls == 0,
          "native prefill: final sampled token avoids an unnecessary decode launch");

    bool overflow_rejected = false;
    try {
        (void)pipeline.generate_ids({0, 1, 2, 3, 0, 1, 2, 3}, gen_cfg);
    } catch (const std::runtime_error&) {
        overflow_rejected = true;
    }
    check(overflow_rejected && prefill_stats->calls == 3,
          "native prefill: rejects prompt plus generation beyond capacity before enqueue");

    cudaStreamDestroy(stream);
}

int main() {
    test_declared_vision_engine_fails_closed();
    test_grounding_decode_preserves_semantic_special_tokens();
    test_vl_sequence_prefill_uses_one_text_launch();
    test_native_kv_text_prefill_chunks_without_copy();
    if (failures > 0)
        std::cerr << failures << " FAILED\n";
    return failures;
}
