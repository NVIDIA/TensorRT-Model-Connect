/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/openpi/config.h"
#include "runtime/models/openpi/paligemma_bpe.h"
#include "runtime/models/openpi/pipeline.h"
#include "runtime/models/openpi/plugin_helpers.h"
#include "utils/sha256.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void check_close(float actual, float expected, float tolerance, const char* name) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++g_failures;
    }
}

template <typename Function>
void check_throws(Function&& function, const char* name) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, name);
}

std::string sha256(std::string_view bytes) {
    trtmc::internal::Sha256 hash;
    hash.update(bytes);
    return hash.hex_digest();
}

std::string droid_config_json(std::string tokenizer_sha256 = std::string(64, '3'),
                              std::string normalization_sha256 = std::string(64, '4'),
                              std::string prefill_sha256 = std::string(64, '5'),
                              std::string action_sha256 = std::string(64, '6')) {
    std::string result = R"json({
      "runtime_strategy": "openpi_vla",
      "task_strategy": "robot_action_generation",
      "user_contract": "robot_action_chunk",
      "model_type": "openpi_pi05_flow",
      "precision": "bf16",
      "openpi_profile": "pi05_droid",
      "openpi_upstream_commit": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
      "openpi_action_horizon": 15,
      "openpi_internal_action_dim": 32,
      "openpi_external_action_dim": 8,
      "openpi_external_state_dim": 8,
      "openpi_prefix_length": 968,
      "openpi_max_token_length": 200,
      "openpi_num_layers": 18,
      "openpi_num_heads": 8,
      "openpi_num_kv_heads": 1,
      "openpi_head_dim": 256,
      "openpi_denoise_steps": 10,
      "openpi_discrete_state_input": true,
      "openpi_camera_names": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
      "openpi_camera_mask": [true, true, false],
      "openpi_batch_size": 1,
      "openpi_runtime_contract": "native_cpp_device_resident_flow",
      "openpi_parameter_dtype": "bfloat16",
      "openpi_tokenizer_sha256": ")json";
    result += tokenizer_sha256;
    result += R"json(",
      "openpi_normalization_sha256": ")json";
    result += normalization_sha256;
    result += R"json(",
      "openpi_prefill_engine_sha256": ")json";
    result += prefill_sha256;
    result += R"json(",
      "openpi_action_engine_sha256": ")json";
    result += action_sha256;
    result += R"json("
    })json";
    return result;
}

std::string normalization_json() {
    return R"json({
      "norm_stats": {
        "state": {
          "q01": [-1,-1,-1,-1,-1,-1,-1,-1],
          "q99": [1,1,1,1,1,1,1,1]
        },
        "actions": {
          "q01": [-1,-1,-1,-1,-1,-1,-1,-1],
          "q99": [1,1,1,1,1,1,1,1]
        }
      }
    })json";
}

std::string padded_normalization_json() {
    return R"json({
      "norm_stats": {
        "state": {
          "q01": [-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          "q99": [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        },
        "actions": {
          "q01": [-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          "q99": [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        }
      }
    })json";
}

trtmc::openpi::PaligemmaBpeAsset make_tokenizer_asset() {
    using trtmc::openpi::BpePiece;
    using trtmc::openpi::SentencePieceType;
    trtmc::openpi::PaligemmaBpeAsset asset;
    asset.byte_fallback = false;
    asset.add_dummy_prefix = false;
    asset.remove_extra_whitespaces = false;
    asset.unknown_id = 1;
    asset.bos_id = 2;
    asset.eos_id = 3;
    asset.pad_id = 0;
    asset.pieces = {
        BpePiece{"<pad>", 0.0F, SentencePieceType::kControl},
        BpePiece{"<unk>", 0.0F, SentencePieceType::kUnknown},
        BpePiece{"<s>", 0.0F, SentencePieceType::kControl},
        BpePiece{"</s>", 0.0F, SentencePieceType::kControl},
    };
    return asset;
}

class FakeModule final : public trtmc::ITrtModule {
  public:
    void add_input(std::string name, std::vector<int64_t> shape, trtmc::DType dtype) {
        tensors_[name] = trtmc::TensorInfo{name, std::move(shape), dtype, true};
    }

    void add_output(std::string name, std::vector<int64_t> shape, trtmc::DType dtype) {
        tensors_[name] = trtmc::TensorInfo{name, std::move(shape), dtype, false};
    }

    trtmc::TensorMap forward(const trtmc::TensorMap&) override {
        throw std::runtime_error("FakeModule::forward must not run");
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override {
        throw std::runtime_error("FakeModule::forward_device must not run");
    }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {
        throw std::runtime_error("FakeModule::forward_device_async must not run");
    }
    void forward_async(const trtmc::TensorMap&) override {
        throw std::runtime_error("FakeModule::forward_async must not run");
    }
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return tensor_info(true); }
    std::vector<trtmc::TensorInfo> output_info() const override { return tensor_info(false); }
    bool has_input(const std::string& name) const override { return has(name, true); }
    bool has_output(const std::string& name) const override { return has(name, false); }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        return tensors_.at(name).dtype;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        return tensors_.at(name).shape;
    }
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
    bool has(const std::string& name, bool input) const {
        const auto iterator = tensors_.find(name);
        return iterator != tensors_.end() && iterator->second.is_input == input;
    }

    std::vector<trtmc::TensorInfo> tensor_info(bool input) const {
        std::vector<trtmc::TensorInfo> result;
        for (const auto& [unused, info] : tensors_) {
            (void)unused;
            if (info.is_input == input) {
                result.push_back(info);
            }
        }
        return result;
    }

    std::unordered_map<std::string, trtmc::TensorInfo> tensors_;
};

class FakeBackend final : public trtmc::IBackend {
  public:
    explicit FakeBackend(const char* backend_name) : backend_name_(backend_name) {}

    std::unique_ptr<trtmc::ITrtModule> create_module(const void*, size_t,
                                                     const trtmc::ModuleCreateOptions&) override {
        ++create_module_calls;
        return std::make_unique<FakeModule>();
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

    const char* name() const override { return backend_name_; }

    int create_module_calls{0};

  private:
    const char* backend_name_;
};

std::pair<std::unique_ptr<FakeModule>, std::unique_ptr<FakeModule>>
make_fake_modules(const trtmc::openpi::OpenPIConfig& config) {
    auto prefill = std::make_unique<FakeModule>();
    prefill->add_input("pixel_values", {3, 3, 224, 224}, trtmc::DType::kFloat32);
    prefill->add_input("token_ids", {1, 200}, trtmc::DType::kInt32);
    prefill->add_input("prefix_mask", {1, 968}, trtmc::DType::kFloat32);
    prefill->add_input("prefix_position_ids", {1, 968}, trtmc::DType::kInt32);
    prefill->add_output("vision_tokens", {1, 3, 256, 2048}, trtmc::DType::kBFloat16);

    auto action = std::make_unique<FakeModule>();
    const std::vector<int64_t> action_shape{1, config.action_horizon, 32};
    action->add_input("noisy_actions", action_shape, trtmc::DType::kFloat32);
    action->add_input("timestep", {1}, trtmc::DType::kFloat32);
    action->add_input("step_size", {1}, trtmc::DType::kFloat32);
    action->add_input("prefix_mask", {1, 968}, trtmc::DType::kFloat32);
    action->add_input("suffix_position_ids", {1, config.action_horizon}, trtmc::DType::kInt32);
    action->add_output("velocity", action_shape, trtmc::DType::kFloat32);
    action->add_output("next_actions", action_shape, trtmc::DType::kFloat32);

    const std::vector<int64_t> cache_shape{1, 968, 1, 256};
    for (int32_t layer = 0; layer < config.num_layers; ++layer) {
        for (char kind : {'k', 'v'}) {
            const std::string name = std::string("prefix_") + kind + "_" + std::to_string(layer);
            prefill->add_output(name, cache_shape, trtmc::DType::kBFloat16);
            action->add_input(name, cache_shape, trtmc::DType::kBFloat16);
        }
    }
    return {std::move(prefill), std::move(action)};
}

void test_strict_config_and_normalization() {
    const auto config = trtmc::openpi::parse_openpi_config(droid_config_json());
    check(config.profile == "pi05_droid" && config.action_horizon == 15 &&
              config.tokenizer_sha256 == std::string(64, '3') &&
              config.normalization_sha256 == std::string(64, '4') &&
              config.prefill_engine_sha256 == std::string(64, '5') &&
              config.action_engine_sha256 == std::string(64, '6'),
          "OpenPI config selects explicit DROID profile");
    const auto normalization =
        trtmc::openpi::parse_openpi_normalization(normalization_json(), config);
    check(normalization.state.q01.size() == 8 && normalization.actions.q99.size() == 8,
          "OpenPI normalization parses profile quantiles");
    const auto padded_normalization =
        trtmc::openpi::parse_openpi_normalization(padded_normalization_json(), config);
    check(padded_normalization.state.q01.size() == 32 &&
              padded_normalization.actions.q99.size() == 32 &&
              padded_normalization.actions.q01[31] == padded_normalization.actions.q99[31],
          "OpenPI normalization accepts official zero-span padded dimensions");

    auto mismatched = droid_config_json();
    const auto position = mismatched.find("\"openpi_action_horizon\": 15");
    mismatched.replace(position, std::string("\"openpi_action_horizon\": 15").size(),
                       "\"openpi_action_horizon\": 10");
    check_throws([&] { (void)trtmc::openpi::parse_openpi_config(mismatched); },
                 "OpenPI config rejects profile/shape mismatch");
    auto duplicate_key_config = droid_config_json();
    const auto profile_position = duplicate_key_config.find("\"openpi_profile\"");
    duplicate_key_config.insert(profile_position, "\"openpi_profile\": \"pi05_droid\",\n      ");
    check_throws([&] { (void)trtmc::openpi::parse_openpi_config(duplicate_key_config); },
                 "OpenPI config rejects duplicate JSON keys");

    check_throws(
        [&] { (void)trtmc::openpi::parse_openpi_config(droid_config_json(std::string(64, 'A'))); },
        "OpenPI config rejects an uppercase payload digest");
    check_throws(
        [&] {
            (void)trtmc::openpi::parse_openpi_config(
                droid_config_json(std::string(64, '3'), std::string(63, '4')));
        },
        "OpenPI config rejects a truncated payload digest");

    auto bfloat16 = droid_config_json();
    const auto bf16_position = bfloat16.find("\"precision\": \"bf16\"");
    bfloat16.replace(bf16_position, std::string("\"precision\": \"bf16\"").size(),
                     "\"precision\": \"bfloat16\"");
    check(trtmc::openpi::parse_openpi_config(bfloat16).precision == "bfloat16",
          "OpenPI config accepts the bfloat16 precision alias");

    for (const std::string_view rejected_precision : {"fp16", "fp32"}) {
        auto rejected = droid_config_json();
        const auto precision_position = rejected.find("\"precision\": \"bf16\"");
        rejected.replace(precision_position, std::string("\"precision\": \"bf16\"").size(),
                         "\"precision\": \"" + std::string(rejected_precision) + "\"");
        const auto parameter_dtype_position =
            rejected.find("\"openpi_parameter_dtype\": \"bfloat16\"");
        rejected.replace(parameter_dtype_position,
                         std::string("\"openpi_parameter_dtype\": \"bfloat16\"").size(),
                         "\"openpi_parameter_dtype\": \"" + std::string(rejected_precision) + "\"");
        check_throws([&] { (void)trtmc::openpi::parse_openpi_config(rejected); },
                     "OpenPI config rejects an unqualified network precision");
    }

    auto mismatched_parameter_dtype = droid_config_json();
    const auto dtype_position =
        mismatched_parameter_dtype.find("\"openpi_parameter_dtype\": \"bfloat16\"");
    mismatched_parameter_dtype.replace(
        dtype_position, std::string("\"openpi_parameter_dtype\": \"bfloat16\"").size(),
        "\"openpi_parameter_dtype\": \"bf16\"");
    check_throws([&] { (void)trtmc::openpi::parse_openpi_config(mismatched_parameter_dtype); },
                 "OpenPI config rejects a non-bfloat16 parameter dtype");
}

void test_bundle_integrity_rejects_missing_duplicate_and_tampered_sections() {
    const auto tokenizer_vector =
        trtmc::openpi::serialize_paligemma_bpe_asset(make_tokenizer_asset());
    const std::string tokenizer(reinterpret_cast<const char*>(tokenizer_vector.data()),
                                tokenizer_vector.size());
    const std::string prefill_plan = "prefill-plan";
    const std::string action_plan = "action-plan";
    const std::string normalization = normalization_json();
    const std::array<std::string, 5> storage = {
        droid_config_json(sha256(tokenizer), sha256(normalization), sha256(prefill_plan),
                          sha256(action_plan)),
        prefill_plan,
        action_plan,
        tokenizer,
        normalization,
    };
    trtmc::BundleFile bundle;
    for (std::size_t index = 0; index < storage.size(); ++index) {
        bundle.sections.push_back(
            {std::string(trtmc::openpi::kRequiredIntegritySectionNames[index]),
             std::vector<char>(storage[index].begin(), storage[index].end())});
    }

    const auto verified = trtmc::openpi::verify_openpi_bundle_integrity(bundle);
    check(verified.prefill_plan != nullptr && verified.action_plan != nullptr &&
              verified.tokenizer_bytes == tokenizer &&
              verified.normalization_bytes == normalization,
          "OpenPI bundle integrity verifies every required section before engine loading");

    auto duplicate = bundle;
    duplicate.sections.push_back(bundle.sections[1]);
    check_throws([&] { (void)trtmc::openpi::verify_openpi_bundle_integrity(duplicate); },
                 "OpenPI bundle integrity rejects duplicate physical sections");

    auto unexpected = bundle;
    unexpected.sections.push_back({"openpi_provenance.json", {'x'}});
    check_throws([&] { (void)trtmc::openpi::verify_openpi_bundle_integrity(unexpected); },
                 "OpenPI bundle integrity rejects the removed provenance section");

    auto missing = bundle;
    missing.sections.erase(missing.sections.begin() + 2);
    check_throws([&] { (void)trtmc::openpi::verify_openpi_bundle_integrity(missing); },
                 "OpenPI bundle integrity rejects missing action plan");

    auto empty = bundle;
    empty.sections[4].data.clear();
    check_throws([&] { (void)trtmc::openpi::verify_openpi_bundle_integrity(empty); },
                 "OpenPI bundle integrity rejects an empty required section");

    for (std::size_t index = 1; index < storage.size(); ++index) {
        auto tampered = bundle;
        tampered.sections[index].data.push_back('x');
        check_throws([&] { (void)trtmc::openpi::verify_openpi_bundle_integrity(tampered); },
                     "OpenPI bundle integrity rejects tampered payload bytes");
    }
}

void test_module_loader_rejects_non_trt_backends_before_deserialization() {
    const std::vector<char> plan{'p', 'l', 'a', 'n'};
    const trtmc::ModuleCreateOptions options{};

    FakeBackend null_name(nullptr);
    check_throws(
        [&] {
            (void)trtmc::openpi::load_openpi_module(&null_name, &plan, "engine_plan", "openpi-test",
                                                    options);
        },
        "OpenPI module loader rejects a null backend name");
    check(null_name.create_module_calls == 0,
          "OpenPI null-name rejection happens before deserialization");

    FakeBackend wrong_name("trt_rtx");
    check_throws(
        [&] {
            (void)trtmc::openpi::load_openpi_module(&wrong_name, &plan, "engine_plan",
                                                    "openpi-test", options);
        },
        "OpenPI module loader rejects a non-trt backend");
    check(wrong_name.create_module_calls == 0,
          "OpenPI wrong-backend rejection happens before deserialization");

    FakeBackend trt_backend("trt");
    auto module = trtmc::openpi::load_openpi_module(&trt_backend, &plan, "engine_plan",
                                                    "openpi-test", options);
    check(module != nullptr && trt_backend.create_module_calls == 1,
          "OpenPI module loader accepts exactly the standard trt backend");
}

trtmc::openpi::ActionRequest make_request(const trtmc::openpi::OpenPIConfig& config) {
    constexpr std::size_t image_elements = 224U * 224U * 3U;
    trtmc::openpi::RobotImage base{
        "base_0_rgb", std::vector<float>(image_elements, 0.5F), 224, 224, 3, true};
    trtmc::openpi::RobotImage left{
        "left_wrist_0_rgb", std::vector<float>(image_elements, 1.0F), 224, 224, 3, true};
    trtmc::openpi::RobotImage right{
        "right_wrist_0_rgb", std::vector<float>(image_elements, 0.0F), 224, 224, 3, false};
    trtmc::openpi::ActionRequest request;
    request.prompt = "pick_up block";
    request.cameras = {std::move(right), std::move(base), std::move(left)};
    request.state.assign(static_cast<std::size_t>(config.external_state_dim), 0.0F);
    request.initial_noise.assign(static_cast<std::size_t>(config.action_horizon) * 32U, 0.25F);
    return request;
}

void test_cpu_request_preparation() {
    const auto config = trtmc::openpi::parse_openpi_config(droid_config_json());
    const auto normalization =
        trtmc::openpi::parse_openpi_normalization(normalization_json(), config);
    trtmc::openpi::PaligemmaBpeTokenizer tokenizer(make_tokenizer_asset());
    const auto prepared = trtmc::openpi::prepare_openpi_inputs(config, normalization, tokenizer,
                                                               make_request(config), true);
    check(prepared.pixel_values.size() == 3U * 3U * 224U * 224U,
          "OpenPI request preparation emits fixed NCHW pixels");
    check(prepared.preprocessed_images.size() == 3U * 224U * 224U * 3U &&
              prepared.preprocessed_images[0] == prepared.pixel_values[0],
          "OpenPI request preparation retains qualification NHWC pixels");
    check_close(prepared.pixel_values[0], -0.0039215684F, 1e-7F,
                "OpenPI float public pixels follow upstream uint8 conversion");
    check_close(prepared.pixel_values[3U * 224U * 224U], 0x1.fffffcp-1F, 0.0F,
                "OpenPI canonical camera order places left wrist second");
    check_close(prepared.pixel_values[6U * 224U * 224U], -1.0F, 0.0F,
                "OpenPI masked right wrist is normalized black");
    check(prepared.token_ids.size() == 200U && prepared.prefix_mask.size() == 968U,
          "OpenPI request preparation emits fixed token/prefix shapes");
    check(prepared.token_mask.size() == 200U &&
              prepared.image_mask == std::vector<uint8_t>({1U, 1U, 0U}) &&
              prepared.normalized_state.size() == 32U,
          "OpenPI request preparation retains exact masks and padded normalized state");
    check(prepared.prefix_mask[0] == 1U && prepared.prefix_mask[511] == 1U &&
              prepared.prefix_mask[512] == 0U && prepared.prefix_mask[767] == 0U,
          "OpenPI request preparation expands camera validity over image tokens");
    check(prepared.prefix_positions[511] == 511 && prepared.prefix_positions[767] == 511,
          "OpenPI prefix positions use cumsum(valid)-1");
    check(prepared.suffix_positions.front() >= 512 && prepared.schedule.timesteps.size() == 10U,
          "OpenPI suffix positions and ten-step schedule are prepared");
    check(prepared.initial_noise.front() == 0.25F && prepared.initial_noise.back() == 0.25F,
          "OpenPI caller-provided parity noise is preserved");

    const auto production_prepared = trtmc::openpi::prepare_openpi_inputs(
        config, normalization, tokenizer, make_request(config));
    check(production_prepared.preprocessed_images.empty() &&
              production_prepared.token_mask.empty() && production_prepared.image_mask.empty() &&
              production_prepared.normalized_state.empty(),
          "OpenPI production preparation does not retain diagnostic-only host tensors");

    auto invalid = make_request(config);
    invalid.denoise_steps = 9;
    check_throws(
        [&] {
            (void)trtmc::openpi::prepare_openpi_inputs(config, normalization, tokenizer, invalid);
        },
        "OpenPI request rejects a non-qualified denoise step override");
}

void test_fake_module_contract_and_pipeline_construction() {
    const auto config = trtmc::openpi::parse_openpi_config(droid_config_json());
    auto modules = make_fake_modules(config);
    trtmc::openpi::validate_openpi_engine_contracts(*modules.first, *modules.second, config);
    const auto normalization =
        trtmc::openpi::parse_openpi_normalization(normalization_json(), config);
    trtmc::openpi::OpenPIPipeline pipeline(
        std::move(modules.first), std::move(modules.second), config, normalization,
        trtmc::openpi::PaligemmaBpeTokenizer(make_tokenizer_asset()), "openpi-test");
    check(std::string(pipeline.pipeline_type()) == "OpenPIPipeline" &&
              std::string(pipeline.model_id()) == "openpi-test",
          "OpenPI pipeline constructs with CPU fake modules after contract validation");

    auto bad_modules = make_fake_modules(config);
    bad_modules.second->add_output("next_actions", {1, 14, 32}, trtmc::DType::kFloat32);
    check_throws(
        [&] {
            trtmc::openpi::validate_openpi_engine_contracts(*bad_modules.first, *bad_modules.second,
                                                            config);
        },
        "OpenPI engine validation rejects action shape mismatch");

    auto stale_prefill_modules = make_fake_modules(config);
    stale_prefill_modules.first->add_output("vision_tokens", {1, 768, 2048},
                                            trtmc::DType::kBFloat16);
    check_throws(
        [&] {
            trtmc::openpi::validate_openpi_engine_contracts(*stale_prefill_modules.first,
                                                            *stale_prefill_modules.second, config);
        },
        "OpenPI engine validation requires rebuilt vision diagnostic output");

    auto request_without_noise = make_request(config);
    request_without_noise.initial_noise.clear();
    check_throws([&] { (void)pipeline.predict_actions_with_diagnostics(request_without_noise); },
                 "OpenPI diagnostics fail closed without external replay noise");
}

} // namespace

int main() {
    test_strict_config_and_normalization();
    test_bundle_integrity_rejects_missing_duplicate_and_tampered_sections();
    test_module_loader_rejects_non_trt_backends_before_deserialization();
    test_cpu_request_preparation();
    test_fake_module_contract_and_pipeline_construction();
    if (g_failures != 0) {
        std::cerr << g_failures << " OpenPI pipeline test(s) failed\n";
        return 1;
    }
    return 0;
}
