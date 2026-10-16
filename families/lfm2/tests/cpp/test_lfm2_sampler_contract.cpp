/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Compile the owning factory parser directly to exercise its metadata checks
// without creating a CUDA inference state or introducing a production test API.
#include "families/lfm2/runtime/plugin.cpp"

#include <cstdlib>
#include <fstream>
#include <iostream>

namespace {

using Json = nlohmann::json;
int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

Json runtime_config() {
    return {{"hidden_size", 4},
            {"num_hidden_layers", 2},
            {"num_attention_heads", 1},
            {"num_key_value_heads", 1},
            {"head_dim", 4},
            {"num_attention_layers", 1},
            {"num_conv_layers", 1},
            {"conv_L_cache", 3},
            {"conv_dim", 4},
            {"vocab_size", 128000},
            {"bos_token_id", 1},
            {"eos_token_id", 2},
            {"max_cache_length", 8},
            {"precision", "bf16"},
            {"intermediate_size", 8},
            {"norm_eps", 1.0e-5},
            {"rope_theta", 1000000.0},
            {"layer_types", {"conv", "full_attention"}},
            {"tie_word_embeddings", true},
            {"pad_token_id", 0},
            {"native_kv_cache", true},
            {"native_kv_contract_version", 1},
            {"decoder_engine_layout", "single"}};
}

Json w2_runtime_config(const char* mode) {
    auto config = runtime_config();
    config["sampler_mode"] = "philox_exponential_v1";
    config["w2_mode"] = mode;
    config["hidden_size"] = 2048;
    config["conv_dim"] = 2048;
    config["intermediate_size"] = 10752;
    config["num_hidden_layers"] = 30;
    config["num_attention_heads"] = 32;
    config["num_key_value_heads"] = 8;
    config["head_dim"] = 64;
    config["num_attention_layers"] = 8;
    config["num_conv_layers"] = 22;
    std::vector<std::string> layers(22, "conv");
    layers.insert(layers.end(), 8, "full_attention");
    config["layer_types"] = layers;
    return config;
}

class Fixture {
  public:
    Fixture() {
        char pattern[] = "/tmp/trtmc_lfm2_sampler_contract_XXXXXX";
        const char* created = mkdtemp(pattern);
        if (created == nullptr)
            throw std::runtime_error("cannot create contract-test directory");
        directory_ = created;
    }
    ~Fixture() { std::filesystem::remove_all(directory_); }
    std::filesystem::path write(const Json& config, int sampler_bytes) {
        const auto path = directory_ / (std::to_string(count_++) + ".bundle");
        const std::string metadata = config.dump();
        Json sections{{"runtime.json", {{"offset", 0}, {"length", metadata.size()}}},
                      {"engine.plan", {{"offset", metadata.size()}, {"length", 1}}}};
        if (sampler_bytes >= 0)
            sections["sampler.plan"] = {{"offset", metadata.size() + 1}, {"length", sampler_bytes}};
        const auto header = Json{
            {"format", 1},
            {"family", "lfm2"},
            {"task", "text_generation"},
            {"backend", "trt"},
            {"sections",
             sections}}.dump();
        std::ofstream output(path, std::ios::binary);
        constexpr unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
        output.write(reinterpret_cast<const char*>(magic), sizeof(magic));
        const std::uint64_t length = header.size();
        for (int shift = 0; shift < 64; shift += 8)
            output.put(static_cast<char>((length >> shift) & 0xffU));
        output.write(header.data(), header.size());
        output.write(metadata.data(), metadata.size());
        output.put('D');
        for (int i = 0; i < sampler_bytes; ++i)
            output.put('S');
        output.close();
        if (!output)
            throw std::runtime_error("cannot write contract-test bundle");
        return path;
    }
    bool rejected(const Json& config, int sampler_bytes) {
        try {
            trtmc::BundleReader reader(write(config, sampler_bytes).string());
            (void)trtmc::lfm2::parse_runtime_config(reader);
            return false;
        } catch (const std::exception&) {
            return true;
        }
    }

  private:
    std::filesystem::path directory_;
    int count_{0};
};

class FakeBackend final : public trtmc::IBackend {
  public:
    explicit FakeBackend(const char* name) : name_(name) {}
    std::unique_ptr<trtmc::ITrtModule> create_module(const void*, size_t,
                                                     const trtmc::ModuleCreateOptions&) override {
        ++calls;
        throw std::runtime_error("fake decoder load reached");
    }
    std::unique_ptr<trtmc::ITrtModule>
    create_module_prebound(const void*, size_t, const trtmc::ModuleCreateOptions&,
                           const std::vector<trtmc::ModuleExternalBinding>&) override {
        return nullptr;
    }
    trtmc::BackendDualProfileModules
    create_dual_profile_modules(const void*, size_t, const trtmc::ModuleCreateOptions&) override {
        return {};
    }
    const char* name() const override { return name_; }
    int calls{0};

  private:
    const char* name_;
};

void accepted_contracts(Fixture& fixture) {
    auto legacy = runtime_config();
    legacy["vocab_size"] = 65536;
    legacy["precision"] = "fp16";
    trtmc::BundleReader reader(fixture.write(legacy, -1).string());
    const auto parsed = trtmc::lfm2::parse_runtime_config(reader);
    check(parsed.sampler_mode.empty() && parsed.w2_mode.empty() && parsed.vocab_size == 65536,
          "legacy 23-field bundle retains its vocabulary and sampler path");
    auto current = runtime_config();
    current["sampler_mode"] = "philox_exponential_v1";
    trtmc::BundleReader current_reader(fixture.write(current, 1).string());
    check(trtmc::lfm2::parse_runtime_config(current_reader).sampler_mode == "philox_exponential_v1",
          "new 24-field contract accepts its required sampler plan");
    check(trtmc::lfm2::parse_runtime_config(current_reader).w2_mode.empty(),
          "sampler-only 24-field contract requires no W2 helper");
    for (const char* mode : {"cublas_bf16_all_w2_v1", "cublas_bf16_first_two_v1"}) {
        trtmc::BundleReader w2_reader(fixture.write(w2_runtime_config(mode), 1).string());
        const auto w2 = trtmc::lfm2::parse_runtime_config(w2_reader);
        check(w2.w2_mode == mode && w2.sampler_mode == "philox_exponential_v1",
              "25-field contract preserves each W2 mode and its sampler dependency");
        FakeBackend other_backend("trt_rtx");
        bool rejected = false;
        try {
            (void)trtmc::lfm2::create({w2_reader, other_backend});
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("TensorRT backend") != std::string::npos;
        }
        check(rejected && other_backend.calls == 0,
              "each W2 mode rejects an unqualified backend before helper/model loading");
    }
    FakeBackend legacy_backend("trt");
    try {
        (void)trtmc::lfm2::create({reader, legacy_backend});
    } catch (const std::runtime_error& error) {
        check(std::string(error.what()) == "fake decoder load reached",
              "legacy factory needs no sampler helper discovery");
    }
    check(legacy_backend.calls == 1, "legacy factory reaches decoder without BYOK dependencies");
    FakeBackend other_backend("trt_rtx");
    bool rejected = false;
    try {
        (void)trtmc::lfm2::create({current_reader, other_backend});
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("TensorRT backend") != std::string::npos;
    }
    check(rejected && other_backend.calls == 0,
          "new mode rejects an unqualified backend before helper/model loading");
}

void rejected_contracts(Fixture& fixture) {
    const auto legacy = runtime_config();
    auto current = legacy;
    current["sampler_mode"] = "philox_exponential_v1";
    check(fixture.rejected(legacy, 1), "legacy metadata rejects a stray sampler plan");
    check(fixture.rejected(legacy, 0), "legacy metadata rejects even an empty stray sampler plan");
    check(fixture.rejected(current, -1), "new mode requires a sampler section");
    check(fixture.rejected(current, 0), "new mode requires nonempty sampler bytes");
    auto changed = current;
    changed["sampler_mode"] = "unknown";
    check(fixture.rejected(changed, 1), "unknown sampler mode cannot fall back to legacy");
    changed["sampler_mode"] = 1;
    check(fixture.rejected(changed, 1), "sampler mode must be a string");
    changed = current;
    changed["vocab_size"] = 65536;
    check(fixture.rejected(changed, 1), "new sampler requires its declared fixed vocabulary");
    changed = current;
    changed["unexpected"] = true;
    check(fixture.rejected(changed, 1), "new contract rejects extra metadata");
    changed = legacy;
    changed["unexpected"] = true;
    check(fixture.rejected(changed, -1), "legacy contract remains exactly 23 fields");
    changed = current;
    changed.erase("norm_eps");
    changed["unexpected"] = true;
    check(fixture.rejected(changed, 1), "field count alone cannot replace required metadata");
}

void rejected_w2_contracts(Fixture& fixture, const char* mode) {
    const auto current = w2_runtime_config(mode);
    auto changed = current;
    changed.erase("sampler_mode");
    check(fixture.rejected(changed, -1), "W2 mode cannot replace the sampler contract");
    check(fixture.rejected(current, -1), "W2 contract still requires sampler.plan");
    check(fixture.rejected(current, 0), "W2 contract rejects an empty sampler.plan");
    changed = current;
    changed["w2_mode"] = "unknown";
    check(fixture.rejected(changed, 1), "unknown W2 mode cannot silently fall back");
    changed["w2_mode"] = 1;
    check(fixture.rejected(changed, 1), "W2 mode must be a string");
    changed = current;
    changed["unexpected"] = true;
    check(fixture.rejected(changed, 1), "W2 contract remains exactly 25 fields");

    const auto requires_w2_geometry = [&](Json config, const char* message) {
        check(fixture.rejected(config, 1), message);
        config.erase("w2_mode");
        check(!fixture.rejected(config, 1), "sampler-only geometry remains accepted");
    };
    changed = current;
    changed["precision"] = "fp16";
    requires_w2_geometry(changed, "W2 mode requires BF16 arithmetic");
    changed = current;
    changed["hidden_size"] = 1024;
    changed["conv_dim"] = 1024;
    changed["head_dim"] = 32;
    requires_w2_geometry(changed, "W2 mode requires the declared output width");
    changed = current;
    changed["intermediate_size"] = 8192;
    requires_w2_geometry(changed, "W2 mode requires the declared input width");
    changed = current;
    changed["num_hidden_layers"] = 29;
    changed["num_conv_layers"] = 21;
    changed["layer_types"].erase(changed["layer_types"].begin());
    requires_w2_geometry(changed, "W2 mode requires the qualified layer count");
    for (int index : {0, 1}) {
        changed = current;
        changed["layer_types"][index] = "full_attention";
        changed["num_attention_layers"] = 9;
        changed["num_conv_layers"] = 21;
        requires_w2_geometry(changed, "W2 mode requires two leading convolution layers");
    }
}

} // namespace

int main() {
    Fixture fixture;
    accepted_contracts(fixture);
    rejected_contracts(fixture);
    rejected_w2_contracts(fixture, "cublas_bf16_all_w2_v1");
    rejected_w2_contracts(fixture, "cublas_bf16_first_two_v1");
    return failures;
}
