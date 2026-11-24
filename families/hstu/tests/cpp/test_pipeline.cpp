/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/pipeline.h"

#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace trtmc;
using namespace trtmc::hstu;
using Json = nlohmann::json;

class Engine final : public ITrtModule {
  public:
    int calls{0};
    bool bad_output{false};
    std::int64_t observed{0};
    float result{0.25F};
    TensorMap forward(const TensorMap& inputs) override {
        ++calls;
        std::memcpy(&observed, inputs.at("ids").data, sizeof(observed));
        return {{"scores", Tensor{&result, {1, bad_output ? 2 : 1}, DType::kFloat32}}};
    }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override {
        throw std::logic_error("unused");
    }
    void forward_device_async(const DeviceTensorMap&) override { throw std::logic_error("unused"); }
    void forward_async(const TensorMap&) override { throw std::logic_error("unused"); }
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<TensorInfo> input_info() const override {
        return {{"ids", {-1, 1}, DType::kInt64, true}, {"bounded", {-1, 1}, DType::kFloat32, true}};
    }
    std::vector<TensorInfo> output_info() const override {
        return {{"scores", {-1, 1}, DType::kFloat32, false}};
    }
    bool has_input(const std::string& name) const override {
        return name == "ids" || name == "bounded";
    }
    bool has_output(const std::string& name) const override { return name == "scores"; }
    DType tensor_dtype(const std::string& name) const override {
        return name == "ids"       ? DType::kInt64
               : name == "bounded" ? DType::kFloat32
                                   : DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {-1, 1}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             ProfileShapeSelector) const override {
        return {1, 1};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 2; }
    bool input_is_dynamic(const std::string&) const override { return true; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}
};

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
void reject(const std::function<void()>& run) {
    try {
        run();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("invalid request reached engine");
}
template <class T>
FeatureTensor feature(const char* name, DType dtype, T value) {
    FeatureTensor out{name, dtype, {1, 1}, {}};
    out.bytes.resize(sizeof(T));
    std::memcpy(out.bytes.data(), &value, sizeof(T));
    return out;
}
int main() {
    try {
        Json bounded = {{"name", "bounded"},
                        {"dtype", "float32"},
                        {"shape", {-1, 1}},
                        {"value_min", 1},
                        {"value_max", 50}};
        Json config = {
            {"family", "hstu"},
            {"max_batch_size", 2},
            {"inputs",
             Json::array({{{"name", "ids"}, {"dtype", "int64"}, {"shape", {-1, 1}}}, bounded})},
            {"outputs",
             Json::array({{{"name", "scores"}, {"dtype", "float32"}, {"shape", {-1, 1}}}})}};
        auto owned = std::make_unique<Engine>();
        auto* engine = owned.get();
        Pipeline pipeline(std::move(owned), config);
        std::vector<FeatureTensor> valid = {feature("ids", DType::kInt64, -(std::int64_t{1} << 40)),
                                            feature("bounded", DType::kFloat32, 50.0F)};
        auto output = pipeline.score(valid);
        check(engine->calls == 1 && engine->observed == -(std::int64_t{1} << 40),
              "signed int64 truncated");
        check(output.size() == 1 && output[0].bytes.size() == sizeof(float),
              "valid output missing");
        auto malformed = valid;
        malformed.pop_back();
        reject([&] { pipeline.score(malformed); });
        malformed = valid;
        malformed[1].name = "ids";
        reject([&] { pipeline.score(malformed); });
        malformed = valid;
        malformed[0].shape = {1, 2};
        reject([&] { pipeline.score(malformed); });
        malformed = valid;
        malformed[0].shape = {3, 1};
        reject([&] { pipeline.score(malformed); });
        malformed = valid;
        malformed[1].shape = {2, 1};
        reject([&] { pipeline.score(malformed); });
        malformed = valid;
        malformed[0].dtype = DType::kInt32;
        reject([&] { pipeline.score(malformed); });
        malformed = valid;
        malformed[0].bytes.pop_back();
        reject([&] { pipeline.score(malformed); });
        malformed = valid;
        malformed[1] = feature("bounded", DType::kFloat32, std::numeric_limits<float>::quiet_NaN());
        reject([&] { pipeline.score(malformed); });
        check(engine->calls == 1, "invalid input must fail before GPU execution");
        reject(
            [&] { checked_bytes({std::numeric_limits<std::int64_t>::max(), 8}, DType::kInt64); });
        engine->bad_output = true;
        reject([&] { pipeline.score(valid); });
        std::cout << "family structured input contract: passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
