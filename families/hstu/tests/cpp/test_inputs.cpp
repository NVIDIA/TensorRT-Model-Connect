/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/validation.h"

#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace trtmc;
using namespace trtmc::hstu;
using Json = nlohmann::json;

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
        std::vector<FeatureTensor> valid = {feature("ids", DType::kInt64, -(std::int64_t{1} << 40)),
                                            feature("bounded", DType::kFloat32, 50.0F)};
        auto bound = validate_inputs(valid, config);
        std::int64_t observed;
        std::memcpy(&observed, bound.at("ids").data, sizeof(observed));
        check(observed == -(std::int64_t{1} << 40), "signed int64 truncated");
        auto malformed = valid;
        malformed.pop_back();
        reject([&] { validate_inputs(malformed, config); });
        malformed = valid;
        malformed[1].name = "ids";
        reject([&] { validate_inputs(malformed, config); });
        malformed = valid;
        malformed[0].shape = {1, 2};
        reject([&] { validate_inputs(malformed, config); });
        malformed = valid;
        malformed[0].shape = {3, 1};
        reject([&] { validate_inputs(malformed, config); });
        malformed = valid;
        malformed[1].shape = {2, 1};
        reject([&] { validate_inputs(malformed, config); });
        malformed = valid;
        malformed[0].dtype = DType::kInt32;
        reject([&] { validate_inputs(malformed, config); });
        malformed = valid;
        malformed[0].bytes.pop_back();
        reject([&] { validate_inputs(malformed, config); });
        malformed = valid;
        malformed[1] = feature("bounded", DType::kFloat32, std::numeric_limits<float>::quiet_NaN());
        reject([&] { validate_inputs(malformed, config); });
        for (float value : {0.0F, 51.0F, std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity()}) {
            malformed[1] = feature("bounded", DType::kFloat32, value);
            reject([&] { validate_inputs(malformed, config); });
        }
        auto unbounded = config;
        unbounded["inputs"][1].erase("value_min");
        unbounded["inputs"][1].erase("value_max");
        malformed[1] = feature("bounded", DType::kFloat32, -1.0e30F);
        check(validate_inputs(malformed, unbounded).size() == 2, "finite unbounded input rejected");
        malformed[1] = feature("bounded", DType::kFloat32, std::numeric_limits<float>::infinity());
        reject([&] { validate_inputs(malformed, unbounded); });
        auto integer_config = config;
        integer_config["inputs"][0]["integer_min"] = 1;
        integer_config["inputs"][0]["integer_max"] = 2;
        auto integer_inputs = valid;
        for (std::int64_t value : {0, 3}) {
            integer_inputs[0] = feature("ids", DType::kInt64, value);
            reject([&] { validate_inputs(integer_inputs, integer_config); });
        }
        integer_inputs[0] = feature("ids", DType::kInt64, std::int64_t{2});
        check(validate_inputs(integer_inputs, integer_config).size() == 2,
              "integer maximum must be inclusive");
        auto fixed_config = config;
        fixed_config["min_batch_size"] = 2;
        reject([&] { validate_inputs(valid, fixed_config); });
        integer_config["inputs"][0]["dtype"] = "int32";
        integer_inputs[0] = feature("ids", DType::kInt32, std::int32_t{2});
        check(validate_inputs(integer_inputs, integer_config).size() == 2, "int32 bounds rejected");
        integer_inputs[0] = feature("ids", DType::kInt32, std::int32_t{-1});
        reject([&] { validate_inputs(integer_inputs, integer_config); });
        reject(
            [&] { checked_bytes({std::numeric_limits<std::int64_t>::max(), 8}, DType::kInt64); });
        std::cout << "family structured input contract: passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
