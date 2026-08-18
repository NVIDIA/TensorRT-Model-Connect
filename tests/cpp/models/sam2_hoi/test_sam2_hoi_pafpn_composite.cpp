/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU-only structural coverage for the SAM2 HOI Phase-A PAFPN composite.

#include "runtime/models/sam2_hoi/pafpn_composite.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using trtmc::DeviceTensorMap;
using trtmc::DType;
using trtmc::ITrtModule;
using trtmc::ProfileShapeSelector;
using trtmc::TensorInfo;
using trtmc::TensorMap;

cudaStream_t fake_stream() {
    return reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(1));
}

struct Contract {
    std::vector<int64_t> shape{1};
    DType dtype{DType::kFloat32};
};

struct FakeModuleOptions {
    cudaStream_t stream{fake_stream()};
    std::vector<int64_t> shape{1};
    DType dtype{DType::kFloat32};
    int32_t profile_count{1};
    int32_t profile_index{0};
    bool dynamic_input{false};
    bool valid{true};
    bool null_outputs{false};
};

class FakeModule final : public ITrtModule {
  public:
    FakeModule(std::size_t ordinal, std::vector<std::string> inputs,
               std::vector<std::string> outputs, std::vector<std::size_t>& order,
               FakeModuleOptions options = {})
        : ordinal_(ordinal), inputs_(std::move(inputs)), outputs_(std::move(outputs)),
          order_(order), options_(std::move(options)) {
        for (const auto& name : outputs_)
            anchors_.emplace(name, std::make_unique<std::uint8_t>(0));
    }

    TensorMap forward(const TensorMap&) override {
        ++host_forward_count;
        return {};
    }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override { return {}; }
    void forward_device_async(const DeviceTensorMap&) override {}
    void forward_async(const TensorMap& inputs) override {
        if (!inputs.empty())
            throw std::runtime_error("composite passed host inputs");
        order_.push_back(ordinal_);
        ++async_count;
    }
    void sync() override { ++sync_count; }
    cudaStream_t stream() const override { return options_.stream; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return options_.profile_index; }
    std::vector<TensorInfo> input_info() const override { return info(inputs_, true); }
    std::vector<TensorInfo> output_info() const override { return info(outputs_, false); }
    bool has_input(const std::string& name) const override {
        return std::find(inputs_.begin(), inputs_.end(), name) != inputs_.end();
    }
    bool has_output(const std::string& name) const override {
        return std::find(outputs_.begin(), outputs_.end(), name) != outputs_.end();
    }
    DType tensor_dtype(const std::string&) const override { return options_.dtype; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return options_.shape; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             ProfileShapeSelector) const override {
        return options_.shape;
    }
    int32_t optimization_profile_count() const override { return options_.profile_count; }
    bool input_is_dynamic(const std::string&) const override { return options_.dynamic_input; }
    void* device_ptr(const std::string& name) const override {
        const auto external = external_.find(name);
        if (external != external_.end())
            return external->second;
        const auto anchor = anchors_.find(name);
        return options_.null_outputs || anchor == anchors_.end() ? nullptr : anchor->second.get();
    }
    void bind_external(const std::string& name, void* pointer) override {
        if (!has_input(name) || pointer == nullptr)
            throw std::runtime_error("invalid fake binding");
        external_[name] = pointer;
    }
    void bind_external(const std::string& name, void* pointer,
                       const std::vector<int64_t>& shape) override {
        if (shape != options_.shape)
            throw std::runtime_error("invalid fake shape");
        bind_external(name, pointer);
    }
    bool ok() const override { return options_.valid; }
    void keep_alive(std::shared_ptr<void>) override {}
    void set_timing_label(std::string label) override { label_ = std::move(label); }

    int host_forward_count{0};
    int async_count{0};
    int sync_count{0};

  private:
    std::vector<TensorInfo> info(const std::vector<std::string>& names, bool input) const {
        std::vector<TensorInfo> result;
        for (const auto& name : names)
            result.push_back({name, options_.shape, options_.dtype, input});
        return result;
    }

    std::size_t ordinal_{0};
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::vector<std::size_t>& order_;
    FakeModuleOptions options_;
    std::unordered_map<std::string, std::unique_ptr<std::uint8_t>> anchors_;
    std::unordered_map<std::string, void*> external_;
    std::string label_;
};

std::string stage(std::size_t ordinal) {
    if (ordinal < 3)
        return "Reduce";
    if (ordinal < 35)
        return "TD0";
    if (ordinal < 64)
        return "TD1";
    if (ordinal < 96)
        return "BU0";
    if (ordinal < 128)
        return "BU1";
    return "Out";
}

std::string section(std::size_t ordinal) {
    std::string digits = std::to_string(ordinal);
    return "sam2_hoi_pafpn_plan_" + std::string(3 - digits.size(), '0') + digits;
}

nlohmann::json manifest_json() {
    nlohmann::json root;
    root["schema_version"] = 1;
    root["external_inputs"] = {"fpn_input_0", "fpn_input_1", "fpn_input_2"};
    root["nodes"] = nlohmann::json::array();
    for (std::size_t ordinal = 0; ordinal < trtmc::sam2_hoi::kPafpnPlanCount; ++ordinal) {
        nlohmann::json inputs = nlohmann::json::array();
        if (ordinal == 0) {
            inputs.push_back({{"tensor", "input"}, {"source", {{"external", "fpn_input_0"}}}});
        } else if (ordinal == 1) {
            inputs.push_back({{"tensor", "input"}, {"source", {{"external", "fpn_input_1"}}}});
        } else if (ordinal == 2) {
            inputs.push_back(
                {{"tensor", "from0"}, {"source", {{"node", 0}, {"tensor", "output"}}}});
            inputs.push_back(
                {{"tensor", "from1"}, {"source", {{"node", 1}, {"tensor", "output"}}}});
            inputs.push_back({{"tensor", "from2"}, {"source", {{"external", "fpn_input_2"}}}});
        } else {
            inputs.push_back(
                {{"tensor", "input"}, {"source", {{"node", ordinal - 1}, {"tensor", "output"}}}});
        }
        root["nodes"].push_back({
            {"ordinal", ordinal},
            {"id", stage(ordinal) + "::node-" + std::to_string(ordinal)},
            {"section", section(ordinal)},
            {"plan_sha256", std::string(64, '0')},
            {"inputs", std::move(inputs)},
        });
    }
    root["outputs"] = {
        {{"name", "detector_feature_0"}, {"source", {{"node", 130}, {"tensor", "output"}}}},
        {{"name", "detector_feature_1"}, {"source", {{"node", 133}, {"tensor", "output"}}}},
        {{"name", "detector_feature_2"}, {"source", {{"node", 136}, {"tensor", "output"}}}},
    };
    return root;
}

std::vector<char> manifest_bytes(const nlohmann::json& root) {
    const std::string text = root.dump();
    return {text.begin(), text.end()};
}

bool parse_rejected(nlohmann::json root) {
    try {
        const auto bytes = manifest_bytes(root);
        (void)trtmc::sam2_hoi::parse_pafpn_manifest(bytes.data(), bytes.size());
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

std::vector<std::string> inputs_for(std::size_t ordinal) {
    if (ordinal < 2)
        return {"input"};
    if (ordinal == 2)
        return {"from0", "from1", "from2"};
    return {"input"};
}

std::vector<std::string> outputs_for(std::size_t ordinal) {
    (void)ordinal;
    return {"output"};
}

using ModuleMutation = std::function<void(std::size_t, std::vector<std::string>&,
                                          std::vector<std::string>&, FakeModuleOptions&)>;

bool construction_rejected(const std::vector<char>& bytes, const ModuleMutation& mutate) {
    std::vector<std::size_t> order;
    std::size_t next_ordinal = 0;
    try {
        auto manifest = trtmc::sam2_hoi::parse_pafpn_manifest(bytes.data(), bytes.size());
        trtmc::sam2_hoi::PafpnComposite invalid(
            std::move(manifest),
            [&](const std::string& name, cudaStream_t stream) -> std::unique_ptr<ITrtModule> {
                const std::size_t ordinal = next_ordinal++;
                if (name != section(ordinal) || stream != fake_stream())
                    throw std::runtime_error("unexpected test loader request");
                auto inputs = inputs_for(ordinal);
                auto outputs = outputs_for(ordinal);
                FakeModuleOptions options;
                mutate(ordinal, inputs, outputs, options);
                return std::make_unique<FakeModule>(ordinal, std::move(inputs), std::move(outputs),
                                                    order, std::move(options));
            },
            fake_stream());
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    const auto valid_document = manifest_json();
    auto bytes = manifest_bytes(valid_document);
    auto manifest = trtmc::sam2_hoi::parse_pafpn_manifest(bytes.data(), bytes.size());

    auto wrong_schema = valid_document;
    wrong_schema["schema_version"] = 2;
    if (!parse_rejected(std::move(wrong_schema)))
        return 10;
    auto extra_key = valid_document;
    extra_key["unexpected"] = true;
    if (!parse_rejected(std::move(extra_key)))
        return 11;
    auto too_few = valid_document;
    too_few["nodes"].erase(too_few["nodes"].end() - 1);
    if (!parse_rejected(std::move(too_few)))
        return 12;
    auto too_many = valid_document;
    too_many["nodes"].push_back(too_many["nodes"].back());
    if (!parse_rejected(std::move(too_many)))
        return 13;
    auto wrong_ordinal = valid_document;
    wrong_ordinal["nodes"][5]["ordinal"] = 6;
    if (!parse_rejected(std::move(wrong_ordinal)))
        return 14;
    auto wrong_stage = valid_document;
    wrong_stage["nodes"][35]["id"] = "TD0::wrong-stage";
    if (!parse_rejected(std::move(wrong_stage)))
        return 15;
    auto wrong_section = valid_document;
    wrong_section["nodes"][7]["section"] = "sam2_hoi_pafpn_plan_999";
    if (!parse_rejected(std::move(wrong_section)))
        return 16;
    auto duplicate_id = valid_document;
    duplicate_id["nodes"][8]["id"] = duplicate_id["nodes"][7]["id"];
    if (!parse_rejected(std::move(duplicate_id)))
        return 17;
    auto bad_sha = valid_document;
    bad_sha["nodes"][9]["plan_sha256"] = std::string(64, 'A');
    if (!parse_rejected(std::move(bad_sha)))
        return 18;
    auto forward_edge = valid_document;
    forward_edge["nodes"][10]["inputs"][0]["source"]["node"] = 11;
    if (!parse_rejected(std::move(forward_edge)))
        return 19;
    auto missing_root = valid_document;
    missing_root["nodes"][1]["inputs"][0]["source"]["external"] = "fpn_input_0";
    if (!parse_rejected(std::move(missing_root)))
        return 20;
    auto duplicate_destination = valid_document;
    duplicate_destination["nodes"][2]["inputs"][1]["tensor"] = "from0";
    if (!parse_rejected(std::move(duplicate_destination)))
        return 21;
    auto dead_node = valid_document;
    dead_node["nodes"][136]["inputs"][0]["source"]["node"] = 134;
    if (!parse_rejected(std::move(dead_node)))
        return 22;
    auto wrong_output = valid_document;
    wrong_output["outputs"][1]["source"]["node"] = 130;
    if (!parse_rejected(std::move(wrong_output)))
        return 23;
    std::vector<std::size_t> order;
    std::vector<FakeModule*> raw_modules;
    trtmc::sam2_hoi::PafpnComposite composite(
        std::move(manifest),
        [&](const std::string& name, cudaStream_t stream) -> std::unique_ptr<ITrtModule> {
            if (stream != fake_stream())
                throw std::runtime_error("loader stream mismatch");
            const std::size_t ordinal = raw_modules.size();
            if (name != section(ordinal))
                throw std::runtime_error("loader section mismatch");
            auto module = std::make_unique<FakeModule>(ordinal, inputs_for(ordinal),
                                                       outputs_for(ordinal), order);
            raw_modules.push_back(module.get());
            return module;
        },
        fake_stream());

    bool rejected_incomplete = false;
    try {
        composite.forward_async();
    } catch (const std::runtime_error&) {
        rejected_incomplete = true;
    }
    if (!rejected_incomplete)
        return 24;

    FakeModule front(999, {}, {"root0", "root1", "root2"}, order);
    FakeModule detector(1000, {"detector_feature_0", "detector_feature_1", "detector_feature_2"},
                        {}, order);
    composite.bind_external_input("fpn_input_0", front, "root0");
    composite.bind_external_input("fpn_input_1", front, "root1");
    composite.bind_external_input("fpn_input_2", front, "root2");
    composite.bind_output_to("detector_feature_0", detector, "detector_feature_0");
    composite.bind_output_to("detector_feature_1", detector, "detector_feature_1");
    composite.bind_output_to("detector_feature_2", detector, "detector_feature_2");
    composite.forward_async();

    if (order.size() != trtmc::sam2_hoi::kPafpnPlanCount)
        return 1;
    for (std::size_t ordinal = 0; ordinal < order.size(); ++ordinal) {
        if (order[ordinal] != ordinal || raw_modules[ordinal]->async_count != 1 ||
            raw_modules[ordinal]->host_forward_count != 0 || raw_modules[ordinal]->sync_count != 0)
            return 2;
    }
    composite.sync();
    if (raw_modules.back()->sync_count != 1)
        return 3;
    bool rejected_stream_mismatch = false;
    try {
        auto stream_manifest = trtmc::sam2_hoi::parse_pafpn_manifest(bytes.data(), bytes.size());
        std::vector<std::size_t> stream_order;
        trtmc::sam2_hoi::PafpnComposite invalid(
            std::move(stream_manifest),
            [&](const std::string&, cudaStream_t) -> std::unique_ptr<ITrtModule> {
                const std::size_t ordinal = stream_order.size();
                return std::make_unique<FakeModule>(ordinal, inputs_for(ordinal),
                                                    outputs_for(ordinal), stream_order);
            },
            reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(2)));
    } catch (const std::runtime_error&) {
        rejected_stream_mismatch = true;
    }
    if (!rejected_stream_mismatch)
        return 25;

    bool rejected_null_loader_result = false;
    try {
        auto null_manifest = trtmc::sam2_hoi::parse_pafpn_manifest(bytes.data(), bytes.size());
        trtmc::sam2_hoi::PafpnComposite invalid(
            std::move(null_manifest),
            [](const std::string&, cudaStream_t) -> std::unique_ptr<ITrtModule> { return nullptr; },
            fake_stream());
    } catch (const std::runtime_error&) {
        rejected_null_loader_result = true;
    }
    if (!rejected_null_loader_result)
        return 26;

    if (!construction_rejected(bytes,
                               [](std::size_t ordinal, auto&, auto&, FakeModuleOptions& options) {
                                   if (ordinal == 0)
                                       options.profile_count = 2;
                               }))
        return 27;
    if (!construction_rejected(bytes,
                               [](std::size_t ordinal, auto&, auto&, FakeModuleOptions& options) {
                                   if (ordinal == 0)
                                       options.profile_index = 1;
                               }))
        return 28;
    if (!construction_rejected(bytes,
                               [](std::size_t ordinal, auto&, auto&, FakeModuleOptions& options) {
                                   if (ordinal == 0)
                                       options.dynamic_input = true;
                               }))
        return 29;
    if (!construction_rejected(bytes,
                               [](std::size_t ordinal, auto& inputs, auto&, FakeModuleOptions&) {
                                   if (ordinal == 0)
                                       inputs = {"wrong_input"};
                               }))
        return 30;
    if (!construction_rejected(bytes,
                               [](std::size_t ordinal, auto&, auto& outputs, FakeModuleOptions&) {
                                   if (ordinal == 0)
                                       outputs = {"wrong_output"};
                               }))
        return 31;
    if (!construction_rejected(bytes,
                               [](std::size_t ordinal, auto& inputs, auto&, FakeModuleOptions&) {
                                   if (ordinal == 0)
                                       inputs = {"input", "input"};
                               }))
        return 32;
    if (!construction_rejected(bytes,
                               [](std::size_t ordinal, auto&, auto& outputs, FakeModuleOptions&) {
                                   if (ordinal == 0)
                                       outputs = {"output", "output"};
                               }))
        return 33;
    if (!construction_rejected(bytes,
                               [](std::size_t ordinal, auto&, auto&, FakeModuleOptions& options) {
                                   if (ordinal == 2)
                                       options.shape = {2};
                               }))
        return 34;
    if (!construction_rejected(bytes,
                               [](std::size_t ordinal, auto&, auto&, FakeModuleOptions& options) {
                                   if (ordinal == 2)
                                       options.dtype = DType::kInt32;
                               }))
        return 35;
    if (!construction_rejected(bytes,
                               [](std::size_t ordinal, auto&, auto&, FakeModuleOptions& options) {
                                   if (ordinal == 0)
                                       options.null_outputs = true;
                               }))
        return 36;

    std::cout << "Phase-A PAFPN composite CPU tests passed\n";
    return 0;
}
