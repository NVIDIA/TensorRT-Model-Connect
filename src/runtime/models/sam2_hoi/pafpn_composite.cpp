/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2_hoi/pafpn_composite.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <initializer_list>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace trtmc::sam2_hoi {
namespace {

using json = nlohmann::json;

constexpr std::array<std::string_view, 3> kExternalInputs{"fpn_input_0", "fpn_input_1",
                                                          "fpn_input_2"};
constexpr std::array<std::string_view, 3> kPublicOutputs{"detector_feature_0", "detector_feature_1",
                                                         "detector_feature_2"};
constexpr std::array<std::size_t, 3> kPublicOutputNodes{130, 133, 136};

[[noreturn]] void manifest_error(const std::string& message) {
    throw std::runtime_error("Invalid SAM2 HOI PAFPN manifest: " + message);
}

void require_object_keys(const json& value, std::initializer_list<std::string_view> expected,
                         const std::string& where) {
    if (!value.is_object())
        manifest_error(where + " must be an object");
    std::set<std::string> actual;
    for (const auto& item : value.items())
        actual.insert(item.key());
    std::set<std::string> wanted;
    for (const auto key : expected)
        wanted.emplace(key);
    if (actual != wanted)
        manifest_error(where + " has missing or unknown keys");
}

std::string expected_section(std::size_t ordinal) {
    std::ostringstream name;
    name << "sam2_hoi_pafpn_plan_" << std::setw(3) << std::setfill('0') << ordinal;
    return name.str();
}

std::string_view expected_stage(std::size_t ordinal) {
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

bool valid_sha256(const std::string& value) {
    return value.size() == 64 &&
           std::all_of(value.begin(), value.end(), [](unsigned char character) {
               return std::isdigit(character) != 0 ||
                      (character >= static_cast<unsigned char>('a') &&
                       character <= static_cast<unsigned char>('f'));
           });
}

PafpnSourceSpec parse_source(const json& source, std::size_t consumer) {
    if (!source.is_object())
        manifest_error("node source must be an object");
    const bool has_external = source.contains("external");
    const bool has_node = source.contains("node");
    if (has_external == has_node)
        manifest_error("node source must choose exactly one of external or node");

    PafpnSourceSpec result;
    if (has_external) {
        require_object_keys(source, {"external"}, "external source");
        result.kind = PafpnSourceSpec::Kind::kExternal;
        result.external = source.at("external").get<std::string>();
        return result;
    }

    require_object_keys(source, {"node", "tensor"}, "internal source");
    result.kind = PafpnSourceSpec::Kind::kNode;
    result.node = source.at("node").get<std::size_t>();
    result.tensor = source.at("tensor").get<std::string>();
    if (result.node >= consumer)
        manifest_error("internal source must precede its consumer");
    if (result.tensor.empty())
        manifest_error("internal source tensor must not be empty");
    return result;
}

template <std::size_t N>
bool exact_names(const std::vector<std::string>& actual,
                 const std::array<std::string_view, N>& expected) {
    if (actual.size() != expected.size())
        return false;
    for (std::size_t index = 0; index < N; ++index) {
        if (actual[index] != expected[index])
            return false;
    }
    return true;
}

std::set<std::string> tensor_names(const std::vector<TensorInfo>& info) {
    std::set<std::string> names;
    for (const auto& tensor : info) {
        if (tensor.name.empty() || !names.insert(tensor.name).second)
            throw std::runtime_error("SAM2 HOI PAFPN module has duplicate tensor names");
    }
    return names;
}

json parse_manifest_json(const void* data, std::size_t size) {
    if (data == nullptr || size == 0)
        manifest_error("payload is empty");

    try {
        const auto* begin = static_cast<const char*>(data);
        return json::parse(begin, begin + size);
    } catch (const std::exception& error) {
        manifest_error(std::string("JSON parse failed: ") + error.what());
    }
}

void validate_manifest_root(const json& root) {
    require_object_keys(root, {"schema_version", "external_inputs", "nodes", "outputs"}, "root");
    if (root.at("schema_version").get<int>() != 1)
        manifest_error("schema_version must be 1");
}

std::vector<std::string> parse_external_inputs(const json& root) {
    auto external_inputs = root.at("external_inputs").get<std::vector<std::string>>();
    if (!exact_names(external_inputs, kExternalInputs))
        manifest_error("external_inputs must be the three canonical FPN roots in order");
    return external_inputs;
}

void validate_node_metadata(const PafpnNodeSpec& node, std::size_t ordinal,
                            std::unordered_set<std::string>& ids) {
    if (node.ordinal != ordinal)
        manifest_error("node ordinals must be contiguous and ordered");
    if (node.id.empty() || !ids.insert(node.id).second)
        manifest_error("node IDs must be non-empty and unique");
    const std::string stage_prefix = std::string(expected_stage(ordinal)) + "::";
    if (node.id.rfind(stage_prefix, 0) != 0)
        manifest_error("node stage partition/order drift at ordinal " + std::to_string(ordinal));
    if (node.section != expected_section(ordinal))
        manifest_error("node section naming drift at ordinal " + std::to_string(ordinal));
    if (!valid_sha256(node.plan_sha256))
        manifest_error("node plan_sha256 must be lowercase SHA-256");
}

PafpnInputSpec parse_node_input(const json& input, std::size_t ordinal,
                                std::unordered_set<std::string>& destination_names) {
    require_object_keys(input, {"tensor", "source"}, "node input");
    PafpnInputSpec spec;
    spec.tensor = input.at("tensor").get<std::string>();
    if (spec.tensor.empty() || !destination_names.insert(spec.tensor).second)
        manifest_error("node destination inputs must be non-empty and unique");
    spec.source = parse_source(input.at("source"), ordinal);
    return spec;
}

std::vector<PafpnInputSpec>
parse_node_inputs(const json& inputs, std::size_t ordinal,
                  const std::vector<std::string>& external_inputs,
                  const std::vector<bool>& reachable_from_external,
                  std::unordered_set<std::string>& used_external_inputs) {
    if (!inputs.is_array() || inputs.empty())
        manifest_error("every node must have at least one input");

    std::vector<PafpnInputSpec> result;
    std::unordered_set<std::string> destination_names;
    bool reachable = false;
    for (const auto& input : inputs) {
        auto spec = parse_node_input(input, ordinal, destination_names);
        if (spec.source.kind == PafpnSourceSpec::Kind::kExternal) {
            if (std::find(external_inputs.begin(), external_inputs.end(), spec.source.external) ==
                external_inputs.end()) {
                manifest_error("node refers to an unknown external input");
            }
            used_external_inputs.insert(spec.source.external);
            reachable = true;
        } else if (reachable_from_external.at(spec.source.node)) {
            reachable = true;
        }
        result.push_back(std::move(spec));
    }
    if (!reachable)
        manifest_error("node is not reachable from an external input");
    return result;
}

PafpnNodeSpec parse_node(const json& raw, std::size_t ordinal,
                         const std::vector<std::string>& external_inputs,
                         const std::vector<bool>& reachable_from_external,
                         std::unordered_set<std::string>& ids,
                         std::unordered_set<std::string>& used_external_inputs) {
    require_object_keys(raw, {"ordinal", "id", "section", "plan_sha256", "inputs"}, "node");
    PafpnNodeSpec node;
    node.ordinal = raw.at("ordinal").get<std::size_t>();
    node.id = raw.at("id").get<std::string>();
    node.section = raw.at("section").get<std::string>();
    node.plan_sha256 = raw.at("plan_sha256").get<std::string>();
    validate_node_metadata(node, ordinal, ids);
    node.inputs = parse_node_inputs(raw.at("inputs"), ordinal, external_inputs,
                                    reachable_from_external, used_external_inputs);
    return node;
}

std::vector<PafpnNodeSpec> parse_nodes(const json& root,
                                       const std::vector<std::string>& external_inputs) {
    const auto& raw_nodes = root.at("nodes");
    if (!raw_nodes.is_array() || raw_nodes.size() != kPafpnPlanCount)
        manifest_error("nodes must contain exactly 137 entries");

    std::vector<PafpnNodeSpec> nodes;
    nodes.reserve(kPafpnPlanCount);
    std::unordered_set<std::string> ids;
    std::unordered_set<std::string> used_external_inputs;
    std::vector<bool> reachable_from_external(kPafpnPlanCount, false);
    for (std::size_t ordinal = 0; ordinal < raw_nodes.size(); ++ordinal) {
        nodes.push_back(parse_node(raw_nodes.at(ordinal), ordinal, external_inputs,
                                   reachable_from_external, ids, used_external_inputs));
        reachable_from_external[ordinal] = true;
    }
    if (used_external_inputs.size() != kExternalInputs.size())
        manifest_error("all three external inputs must be used");
    return nodes;
}

PafpnOutputSpec parse_output(const json& raw, std::size_t index,
                             std::set<std::pair<std::size_t, std::string>>& public_sources) {
    require_object_keys(raw, {"name", "source"}, "public output");
    const auto name = raw.at("name").get<std::string>();
    if (name != kPublicOutputs[index])
        manifest_error("public outputs must use canonical names and order");
    const auto& source = raw.at("source");
    require_object_keys(source, {"node", "tensor"}, "public output source");
    PafpnOutputSpec output{name, source.at("node").get<std::size_t>(),
                           source.at("tensor").get<std::string>()};
    if (output.node != kPublicOutputNodes[index] || output.tensor != "output" ||
        !public_sources.emplace(output.node, output.tensor).second) {
        manifest_error("public output source is invalid");
    }
    return output;
}

std::vector<PafpnOutputSpec> parse_outputs(const json& root) {
    const auto& raw_outputs = root.at("outputs");
    if (!raw_outputs.is_array() || raw_outputs.size() != kPublicOutputs.size())
        manifest_error("outputs must contain exactly three entries");

    std::vector<PafpnOutputSpec> outputs;
    outputs.reserve(kPublicOutputs.size());
    std::set<std::pair<std::size_t, std::string>> public_sources;
    for (std::size_t index = 0; index < raw_outputs.size(); ++index)
        outputs.push_back(parse_output(raw_outputs.at(index), index, public_sources));
    return outputs;
}

void validate_node_liveness(const std::vector<PafpnNodeSpec>& nodes,
                            const std::vector<PafpnOutputSpec>& outputs) {
    std::vector<bool> reaches_public_output(kPafpnPlanCount, false);
    for (const auto& output : outputs)
        reaches_public_output[output.node] = true;

    // Reverse liveness: every node must contribute to at least one public output.
    for (std::size_t index = kPafpnPlanCount; index-- > 0;) {
        if (!reaches_public_output[index])
            continue;
        for (const auto& input : nodes[index].inputs) {
            if (input.source.kind == PafpnSourceSpec::Kind::kNode)
                reaches_public_output[input.source.node] = true;
        }
    }
    if (std::find(reaches_public_output.begin(), reaches_public_output.end(), false) !=
        reaches_public_output.end()) {
        manifest_error("manifest contains a dead node");
    }
}

void validate_composite_arguments(const PafpnManifest& manifest, const PafpnModuleLoader& loader,
                                  cudaStream_t stream) {
    if (stream == nullptr)
        throw std::invalid_argument("SAM2 HOI PAFPN composite requires a CUDA stream");
    if (!loader)
        throw std::invalid_argument("SAM2 HOI PAFPN composite requires a module loader");
    if (manifest.nodes.size() != kPafpnPlanCount)
        throw std::invalid_argument("SAM2 HOI PAFPN composite requires 137 nodes");
}

template <typename DestinationMap, typename OutputMap>
std::vector<std::set<std::string>> collect_referenced_outputs(const PafpnManifest& manifest,
                                                              DestinationMap& external_destinations,
                                                              OutputMap& outputs) {
    std::vector<std::set<std::string>> referenced_outputs(kPafpnPlanCount);
    for (std::size_t consumer_index = 0; consumer_index < manifest.nodes.size(); ++consumer_index) {
        for (const auto& input : manifest.nodes[consumer_index].inputs) {
            if (input.source.kind == PafpnSourceSpec::Kind::kExternal) {
                external_destinations[input.source.external].push_back(
                    {consumer_index, input.tensor});
            } else {
                referenced_outputs.at(input.source.node).insert(input.source.tensor);
            }
        }
    }
    for (const auto& output : manifest.outputs) {
        if (!outputs
                 .emplace(output.name, typename OutputMap::mapped_type{output.node, output.tensor})
                 .second) {
            throw std::runtime_error("SAM2 HOI PAFPN duplicate public output: " + output.name);
        }
        referenced_outputs.at(output.node).insert(output.tensor);
    }
    if (external_destinations.size() != kExternalInputs.size() ||
        outputs.size() != kPublicOutputs.size()) {
        throw std::runtime_error("SAM2 HOI PAFPN external contract is incomplete");
    }
    return referenced_outputs;
}

std::unique_ptr<ITrtModule>
load_valid_module(const PafpnNodeSpec& spec, const PafpnModuleLoader& loader, cudaStream_t stream) {
    auto module = loader(spec.section, stream);
    if (module == nullptr || !module->ok() || module->stream() != stream)
        throw std::runtime_error("SAM2 HOI PAFPN module is invalid: " + spec.section);
    if (module->optimization_profile_count() != 1 || module->profile_idx() != 0)
        throw std::runtime_error("SAM2 HOI PAFPN profile contract drift: " + spec.section);
    return module;
}

void validate_module_tensor_contract(const PafpnNodeSpec& spec, const ITrtModule& module,
                                     const std::set<std::string>& referenced_outputs) {
    std::set<std::string> manifest_inputs;
    for (const auto& input : spec.inputs) {
        if (module.input_is_dynamic(input.tensor))
            throw std::runtime_error("SAM2 HOI PAFPN inputs must be static: " + spec.section);
        manifest_inputs.insert(input.tensor);
    }
    if (manifest_inputs != tensor_names(module.input_info()))
        throw std::runtime_error("SAM2 HOI PAFPN input contract drift: " + spec.section);
    if (referenced_outputs != tensor_names(module.output_info()))
        throw std::runtime_error("SAM2 HOI PAFPN output contract drift: " + spec.section);
}

} // namespace

PafpnManifest parse_pafpn_manifest(const void* data, std::size_t size) {
    const json root = parse_manifest_json(data, size);
    validate_manifest_root(root);
    PafpnManifest manifest;
    manifest.external_inputs = parse_external_inputs(root);
    manifest.nodes = parse_nodes(root, manifest.external_inputs);
    manifest.outputs = parse_outputs(root);
    validate_node_liveness(manifest.nodes, manifest.outputs);
    return manifest;
}

PafpnComposite::PafpnComposite(PafpnManifest manifest, PafpnModuleLoader loader,
                               cudaStream_t stream)
    : stream_(stream) {
    validate_composite_arguments(manifest, loader, stream_);
    const auto referenced_outputs =
        collect_referenced_outputs(manifest, external_destinations_, outputs_);

    nodes_.reserve(kPafpnPlanCount);
    for (auto& spec : manifest.nodes) {
        auto module = load_valid_module(spec, loader, stream_);
        validate_module_tensor_contract(spec, *module, referenced_outputs.at(spec.ordinal));
        // Bind internal inputs as soon as the consumer is deserialized. This
        // releases its now-unused owned input buffers before the next plan is
        // loaded, while every earlier producer output remains alive.
        for (const auto& input : spec.inputs) {
            if (input.source.kind == PafpnSourceSpec::Kind::kNode) {
                bind_compatible(*nodes_.at(input.source.node).module, input.source.tensor, *module,
                                input.tensor);
            }
        }
        module->set_timing_label(spec.section);
        nodes_.push_back(Node{std::move(spec), std::move(module)});
    }
}

void PafpnComposite::bind_compatible(ITrtModule& producer, const std::string& output,
                                     ITrtModule& consumer, const std::string& input) {
    if (!producer.has_output(output) || !consumer.has_input(input))
        throw std::runtime_error("SAM2 HOI PAFPN edge names do not exist");
    const auto shape = producer.tensor_shape(output);
    if (shape.empty() || shape != consumer.tensor_shape(input) ||
        producer.tensor_dtype(output) != consumer.tensor_dtype(input)) {
        throw std::runtime_error("SAM2 HOI PAFPN edge shape/dtype mismatch");
    }
    void* pointer = producer.device_ptr(output);
    if (pointer == nullptr)
        throw std::runtime_error("SAM2 HOI PAFPN producer output pointer is null");
    consumer.bind_external(input, pointer, shape);
    if (consumer.device_ptr(input) != pointer)
        throw std::runtime_error("SAM2 HOI PAFPN external binding did not take effect");
}

void PafpnComposite::bind_external_input(const std::string& composite_name, ITrtModule& producer,
                                         const std::string& producer_output) {
    const auto destinations = external_destinations_.find(composite_name);
    if (destinations == external_destinations_.end())
        throw std::invalid_argument("Unknown SAM2 HOI PAFPN external input: " + composite_name);
    if (!bound_external_inputs_.insert(composite_name).second)
        throw std::invalid_argument("SAM2 HOI PAFPN external input was bound twice: " +
                                    composite_name);
    if (producer.stream() != stream_)
        throw std::runtime_error("SAM2 HOI PAFPN front/composite stream mismatch");
    for (const auto& destination : destinations->second) {
        bind_compatible(producer, producer_output, *nodes_.at(destination.node).module,
                        destination.tensor);
    }
}

void PafpnComposite::bind_output_to(const std::string& composite_name, ITrtModule& consumer,
                                    const std::string& consumer_input) {
    const auto& output = output_ref(composite_name);
    if (!bound_outputs_.insert(composite_name).second)
        throw std::invalid_argument("SAM2 HOI PAFPN output was bound twice: " + composite_name);
    if (consumer.stream() != stream_)
        throw std::runtime_error("SAM2 HOI PAFPN composite/detector stream mismatch");
    bind_compatible(*nodes_.at(output.node).module, output.tensor, consumer, consumer_input);
}

void PafpnComposite::forward_async() {
    if (bound_external_inputs_.size() != kExternalInputs.size() ||
        bound_outputs_.size() != kPublicOutputs.size()) {
        throw std::runtime_error("SAM2 HOI PAFPN composite is not fully bound");
    }
    for (auto& node : nodes_)
        node.module->forward_async({});
}

void PafpnComposite::sync() {
    if (nodes_.empty())
        throw std::runtime_error("SAM2 HOI PAFPN composite has no modules");
    nodes_.back().module->sync();
}

const PafpnComposite::OutputRef& PafpnComposite::output_ref(const std::string& name) const {
    const auto output = outputs_.find(name);
    if (output == outputs_.end())
        throw std::invalid_argument("Unknown SAM2 HOI PAFPN output: " + name);
    return output->second;
}

bool PafpnComposite::has_output(const std::string& name) const {
    return outputs_.count(name) != 0;
}

void* PafpnComposite::device_ptr(const std::string& name) const {
    const auto& output = output_ref(name);
    return nodes_.at(output.node).module->device_ptr(output.tensor);
}

DType PafpnComposite::tensor_dtype(const std::string& name) const {
    const auto& output = output_ref(name);
    return nodes_.at(output.node).module->tensor_dtype(output.tensor);
}

std::vector<int64_t> PafpnComposite::tensor_shape(const std::string& name) const {
    const auto& output = output_ref(name);
    return nodes_.at(output.node).module->tensor_shape(output.tensor);
}

} // namespace trtmc::sam2_hoi
