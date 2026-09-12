/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/model.h"
#include "trtmc/internal/structure.h"
#include "trtmc/runtime/family_factory.h"

#include <limits>

namespace {
using namespace trtmc::internal;

std::string hex(trtmc::Span<const std::uint8_t> bytes) {
    constexpr char digits[] = "0123456789abcdef";
    std::string out;
    for (const auto value : bytes) {
        out += digits[value >> 4];
        out += digits[value & 15];
    }
    return out;
}

class StructureModel final : public IModel, public IMolecularDocumentToStructure {
  public:
    explicit StructureModel(std::string mode) : mode_(std::move(mode)) {
        fields_[1].default_value = ConfigValue{mode_ != "without_confidence"};
        fields_[3].default_value =
            ConfigValue{std::int64_t{mode_ == "without_confidence" ? 7 : 200}};
    }
    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<TaskInstance> task_bindings() override {
        if (mode_ == "disabled")
            return {};
        return {bind<IMolecularDocumentToStructure>(*this, {fields_, 4})};
    }
    MolecularStructureResult run(const MolecularDocumentToStructureRequest& request,
                                 ConfigView config) override {
        if (mode_ == "must_not_run")
            throw std::runtime_error("invalid Config reached structure execution");
        if (request.encoding != "yaml" && request.encoding != "json" && request.encoding != "b2rq")
            throw UnsupportedTask("fixture does not accept this molecular document encoding");
        if (request.encoding == "b2rq" &&
            (request.document.size() < 4 || request.document[0] != 'B' ||
             request.document[1] != '2' || request.document[2] != 'R' ||
             request.document[3] != 'Q'))
            throw std::invalid_argument("fixture prepared document has an invalid header");
        const trtmc::Span<const ConfigField> fields{fields_, 4};
        const auto seed = config_get<std::int64_t>(config, fields, "seed").value();
        const auto steps = config_get<std::int64_t>(config, fields, "sampling_steps").value();
        const auto confidence = config_get<bool>(config, fields, "include_confidence").value();
        const auto format = config_get<std::string_view>(config, fields, "output_format").value();
        if (seed < 0)
            throw ConfigError("fixture seed must be nonnegative");
        if (steps < 1)
            throw ConfigError("fixture sampling steps must be positive");
        if (format != "mmcif" && format != "pdb")
            throw ConfigError("fixture output format must be mmcif or pdb");
        MolecularStructureResult result;
        result.format =
            format == "mmcif" ? trtmc::StructureFormat::kMmcif : trtmc::StructureFormat::kPdb;
        result.structure = (format == "mmcif" ? "data_fixture\n# " : "HEADER fixture\nREMARK ") +
                           std::string(request.encoding) + ":" + hex(request.document) + "\n";
        result.metadata_json = "{\"encoding\":\"" + std::string(request.encoding) +
                               "\",\"source_path\":\"" + std::string(request.source_path) +
                               "\",\"seed\":" + std::to_string(seed) +
                               ",\"sampling_steps\":" + std::to_string(steps) + "}";
        if (confidence)
            result.confidence = trtmc::StructureConfidence{
                0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 66.0F, 77.0F, {88.0F, 0.0F, 99.0F}};
        if (mode_ == "empty_structure")
            result.structure.clear();
        if (mode_ == "bad_format")
            result.format = static_cast<trtmc::StructureFormat>(99);
        if (mode_ == "bad_confidence")
            result.confidence->ptm = std::numeric_limits<float>::quiet_NaN();
        if (mode_ == "bad_plddt")
            result.confidence->plddt[1] = std::numeric_limits<float>::infinity();
        return result;
    }

  private:
    std::string mode_;
    ConfigField fields_[4] = {
        {"seed", ConfigKind::I64, ConfigValue{std::int64_t{42}}, "Fixture sampling seed"},
        {"include_confidence", ConfigKind::Bool, ConfigValue{true}, "Return complete confidence"},
        {"output_format", ConfigKind::String, ConfigValue{std::string_view{"mmcif"}},
         "Structure output format"},
        {"sampling_steps", ConfigKind::I64, ConfigValue{std::int64_t{200}},
         "Fixture sampling steps"},
    };
};
} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return new StructureModel(context.reader.info().task);
}
