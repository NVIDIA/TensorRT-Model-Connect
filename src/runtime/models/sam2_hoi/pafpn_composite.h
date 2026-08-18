/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/trt_module.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc::sam2_hoi {

inline constexpr std::size_t kPafpnPlanCount = 137;

struct PafpnSourceSpec {
    enum class Kind { kExternal, kNode };
    Kind kind{Kind::kExternal};
    std::string external;
    std::size_t node{0};
    std::string tensor;
};

struct PafpnInputSpec {
    std::string tensor;
    PafpnSourceSpec source;
};

struct PafpnNodeSpec {
    std::size_t ordinal{0};
    std::string id;
    std::string section;
    std::string plan_sha256;
    std::vector<PafpnInputSpec> inputs;
};

struct PafpnOutputSpec {
    std::string name;
    std::size_t node{0};
    std::string tensor;
};

struct PafpnManifest {
    std::vector<std::string> external_inputs;
    std::vector<PafpnNodeSpec> nodes;
    std::vector<PafpnOutputSpec> outputs;
};

PafpnManifest parse_pafpn_manifest(const void* data, std::size_t size);

using PafpnModuleLoader =
    std::function<std::unique_ptr<ITrtModule>(const std::string& section, cudaStream_t stream)>;

class IPafpnComposite {
  public:
    virtual ~IPafpnComposite() = default;
    virtual void bind_external_input(const std::string& composite_name, ITrtModule& producer,
                                     const std::string& producer_output) = 0;
    virtual void bind_output_to(const std::string& composite_name, ITrtModule& consumer,
                                const std::string& consumer_input) = 0;
    virtual void forward_async() = 0;
    virtual void sync() = 0;
    virtual cudaStream_t stream() const = 0;
};

// PafpnComposite deliberately is not an ITrtModule. It does not own one engine;
// it owns an ordered collection of exact engines and exposes only the small
// orchestration surface required by Sam2HoiPipeline.
class PafpnComposite final : public IPafpnComposite {
  public:
    PafpnComposite(PafpnManifest manifest, PafpnModuleLoader loader, cudaStream_t stream);
    ~PafpnComposite() = default;

    PafpnComposite(const PafpnComposite&) = delete;
    PafpnComposite& operator=(const PafpnComposite&) = delete;

    void bind_external_input(const std::string& composite_name, ITrtModule& producer,
                             const std::string& producer_output) override;
    void bind_output_to(const std::string& composite_name, ITrtModule& consumer,
                        const std::string& consumer_input) override;

    void forward_async() override;
    void sync() override;

    cudaStream_t stream() const override { return stream_; }
    std::size_t module_count() const { return nodes_.size(); }
    bool has_output(const std::string& name) const;
    void* device_ptr(const std::string& name) const;
    DType tensor_dtype(const std::string& name) const;
    std::vector<int64_t> tensor_shape(const std::string& name) const;

  private:
    struct Node {
        PafpnNodeSpec spec;
        std::unique_ptr<ITrtModule> module;
    };
    struct Destination {
        std::size_t node{0};
        std::string tensor;
    };
    struct OutputRef {
        std::size_t node{0};
        std::string tensor;
    };

    static void bind_compatible(ITrtModule& producer, const std::string& output,
                                ITrtModule& consumer, const std::string& input);
    const OutputRef& output_ref(const std::string& name) const;

    // The caller owns stream_. It must outlive all nodes_.
    cudaStream_t stream_{nullptr};
    std::vector<Node> nodes_;
    std::unordered_map<std::string, std::vector<Destination>> external_destinations_;
    std::unordered_map<std::string, OutputRef> outputs_;
    std::unordered_set<std::string> bound_external_inputs_;
    std::unordered_set<std::string> bound_outputs_;
};

} // namespace trtmc::sam2_hoi
