/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Executable release-qualification fixture for the historical cuDNN SDPA
// optional-output failure.  This is deliberately separate from the production
// NativeContiguousAttention implementation:
//
//   legacy: set_generate_stats(false) + logit-max + score-sum-exp
//   lse:    set_generate_stats(true), with neither legacy optional output
//
// The producer launches each mode in a fresh process with an isolated CUDA
// cache and a pinned CUDA-13.0 NVRTC/builtins pair.  A legacy failure is the
// expected result; the standard-LSE graph must build and execute.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <nvrtc.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef TRTMC_CUDNN_FRONTEND_REVISION
#define TRTMC_CUDNN_FRONTEND_REVISION "unavailable"
#endif

namespace fe = cudnn_frontend;
using json = nlohmann::json;

namespace {

constexpr int64_t kQueryUid = 1;
constexpr int64_t kKeyUid = 2;
constexpr int64_t kValueUid = 3;
constexpr int64_t kOutputUid = 4;
constexpr int64_t kSequenceQueryUid = 5;
constexpr int64_t kSequenceKvUid = 6;
constexpr int64_t kLegacyLogitMaxUid = 7;
constexpr int64_t kLegacyScoreSumExpUid = 8;
constexpr int64_t kLseUid = 9;

// This is the exact production Qwen decode-history geometry that exposed the
// regression.  The first Heur-A candidate also fails at T=1,024; T=512 keeps
// the release fixture fast while preserving the historical failure contract.
struct Shape {
    int64_t query_heads{16};
    int64_t kv_heads{8};
    int64_t head_dim{128};
    int64_t query_rows{1};
    int64_t history_rows{512};
    int32_t valid_history_rows{511};
};

struct Arguments {
    std::string mode;
    std::string graph_output;
};

Arguments parse_arguments(int argc, char** argv) {
    if (argc != 5 || std::string(argv[1]) != "--mode" || std::string(argv[3]) != "--graph-output") {
        throw std::invalid_argument("usage: trtmc_nvrtc_optional_output_regression "
                                    "--mode legacy|lse --graph-output <path>");
    }
    Arguments result{argv[2], argv[4]};
    if (result.mode != "legacy" && result.mode != "lse") {
        throw std::invalid_argument("--mode must be legacy or lse");
    }
    if (result.graph_output.empty()) {
        throw std::invalid_argument("--graph-output must not be empty");
    }
    return result;
}

void require_cuda(cudaError_t status, char const* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void require_cudnn(cudnnStatus_t status, char const* operation) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + cudnnGetErrorString(status));
    }
}

void require_good(fe::error_t status, char const* operation) {
    if (!status.is_good()) {
        throw std::runtime_error(std::string(operation) + ": " + status.get_message());
    }
}

class CudnnHandle {
  public:
    CudnnHandle() {
        require_cudnn(cudnnCreate(&value_), "cudnnCreate");
        if (value_ == nullptr) {
            throw std::runtime_error("cudnnCreate returned a null handle");
        }
    }

    ~CudnnHandle() {
        if (value_ != nullptr) {
            cudnnDestroy(value_);
        }
    }

    CudnnHandle(CudnnHandle const&) = delete;
    CudnnHandle& operator=(CudnnHandle const&) = delete;

    operator cudnnHandle_t() const noexcept { return value_; }

  private:
    cudnnHandle_t value_{nullptr};
};

class DynamicLibrary {
  public:
    explicit DynamicLibrary(char const* soname) {
        value_ = dlopen(soname, RTLD_NOW | RTLD_LOCAL);
        if (value_ == nullptr) {
            char const* error = dlerror();
            throw std::runtime_error(std::string("dlopen(") + soname +
                                     ") failed: " + (error == nullptr ? "unknown error" : error));
        }
    }

    ~DynamicLibrary() {
        if (value_ != nullptr) {
            dlclose(value_);
        }
    }

    DynamicLibrary(DynamicLibrary const&) = delete;
    DynamicLibrary& operator=(DynamicLibrary const&) = delete;

  private:
    void* value_{nullptr};
};

class DeviceAllocation {
  public:
    explicit DeviceAllocation(std::size_t bytes) {
        if (bytes != 0) {
            require_cuda(cudaMalloc(&value_, bytes), "cudaMalloc");
        }
    }

    ~DeviceAllocation() {
        if (value_ != nullptr) {
            cudaFree(value_);
        }
    }

    DeviceAllocation(DeviceAllocation const&) = delete;
    DeviceAllocation& operator=(DeviceAllocation const&) = delete;

    void* get() const noexcept { return value_; }

  private:
    void* value_{nullptr};
};

struct GraphFixture {
    std::shared_ptr<fe::graph::Graph> graph;
    std::string serialized_contract;
};

GraphFixture make_graph(Shape const& shape, bool standard_lse) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto query =
        graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("query")
                          .set_uid(kQueryUid)
                          .set_dim({1, shape.query_heads, shape.query_rows, shape.head_dim})
                          .set_stride({shape.query_heads * shape.query_rows * shape.head_dim,
                                       shape.query_rows * shape.head_dim, shape.head_dim, 1}));
    auto key = graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("key_history_token_major")
                                 .set_uid(kKeyUid)
                                 .set_dim({1, shape.kv_heads, shape.history_rows, shape.head_dim})
                                 .set_stride({shape.history_rows * shape.kv_heads * shape.head_dim,
                                              shape.head_dim, shape.kv_heads * shape.head_dim, 1}));
    auto value =
        graph->tensor(fe::graph::Tensor_attributes()
                          .set_name("value_history_token_major")
                          .set_uid(kValueUid)
                          .set_dim({1, shape.kv_heads, shape.history_rows, shape.head_dim})
                          .set_stride({shape.history_rows * shape.kv_heads * shape.head_dim,
                                       shape.head_dim, shape.kv_heads * shape.head_dim, 1}));
    auto sequence_query = graph->tensor(fe::graph::Tensor_attributes()
                                            .set_name("sequence_length_q")
                                            .set_uid(kSequenceQueryUid)
                                            .set_dim({1, 1, 1, 1})
                                            .set_stride({1, 1, 1, 1})
                                            .set_data_type(fe::DataType_t::INT32));
    auto sequence_history = graph->tensor(fe::graph::Tensor_attributes()
                                              .set_name("sequence_length_history")
                                              .set_uid(kSequenceKvUid)
                                              .set_dim({1, 1, 1, 1})
                                              .set_stride({1, 1, 1, 1})
                                              .set_data_type(fe::DataType_t::INT32));

    auto attributes = fe::graph::SDPA_attributes()
                          .set_name(standard_lse ? "trtmc_standard_lse_history_sdpa"
                                                 : "trtmc_legacy_optional_output_history_sdpa")
                          .set_generate_stats(standard_lse)
                          .set_attn_scale(1.0F / std::sqrt(static_cast<float>(shape.head_dim)))
                          .set_padding_mask(true)
                          .set_seq_len_q(sequence_query)
                          .set_seq_len_kv(sequence_history);

    if (!standard_lse) {
        auto logit_max = graph->tensor(
            fe::graph::Tensor_attributes()
                .set_output(true)
                .set_name("legacy_logit_max")
                .set_uid(kLegacyLogitMaxUid)
                .set_dim({1, shape.query_heads, shape.query_rows, 1})
                .set_stride({shape.query_heads * shape.query_rows, shape.query_rows, 1, 1})
                .set_data_type(fe::DataType_t::FLOAT));
        auto score_sum_exp = graph->tensor(
            fe::graph::Tensor_attributes()
                .set_output(true)
                .set_name("legacy_score_sum_exp")
                .set_uid(kLegacyScoreSumExpUid)
                .set_dim({1, shape.query_heads, shape.query_rows, 1})
                .set_stride({shape.query_heads * shape.query_rows, shape.query_rows, 1, 1})
                .set_data_type(fe::DataType_t::FLOAT));
        attributes.set_logit_max(logit_max).set_score_sum_exp(score_sum_exp);
    }

    auto [output, stats] = graph->sdpa(query, key, value, attributes);
    output->set_output(true)
        .set_name("context")
        .set_uid(kOutputUid)
        .set_dim({1, shape.query_heads, shape.query_rows, shape.head_dim})
        .set_stride({shape.query_heads * shape.query_rows * shape.head_dim,
                     shape.query_rows * shape.head_dim, shape.head_dim, 1});

    if (standard_lse) {
        if (stats == nullptr) {
            throw std::runtime_error(
                "standard-LSE graph did not return its requested stats tensor");
        }
        stats->set_output(true)
            .set_name("log_sum_exp")
            .set_uid(kLseUid)
            .set_dim({1, shape.query_heads, shape.query_rows, 1})
            .set_stride({shape.query_heads * shape.query_rows, shape.query_rows, 1, 1})
            .set_data_type(fe::DataType_t::FLOAT);
    } else if (stats != nullptr) {
        throw std::runtime_error("legacy optional-output graph unexpectedly returned LSE stats");
    }

    std::string contract = graph->print();
    bool const has_legacy_max = contract.find("legacy_logit_max") != std::string::npos;
    bool const has_legacy_sum = contract.find("legacy_score_sum_exp") != std::string::npos;
    bool const has_lse = contract.find("log_sum_exp") != std::string::npos;
    if (standard_lse) {
        if (has_legacy_max || has_legacy_sum || !has_lse) {
            throw std::runtime_error("serialized standard-LSE graph contract contains legacy "
                                     "optional outputs or omits log_sum_exp");
        }
    } else if (!has_legacy_max || !has_legacy_sum || has_lse) {
        throw std::runtime_error("serialized legacy graph contract does not contain exactly the "
                                 "historical max/sum-exp optional outputs");
    }
    return {std::move(graph), std::move(contract)};
}

void write_graph_contract(std::string const& path, std::string const& value) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("unable to create graph contract: " + path);
    }
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
    output.flush();
    if (!output) {
        throw std::runtime_error("unable to write graph contract: " + path);
    }
}

json runtime_identity() {
    int nvrtc_major = 0;
    int nvrtc_minor = 0;
    auto const nvrtc_status = nvrtcVersion(&nvrtc_major, &nvrtc_minor);
    if (nvrtc_status != NVRTC_SUCCESS) {
        throw std::runtime_error(std::string("nvrtcVersion failed: ") +
                                 nvrtcGetErrorString(nvrtc_status));
    }

    int runtime_version = 0;
    int driver_version = 0;
    require_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    require_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");

    int device = -1;
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    require_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");

    Dl_info nvrtc_info{};
    auto const nvrtc_address =
        reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(&nvrtcVersion));
    if (dladdr(nvrtc_address, &nvrtc_info) == 0 || nvrtc_info.dli_fname == nullptr) {
        throw std::runtime_error("dladdr could not resolve the mapped NVRTC library");
    }

    return {
        {"pid", static_cast<int64_t>(getpid())},
        {"device", device},
        {"device_name", properties.name},
        {"sm", properties.major * 10 + properties.minor},
        {"cuda_runtime_version", runtime_version},
        {"cuda_driver_api_version", driver_version},
        {"cudnn_backend_version", cudnnGetVersion()},
        {"cudnn_frontend_revision", TRTMC_CUDNN_FRONTEND_REVISION},
        {"nvrtc_major", nvrtc_major},
        {"nvrtc_minor", nvrtc_minor},
        {"nvrtc_dladdr_path", nvrtc_info.dli_fname},
    };
}

json run_legacy(cudnnHandle_t handle, GraphFixture& fixture) {
    require_good(fixture.graph->validate(), "legacy graph validate");
    require_good(fixture.graph->build_operation_graph(handle), "legacy build_operation_graph");
    require_good(fixture.graph->create_execution_plans({fe::HeurMode_t::A}),
                 "legacy create Heur-A execution plans");

    auto const plan_count = fixture.graph->get_execution_plan_count();
    if (plan_count < 1) {
        throw std::runtime_error("legacy Heur-A returned no execution plans");
    }
    std::string plan_name;
    require_good(fixture.graph->get_plan_name_at_index(0, plan_name),
                 "legacy get first Heur-A plan name");
    auto build_status = fixture.graph->build_plan_at_index(handle, 0);
    std::string const build_message = build_status.get_message();

    bool const exact_candidate = plan_name == "eng3_k24=7";
    bool const compilation_failed =
        !build_status.is_good() &&
        build_message.find("compilationResult != NVRTC_SUCCESS") != std::string::npos &&
        build_message.find("CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED") != std::string::npos;
    if (!exact_candidate || !compilation_failed) {
        throw std::runtime_error("legacy first Heur-A candidate did not reproduce the exact "
                                 "eng3_k24=7 NVRTC compilation failure: plan=" +
                                 plan_name + " status=" + build_message);
    }

    return {
        {"plan_count", plan_count},
        {"candidate_index", 0},
        {"candidate_plan", plan_name},
        {"candidate_build_succeeded", build_status.is_good()},
        {"candidate_build_error_code", static_cast<int>(build_status.get_code())},
        {"candidate_build_message", build_message},
        {"expected_nvrtc_failure_observed", true},
        {"fallback_plan_selected", false},
        {"graph_executed", false},
    };
}

json run_lse(cudnnHandle_t handle, Shape const& shape, GraphFixture& fixture) {
    require_good(fixture.graph->validate(), "standard-LSE graph validate");
    require_good(fixture.graph->build(handle, {fe::HeurMode_t::A}), "standard-LSE graph build");

    std::string plan_name;
    require_good(fixture.graph->get_plan_name(plan_name), "standard-LSE selected-plan identity");
    if (plan_name.empty()) {
        throw std::runtime_error("standard-LSE selected-plan identity is empty");
    }

    int64_t workspace_bytes = -1;
    require_good(fixture.graph->get_workspace_size(workspace_bytes),
                 "standard-LSE workspace query");
    if (workspace_bytes < 0) {
        throw std::runtime_error("standard-LSE workspace size is negative");
    }

    auto const query_elements =
        static_cast<std::size_t>(shape.query_heads * shape.query_rows * shape.head_dim);
    auto const kv_elements =
        static_cast<std::size_t>(shape.history_rows * shape.kv_heads * shape.head_dim);
    auto const stats_elements = static_cast<std::size_t>(shape.query_heads * shape.query_rows);
    DeviceAllocation query(query_elements * sizeof(std::uint16_t));
    DeviceAllocation key(kv_elements * sizeof(std::uint16_t));
    DeviceAllocation value(kv_elements * sizeof(std::uint16_t));
    DeviceAllocation output(query_elements * sizeof(std::uint16_t));
    DeviceAllocation stats(stats_elements * sizeof(float));
    DeviceAllocation sequence_query(sizeof(int32_t));
    DeviceAllocation sequence_history(sizeof(int32_t));
    DeviceAllocation workspace(static_cast<std::size_t>(workspace_bytes));

    require_cuda(cudaMemset(query.get(), 0, query_elements * sizeof(std::uint16_t)), "zero query");
    require_cuda(cudaMemset(key.get(), 0, kv_elements * sizeof(std::uint16_t)), "zero key");
    require_cuda(cudaMemset(value.get(), 0, kv_elements * sizeof(std::uint16_t)), "zero value");
    int32_t const valid_query_rows = 1;
    require_cuda(cudaMemcpy(sequence_query.get(), &valid_query_rows, sizeof(valid_query_rows),
                            cudaMemcpyHostToDevice),
                 "copy sequence_length_q");
    require_cuda(cudaMemcpy(sequence_history.get(), &shape.valid_history_rows,
                            sizeof(shape.valid_history_rows), cudaMemcpyHostToDevice),
                 "copy sequence_length_history");

    std::unordered_map<int64_t, void*> variant_pack{
        {kQueryUid, query.get()},
        {kKeyUid, key.get()},
        {kValueUid, value.get()},
        {kOutputUid, output.get()},
        {kSequenceQueryUid, sequence_query.get()},
        {kSequenceKvUid, sequence_history.get()},
        {kLseUid, stats.get()},
    };
    require_good(fixture.graph->execute(handle, variant_pack, workspace.get()),
                 "standard-LSE graph execute");
    require_cuda(cudaDeviceSynchronize(), "standard-LSE synchronize");

    std::vector<std::uint16_t> host_output(query_elements);
    std::vector<float> host_stats(stats_elements);
    require_cuda(cudaMemcpy(host_output.data(), output.get(),
                            host_output.size() * sizeof(host_output.front()),
                            cudaMemcpyDeviceToHost),
                 "copy standard-LSE output");
    require_cuda(cudaMemcpy(host_stats.data(), stats.get(),
                            host_stats.size() * sizeof(host_stats.front()), cudaMemcpyDeviceToHost),
                 "copy standard-LSE stats");
    for (auto const value : host_output) {
        if (value != 0) {
            throw std::runtime_error("zero-input standard-LSE output is not zero");
        }
    }
    for (auto const value : host_stats) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("standard-LSE stats contain a non-finite value");
        }
    }

    return {
        {"selected_plan", plan_name},
        {"workspace_bytes", workspace_bytes},
        {"graph_build_succeeded", true},
        {"graph_executed", true},
        {"device_synchronize_succeeded", true},
        {"finite_lse_observed", true},
        {"legacy_optional_outputs_bound", false},
    };
}

json graph_contract(std::string const& mode, std::string const& serialized_contract) {
    bool const standard_lse = mode == "lse";
    return {
        {"generate_stats", standard_lse},
        {"optional_logit_max", !standard_lse},
        {"optional_score_sum_exp", !standard_lse},
        {"output_context", true},
        {"output_log_sum_exp", standard_lse},
        {"serialized_contract_bytes", serialized_contract.size()},
        {"serialized_contains_legacy_logit_max",
         serialized_contract.find("legacy_logit_max") != std::string::npos},
        {"serialized_contains_legacy_score_sum_exp",
         serialized_contract.find("legacy_score_sum_exp") != std::string::npos},
        {"serialized_contains_log_sum_exp",
         serialized_contract.find("log_sum_exp") != std::string::npos},
    };
}

} // namespace

int main(int argc, char** argv) {
    try {
        auto const arguments = parse_arguments(argc, argv);

        // Keep the exact builtins DSO mapped until the producer has inspected
        // /proc/<pid>/maps.  LD_LIBRARY_PATH is constructed by the producer
        // from already-open CUDA-13.0 library descriptors.
        DynamicLibrary builtins("libnvrtc-builtins.so.13.0");
        auto identity = runtime_identity();
        if (identity.at("nvrtc_major") != 13 || identity.at("nvrtc_minor") != 0) {
            throw std::runtime_error("this negative replay requires exactly NVRTC 13.0");
        }

        Shape const shape{};
        bool const standard_lse = arguments.mode == "lse";
        auto fixture = make_graph(shape, standard_lse);
        write_graph_contract(arguments.graph_output, fixture.serialized_contract);

        CudnnHandle handle;
        auto result = standard_lse ? run_lse(handle, shape, fixture) : run_legacy(handle, fixture);

        json receipt{
            {"schema_version", "trtmc.nvrtc-optional-output-probe/v1"},
            {"mode", arguments.mode},
            {"shape",
             {
                 {"query_heads", shape.query_heads},
                 {"kv_heads", shape.kv_heads},
                 {"head_dim", shape.head_dim},
                 {"query_rows", shape.query_rows},
                 {"history_rows", shape.history_rows},
                 {"valid_history_rows", shape.valid_history_rows},
             }},
            {"runtime", std::move(identity)},
            {"graph_contract", graph_contract(arguments.mode, fixture.serialized_contract)},
            {"result", std::move(result)},
            {"probe_passed", true},
        };
        std::cout << receipt.dump() << '\n';
        std::cout.flush();

        // The qualification producer owns stdin and releases the child only
        // after it has pinned and validated the live process mappings.
        char const* hold = std::getenv("TRTMC_NVRTC_PROBE_WAIT_FOR_RELEASE");
        if (hold != nullptr && std::string(hold) == "1") {
            std::string release;
            if (!std::getline(std::cin, release)) {
                throw std::runtime_error("producer closed the release pipe before mapping capture");
            }
        }
        return 0;
    } catch (std::exception const& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
