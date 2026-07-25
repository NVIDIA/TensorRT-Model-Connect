/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Product-owned internal calibrator for native dynamic-memory bundles. It
// loads the real bundle through the same public factory as `trtmc run`, then
// discovers the independently versioned private qualification interface.

#include "native_dynamic_memory_calibrator_schema.h"
#include "native_dynamic_memory_calibrator_paths.h"
#include "runtime/domains/text/dynamic_memory/runtime_kv_setup.h"
#include "runtime/domains/text/dynamic_memory/runtime_memory_qualification.h"
#include "trtmc/bundle.h"
#include "trtmc/pipeline.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#ifndef TRTMC_HAS_NVML
#define TRTMC_HAS_NVML 0
#endif
#ifndef TRTMC_INTERNAL_RUNTIME_LIB_RELATIVE
#define TRTMC_INTERNAL_RUNTIME_LIB_RELATIVE "."
#endif
#ifndef TRTMC_INTERNAL_PRODUCT_VERSION
#error "TRTMC_INTERNAL_PRODUCT_VERSION must be defined by the product build"
#endif
#ifndef TRTMC_INTERNAL_CALIBRATOR_BUILD_IDENTITY
#error "TRTMC_INTERNAL_CALIBRATOR_BUILD_IDENTITY must be defined by the product build"
#endif
#if TRTMC_HAS_NVML
#include <nvml.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;

#if defined(__GNUC__) || defined(__clang__)
[[gnu::used]]
#endif
constexpr char kProductIdentityMarker[] =
    "TRTMC_INTERNAL_DYNAMIC_MEMORY_CALIBRATOR_IDENTITY_V1:"
    TRTMC_INTERNAL_PRODUCT_VERSION ":" TRTMC_INTERNAL_CALIBRATOR_BUILD_IDENTITY;
static_assert(sizeof(TRTMC_INTERNAL_CALIBRATOR_BUILD_IDENTITY) == 65,
              "calibrator build identity must be a lowercase SHA256");

class QualificationDiagnosticError : public std::runtime_error {
  public:
    QualificationDiagnosticError(std::string message, json diagnostic)
        : std::runtime_error(std::move(message)), diagnostic_(std::move(diagnostic)) {}

    const json& diagnostic() const noexcept { return diagnostic_; }

  private:
    json diagnostic_;
};

std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs, const char* what) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs)
        throw std::overflow_error(std::string(what) + " overflows uint64");
    return lhs + rhs;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        throw std::invalid_argument("alignment must be a power of two");
    const auto remainder = value % alignment;
    return remainder == 0 ? value : checked_add(value, alignment - remainder, "aligned bytes");
}

std::uint64_t ceil_fraction_denominator(std::uint64_t bytes, double fraction) {
    if (!(std::isfinite(fraction) && fraction > 0.0 && fraction <= 1.0))
        throw std::invalid_argument("auto policy fraction must be in (0, 1]");
    const long double quotient =
        static_cast<long double>(bytes) / static_cast<long double>(fraction);
    if (quotient > static_cast<long double>(std::numeric_limits<std::uint64_t>::max()))
        throw std::overflow_error("auto policy safe-byte requirement overflows uint64");
    return static_cast<std::uint64_t>(std::ceil(quotient));
}

struct Arguments {
    bool query_module_loading_mode{false};
    bool query_product_identity{false};
    std::string bundle;
    std::string token_file;
    std::string logits_file;
    std::int32_t max_new_tokens{0};
    std::uint64_t max_sequence_length{0};
    std::uint64_t second_max_sequence_length{0};
    std::uint64_t controlled_reservation_target_tokens{0};
    std::uint64_t kv_cache_bytes{0};
    double kv_cache_fraction{0.0};
    std::uint32_t repeat{1};
    std::uint32_t load_cycles{1};
    bool warmup_load_cycle{false};
    std::vector<std::string> profile_sweep_token_files;
    std::vector<std::string> backend_dirs;
    std::vector<std::string> model_plugin_dirs;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: trtmc_dynamic_memory_qualify --query-product-identity\n"
        "   or: trtmc_dynamic_memory_qualify --query-module-loading-mode\n"
        "   or: trtmc_dynamic_memory_qualify --bundle MODEL.trtfb "
                  "--tokens IDS.txt --logits LOGITS.bin [--max-new-tokens N] "
                  "[--max-sequence-length N] [--kv-cache-bytes N | --kv-cache-fraction F] "
                  "[--second-max-sequence-length N] "
                  "[--controlled-reservation-target-tokens N] "
                  "[--repeat N | --load-cycles N] [--warmup-load-cycle] "
                  "[--profile-sweep-tokens IDS.txt ...] "
                  "[--backend-dir DIR] [--model-plugin-dir DIR]");
}

std::uint64_t parse_u64(const std::string& text, const char* name) {
    std::size_t consumed = 0;
    const auto value = std::stoull(text, &consumed, 10);
    if (consumed != text.size())
        usage_error(std::string(name) + " must be an unsigned integer");
    return value;
}

std::int32_t parse_i32(const std::string& text, const char* name) {
    std::size_t consumed = 0;
    const auto value = std::stoll(text, &consumed, 10);
    if (consumed != text.size() || value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max()) {
        usage_error(std::string(name) + " must fit int32");
    }
    return static_cast<std::int32_t>(value);
}

double parse_fraction(const std::string& text) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size() || !(value > 0.0 && value <= 1.0))
        usage_error("--kv-cache-fraction must be in (0, 1]");
    return value;
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments out;
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        auto require_value = [&]() -> std::string {
            if (index + 1 >= argc)
                usage_error(flag + " requires a value");
            return argv[++index];
        };
        if (flag == "--query-module-loading-mode")
            out.query_module_loading_mode = true;
        else if (flag == "--query-product-identity")
            out.query_product_identity = true;
        else if (flag == "--bundle")
            out.bundle = require_value();
        else if (flag == "--tokens")
            out.token_file = require_value();
        else if (flag == "--logits")
            out.logits_file = require_value();
        else if (flag == "--max-new-tokens")
            out.max_new_tokens = parse_i32(require_value(), "--max-new-tokens");
        else if (flag == "--max-sequence-length")
            out.max_sequence_length = parse_u64(require_value(), "--max-sequence-length");
        else if (flag == "--second-max-sequence-length")
            out.second_max_sequence_length =
                parse_u64(require_value(), "--second-max-sequence-length");
        else if (flag == "--controlled-reservation-target-tokens")
            out.controlled_reservation_target_tokens =
                parse_u64(require_value(), "--controlled-reservation-target-tokens");
        else if (flag == "--kv-cache-bytes")
            out.kv_cache_bytes = parse_u64(require_value(), "--kv-cache-bytes");
        else if (flag == "--kv-cache-fraction")
            out.kv_cache_fraction = parse_fraction(require_value());
        else if (flag == "--repeat")
            out.repeat = static_cast<std::uint32_t>(parse_u64(require_value(), "--repeat"));
        else if (flag == "--load-cycles")
            out.load_cycles =
                static_cast<std::uint32_t>(parse_u64(require_value(), "--load-cycles"));
        else if (flag == "--warmup-load-cycle")
            out.warmup_load_cycle = true;
        else if (flag == "--profile-sweep-tokens")
            out.profile_sweep_token_files.push_back(require_value());
        else if (flag == "--backend-dir")
            out.backend_dirs.push_back(require_value());
        else if (flag == "--model-plugin-dir")
            out.model_plugin_dirs.push_back(require_value());
        else
            usage_error("unknown argument: " + flag);
    }
    if (out.query_module_loading_mode || out.query_product_identity) {
        if (argc != 2) {
            usage_error(
                "internal product queries are isolated and accept no other arguments");
        }
        return out;
    }
    const bool profile_sweep = !out.profile_sweep_token_files.empty();
    if (out.bundle.empty() || out.logits_file.empty() ||
        (out.token_file.empty() && !profile_sweep)) {
        usage_error("--bundle, --logits, and either --tokens or "
                    "--profile-sweep-tokens are required");
    }
    if (profile_sweep && !out.token_file.empty())
        usage_error("--tokens cannot be combined with --profile-sweep-tokens");
    if (out.max_new_tokens < 0)
        usage_error("--max-new-tokens must be non-negative");
    if (out.kv_cache_bytes != 0 && out.kv_cache_fraction != 0.0)
        usage_error("--kv-cache-bytes and --kv-cache-fraction are mutually exclusive");
    if (out.repeat == 0 || out.load_cycles == 0)
        usage_error("--repeat and --load-cycles must be positive");
    if (out.repeat > 1 && out.load_cycles > 1)
        usage_error("--repeat and --load-cycles cannot both exceed one");
    if (out.second_max_sequence_length != 0) {
        if (out.max_sequence_length == 0 ||
            out.second_max_sequence_length <= out.max_sequence_length) {
            usage_error("--second-max-sequence-length requires a smaller explicit "
                        "--max-sequence-length");
        }
        if (out.repeat != 1 || out.load_cycles != 1 || out.kv_cache_bytes != 0 ||
            out.kv_cache_fraction != 0.0) {
            usage_error("--second-max-sequence-length requires one request per "
                        "lifetime and explicit sequence-length policy");
        }
    }
    if (out.controlled_reservation_target_tokens != 0) {
        if (out.max_sequence_length != 0 || out.second_max_sequence_length != 0 ||
            out.kv_cache_bytes != 0 || out.kv_cache_fraction != 0.0 || out.repeat != 1 ||
            out.load_cycles != 1) {
            usage_error("--controlled-reservation-target-tokens requires one "
                        "auto-policy request per lifetime");
        }
    }
    if (profile_sweep) {
        if (out.max_new_tokens != 0 && out.max_new_tokens != 1)
            usage_error("--profile-sweep-tokens fixes --max-new-tokens at one");
        if (out.repeat != 1 || out.load_cycles != 1 || out.warmup_load_cycle ||
            out.second_max_sequence_length != 0 || out.controlled_reservation_target_tokens != 0) {
            usage_error("--profile-sweep-tokens is a self-contained two-sweep "
                        "single-lifetime mode");
        }
    }
    trtmc::qualification::validate_single_warmup_arguments(
        out.warmup_load_cycle, out.repeat, out.load_cycles, out.second_max_sequence_length,
        out.controlled_reservation_target_tokens);
    return out;
}

json requested_policy_json(const Arguments& args) {
    if (args.kv_cache_bytes != 0)
        return trtmc::qualification::make_bytes_policy(args.kv_cache_bytes);
    if (args.kv_cache_fraction != 0.0)
        return trtmc::qualification::make_fraction_policy(args.kv_cache_fraction);
    if (args.max_sequence_length != 0)
        return trtmc::qualification::make_max_sequence_length_policy(args.max_sequence_length);
    return trtmc::qualification::make_auto_policy();
}

std::vector<std::int32_t> read_tokens(const std::string& path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open token file: " + path);
    std::vector<std::int32_t> tokens;
    std::int64_t value = 0;
    while (input >> value) {
        if (value < std::numeric_limits<std::int32_t>::min() ||
            value > std::numeric_limits<std::int32_t>::max()) {
            throw std::runtime_error("token file contains a value outside int32");
        }
        tokens.push_back(static_cast<std::int32_t>(value));
    }
    if (!input.eof())
        throw std::runtime_error("token file contains a non-integer token");
    return tokens;
}

template <typename T>
void write_scalar(std::ostream& output, const T& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_logits(const std::string& path,
                  const trtmc::RuntimeMemoryQualificationResultV1& result) {
    if (result.step_logits.empty())
        throw std::runtime_error("qualification returned no logits");
    const auto columns = result.step_logits.front().size();
    if (columns == 0)
        throw std::runtime_error("qualification returned an empty logit row");
    for (const auto& row : result.step_logits) {
        if (row.size() != columns)
            throw std::runtime_error("qualification returned ragged logits");
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("cannot open logits output: " + path);
    constexpr char kMagic[8] = {'T', 'R', 'T', 'M', 'C', 'Q', 'L', '1'};
    output.write(kMagic, sizeof(kMagic));
    const std::uint32_t version = 1;
    const std::uint32_t dtype = 1; // IEEE-754 float32
    const std::uint64_t rows = result.step_logits.size();
    const std::uint64_t cols = columns;
    write_scalar(output, version);
    write_scalar(output, dtype);
    write_scalar(output, rows);
    write_scalar(output, cols);
    for (const auto& row : result.step_logits) {
        output.write(reinterpret_cast<const char*>(row.data()),
                     static_cast<std::streamsize>(row.size() * sizeof(float)));
    }
    if (!output)
        throw std::runtime_error("failed while writing logits output: " + path);
}

json logits_artifact_json(const std::string& path,
                          const trtmc::RuntimeMemoryQualificationResultV1& result) {
    if (result.step_logits.empty() || result.step_logits.front().empty())
        throw std::runtime_error("qualification returned no logits for artifact metadata");
    return {
        {"path", std::filesystem::absolute(path).string()},
        {"format", "trtmc-qualification-logits-v1"},
        {"dtype", "float32"},
        {"rows", result.step_logits.size()},
        {"vocab_size", result.step_logits.front().size()},
    };
}

json parse_receipt(const std::string& receipt) {
    if (receipt.empty())
        throw std::runtime_error("runtime-memory qualification returned an empty receipt");
    return json::parse(receipt);
}

void append_default_search_paths(const char* argv0, trtmc::LoadOptionsV2& options) {
    const auto paths = trtmc::qualification::internal_calibrator_search_paths(
        std::filesystem::path(argv0),
        std::filesystem::path(TRTMC_INTERNAL_RUNTIME_LIB_RELATIVE));
    options.backend_search_paths.insert(options.backend_search_paths.end(),
                                        paths.backend.begin(), paths.backend.end());
    options.model_plugin_search_paths.insert(options.model_plugin_search_paths.end(),
                                             paths.model_plugin.begin(),
                                             paths.model_plugin.end());
}

json invocation_json(const trtmc::RuntimeMemoryQualificationResultV1& result) {
    auto invocations = json::array();
    for (const auto& trace : result.invocations) {
        invocations.push_back({
            {"invocation_index", trace.invocation_index},
            {"role", trace.role},
            {"plan_id", trace.plan_id},
            {"profile_id", trace.profile_id},
            {"chunk_range", {trace.chunk_begin, trace.chunk_end}},
            {"launch_count", trace.launch_count},
            {"kv_allocation_id", trace.kv_allocation_id},
            {"kv_base_address", trace.kv_base_address},
            {"context_device_memory_bytes", trace.context_device_memory_bytes},
            {"H", trace.history_tokens},
            {"A", trace.active_tokens},
            {"T", trace.bound_tokens},
            {"R", trace.reserved_tokens},
            {"cuda_graph_status", trace.cuda_graph_status},
            {"kv_device_to_host_bytes", trace.kv_device_to_host_bytes},
            {"kv_append_bytes", trace.kv_append_bytes},
            {"full_history_device_to_device_bytes", trace.full_history_device_to_device_bytes},
        });
    }
    return invocations;
}

std::vector<std::int32_t>
step_top1_token_ids(const trtmc::RuntimeMemoryQualificationResultV1& result) {
    std::vector<std::int32_t> top1;
    top1.reserve(result.step_logits.size());
    for (const auto& logits : result.step_logits) {
        if (logits.empty())
            throw std::runtime_error("qualification returned an empty logit row");
        top1.push_back(static_cast<std::int32_t>(
            std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()))));
    }
    return top1;
}

json cold_warm_output_equivalence(const trtmc::RuntimeMemoryQualificationResultV1& warmup,
                                  const trtmc::RuntimeMemoryQualificationResultV1& measured) {
    const auto warmup_top1 = step_top1_token_ids(warmup);
    const auto measured_top1 = step_top1_token_ids(measured);
    return trtmc::qualification::make_cold_warm_output_equivalence(
        warmup.prompt_tokens == measured.prompt_tokens,
        warmup.prefill_launches == measured.prefill_launches,
        warmup.decode_launches == measured.decode_launches,
        warmup.final_kv_position == measured.final_kv_position,
        warmup.selected_token_ids == measured.selected_token_ids, warmup_top1 == measured_top1,
        trtmc::qualification::float32_logits_bitwise_equal(warmup.step_logits,
                                                           measured.step_logits));
}

json success_json(const std::string& model_id, const std::string& pipeline_type,
                  const trtmc::RuntimeMemoryQualificationResultV1& result,
                  const std::string& logits_path) {
    const json receipt = parse_receipt(result.runtime_memory_receipt_json);
    json out{
        {"status", "ok"},
        {"qualification_api_version", trtmc::kRuntimeMemoryQualificationApiVersionV1},
        {"model_id", model_id},
        {"pipeline_type", pipeline_type},
        {"prompt_tokens", result.prompt_tokens},
        {"selected_token_ids", result.selected_token_ids},
        {"prefill_chunk_limit", result.prefill_chunk_limit},
        {"prefill_launches", result.prefill_launches},
        {"decode_launches", result.decode_launches},
        {"final_kv_position", result.final_kv_position},
        {"runtime_kv_capacity_tokens", result.runtime_kv_capacity_tokens},
        {"effective_request_limit", result.effective_request_limit},
        {"runtime_memory_receipt", receipt},
        {"kv_allocation_id", receipt.at("kv_allocation_id")},
        {"logits_artifact", logits_artifact_json(logits_path, result)},
    };
    out["invocations"] = invocation_json(result);
    out["step_top1_token_ids"] = step_top1_token_ids(result);
    return out;
}

struct ProcessMemoryEntry {
    std::uint32_t pid{0};
    std::uint64_t used_bytes{0};
};

struct ProcessMemoryObservation {
    std::uint64_t current_process_used_bytes{0};
    std::uint64_t all_compute_process_used_bytes{0};
    std::uint64_t nvml_device_total_bytes{0};
    std::uint64_t nvml_device_reserved_bytes{0};
    std::uint64_t nvml_device_free_bytes{0};
    std::uint64_t nvml_device_used_bytes{0};
    std::vector<ProcessMemoryEntry> compute_processes;
};

#if TRTMC_HAS_NVML
class ProcessMemorySampler {
  public:
    ProcessMemorySampler() {
        check(nvmlInit_v2(), "nvmlInit_v2");
        const auto device_status = cudaGetDevice(&logical_device_index_);
        if (device_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaGetDevice failed during NVML setup: ") +
                                     cudaGetErrorString(device_status));
        }
        char pci_bus_id[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE]{};
        const auto pci_status =
            cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), logical_device_index_);
        if (pci_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaDeviceGetPCIBusId failed during NVML setup: ") +
                cudaGetErrorString(pci_status));
        }
        pci_bus_id_ = pci_bus_id;
        check(nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id, &device_),
              "nvmlDeviceGetHandleByPciBusId_v2");
        unsigned int current_mig_mode = NVML_DEVICE_MIG_DISABLE;
        unsigned int pending_mig_mode = NVML_DEVICE_MIG_DISABLE;
        const auto mig_status = nvmlDeviceGetMigMode(device_, &current_mig_mode, &pending_mig_mode);
        if (mig_status == NVML_SUCCESS && current_mig_mode == NVML_DEVICE_MIG_ENABLE) {
            throw std::runtime_error("native dynamic-memory qualification requires a full GPU; "
                                     "MIG instance attribution is not supported");
        }
        if (mig_status != NVML_SUCCESS && mig_status != NVML_ERROR_NOT_SUPPORTED)
            check(mig_status, "nvmlDeviceGetMigMode");
        check(nvmlDeviceGetIndex(device_, &physical_device_index_), "nvmlDeviceGetIndex");
        char uuid[NVML_DEVICE_UUID_V2_BUFFER_SIZE]{};
        check(nvmlDeviceGetUUID(device_, uuid, sizeof(uuid)), "nvmlDeviceGetUUID");
        gpu_uuid_ = uuid;
        pid_ = static_cast<unsigned int>(getpid());
    }

    ProcessMemoryObservation sample() const {
        std::vector<nvmlProcessInfo_t> processes(64);
        for (int attempt = 0; attempt < 3; ++attempt) {
            auto count = static_cast<unsigned int>(processes.size());
            const auto status =
                nvmlDeviceGetComputeRunningProcesses_v3(device_, &count, processes.data());
            if (status == NVML_ERROR_INSUFFICIENT_SIZE) {
                processes.resize(std::max<std::size_t>(count + 16, processes.size() * 2));
                continue;
            }
            check(status, "nvmlDeviceGetComputeRunningProcesses_v3");
            ProcessMemoryObservation observation;
            observation.compute_processes.reserve(count);
            bool found_current_process = false;
            for (unsigned int index = 0; index < count; ++index) {
                if (processes[index].usedGpuMemory == NVML_VALUE_NOT_AVAILABLE) {
                    throw std::runtime_error("NVML did not report process GPU memory");
                }
                const auto used_bytes = static_cast<std::uint64_t>(processes[index].usedGpuMemory);
                observation.all_compute_process_used_bytes =
                    checked_add(observation.all_compute_process_used_bytes, used_bytes,
                                "NVML all-process GPU memory");
                observation.compute_processes.push_back({processes[index].pid, used_bytes});
                if (processes[index].pid == pid_) {
                    observation.current_process_used_bytes =
                        checked_add(observation.current_process_used_bytes, used_bytes,
                                    "NVML current-process GPU memory");
                    found_current_process = true;
                }
            }
            if (!found_current_process)
                throw std::runtime_error("NVML did not list the qualification runner process");
            std::sort(observation.compute_processes.begin(), observation.compute_processes.end(),
                      [](const ProcessMemoryEntry& lhs, const ProcessMemoryEntry& rhs) {
                          return lhs.pid < rhs.pid;
                      });

            nvmlMemory_v2_t memory{};
            memory.version = nvmlMemory_v2;
            check(nvmlDeviceGetMemoryInfo_v2(device_, &memory), "nvmlDeviceGetMemoryInfo_v2");
            observation.nvml_device_total_bytes = memory.total;
            observation.nvml_device_reserved_bytes = memory.reserved;
            observation.nvml_device_free_bytes = memory.free;
            observation.nvml_device_used_bytes = memory.used;
            return observation;
        }
        throw std::runtime_error("NVML process list changed during every sampling attempt");
    }

    json metadata() const {
        return {
            {"source", "nvmlDeviceGetComputeRunningProcesses_v3"},
            {"pid", pid_},
            {"cuda_logical_device_index", logical_device_index_},
            {"physical_device_index", physical_device_index_},
            {"pci_bus_id", pci_bus_id_},
            {"gpu_uuid", gpu_uuid_},
            {"captures_all_compute_processes", true},
            {"device_memory_source", "nvmlDeviceGetMemoryInfo_v2"},
        };
    }

  private:
    static void check(nvmlReturn_t status, const char* operation) {
        if (status != NVML_SUCCESS) {
            throw std::runtime_error(std::string(operation) +
                                     " failed: " + nvmlErrorString(status));
        }
    }

    nvmlDevice_t device_{};
    unsigned int pid_{0};
    int logical_device_index_{0};
    unsigned int physical_device_index_{0};
    std::string pci_bus_id_;
    std::string gpu_uuid_;
};

ProcessMemorySampler& process_memory_sampler() {
    static ProcessMemorySampler sampler;
    return sampler;
}
#endif

struct DeviceMemorySample {
    std::uint64_t free_bytes{0};
    std::uint64_t total_bytes{0};
    std::uint64_t process_used_bytes{0};
    std::uint64_t all_compute_process_used_bytes{0};
    std::uint64_t nvml_device_total_bytes{0};
    std::uint64_t nvml_device_reserved_bytes{0};
    std::uint64_t nvml_device_free_bytes{0};
    std::uint64_t nvml_device_used_bytes{0};
    std::uint64_t post_nvml_free_bytes{0};
    std::uint64_t post_nvml_total_bytes{0};
    std::vector<ProcessMemoryEntry> compute_processes;
};

struct CudaMemorySample {
    std::uint64_t free_bytes{0};
    std::uint64_t total_bytes{0};
};

CudaMemorySample sample_cuda_device_memory() {
    const auto sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaDeviceSynchronize failed during memory sampling: ") +
            cudaGetErrorString(sync_status));
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const auto info_status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (info_status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemGetInfo failed during memory sampling: ") +
                                 cudaGetErrorString(info_status));
    }
    return {
        static_cast<std::uint64_t>(free_bytes),
        static_cast<std::uint64_t>(total_bytes),
    };
}

DeviceMemorySample enrich_device_memory_sample(std::uint64_t free_bytes,
                                               std::uint64_t total_bytes) {
    ProcessMemoryObservation observation;
#if TRTMC_HAS_NVML
    observation = process_memory_sampler().sample();
#else
    observation.current_process_used_bytes = total_bytes - free_bytes;
    observation.all_compute_process_used_bytes = total_bytes - free_bytes;
    observation.nvml_device_total_bytes = total_bytes;
    observation.nvml_device_free_bytes = free_bytes;
    observation.nvml_device_used_bytes = total_bytes - free_bytes;
#endif
    const auto post_nvml = sample_cuda_device_memory();
    return {
        free_bytes,
        total_bytes,
        observation.current_process_used_bytes,
        observation.all_compute_process_used_bytes,
        observation.nvml_device_total_bytes,
        observation.nvml_device_reserved_bytes,
        observation.nvml_device_free_bytes,
        observation.nvml_device_used_bytes,
        post_nvml.free_bytes,
        post_nvml.total_bytes,
        std::move(observation.compute_processes),
    };
}

DeviceMemorySample sample_device_memory() {
    const auto cuda = sample_cuda_device_memory();
    return enrich_device_memory_sample(cuda.free_bytes, cuda.total_bytes);
}

json sample_json(const DeviceMemorySample& sample) {
    auto processes = json::array();
    for (const auto& process : sample.compute_processes) {
        processes.push_back({
            {"pid", process.pid},
            {"used_bytes", process.used_bytes},
        });
    }
    return {
        {"free_bytes", sample.free_bytes},
        {"total_bytes", sample.total_bytes},
        {"used_bytes", sample.total_bytes - sample.free_bytes},
        {"process_used_bytes", sample.process_used_bytes},
        {"all_compute_process_used_bytes", sample.all_compute_process_used_bytes},
        {"other_compute_process_used_bytes",
         sample.all_compute_process_used_bytes - sample.process_used_bytes},
        {"nvml_device_total_bytes", sample.nvml_device_total_bytes},
        {"nvml_device_reserved_bytes", sample.nvml_device_reserved_bytes},
        {"nvml_device_free_bytes", sample.nvml_device_free_bytes},
        {"nvml_device_used_bytes", sample.nvml_device_used_bytes},
        {"post_nvml_free_bytes", sample.post_nvml_free_bytes},
        {"post_nvml_total_bytes", sample.post_nvml_total_bytes},
        {"compute_processes", std::move(processes)},
    };
}

struct CudaModuleLoadingModeEvidence {
    std::string mode;
    std::int32_t driver_value{0};
};

CudaModuleLoadingModeEvidence query_cuda_module_loading_mode() {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11070
    void* entry_point = nullptr;
    cudaDriverEntryPointQueryResult query_result = cudaDriverEntryPointSymbolNotFound;
    const auto lookup_status = cudaGetDriverEntryPointByVersion(
        "cuModuleGetLoadingMode", &entry_point, 11070, cudaEnableDefault, &query_result);
    if (lookup_status != cudaSuccess || query_result != cudaDriverEntryPointSuccess ||
        entry_point == nullptr) {
        throw std::runtime_error(
            "internal calibrator could not resolve cuModuleGetLoadingMode from the active CUDA "
            "driver");
    }

    using QueryModuleLoadingMode = CUresult(CUDAAPI*)(CUmoduleLoadingMode*);
    const auto query = reinterpret_cast<QueryModuleLoadingMode>(entry_point);
    CUmoduleLoadingMode mode{};
    const auto query_status = query(&mode);
    if (query_status != CUDA_SUCCESS) {
        throw std::runtime_error("cuModuleGetLoadingMode failed with CUDA driver status " +
                                 std::to_string(static_cast<std::int32_t>(query_status)));
    }
    if (mode == CU_MODULE_LAZY_LOADING)
        return {"lazy", static_cast<std::int32_t>(mode)};
    if (mode == CU_MODULE_EAGER_LOADING)
        return {"eager", static_cast<std::int32_t>(mode)};
    throw std::runtime_error("cuModuleGetLoadingMode returned an unknown loading mode " +
                             std::to_string(static_cast<std::int32_t>(mode)));
#else
    throw std::runtime_error(
        "internal calibrator requires CUDA 11.7 or newer for cuModuleGetLoadingMode");
#endif
}

using RuntimePhaseAfterSnapshot = std::function<void(const char*, const DeviceMemorySample&)>;

class RuntimePhaseMemoryObserverScope {
  public:
    explicit RuntimePhaseMemoryObserverScope(json& samples,
                                             RuntimePhaseAfterSnapshot after_snapshot = {})
        : samples_(samples), after_snapshot_(std::move(after_snapshot)) {
        trtmc::set_runtime_device_memory_qualification_observer(
            [this](const char* phase, const trtmc::RuntimeDeviceMemorySnapshot& snapshot) {
                const auto sample =
                    enrich_device_memory_sample(snapshot.free_bytes, snapshot.total_bytes);
                samples_.push_back(trtmc::qualification::make_runtime_phase_memory_sample(
                    phase, snapshot.device, sample_json(sample)));
                if (after_snapshot_)
                    after_snapshot_(phase, sample);
            });
    }

    RuntimePhaseMemoryObserverScope(const RuntimePhaseMemoryObserverScope&) = delete;
    RuntimePhaseMemoryObserverScope& operator=(const RuntimePhaseMemoryObserverScope&) = delete;

    ~RuntimePhaseMemoryObserverScope() {
        trtmc::set_runtime_device_memory_qualification_observer({});
    }

  private:
    json& samples_;
    RuntimePhaseAfterSnapshot after_snapshot_;
};

class RuntimePreSnapshotActionScope {
  public:
    explicit RuntimePreSnapshotActionScope(
        trtmc::RuntimeDeviceMemoryQualificationPreSnapshotAction action) {
        trtmc::set_runtime_device_memory_qualification_pre_snapshot_action(std::move(action));
    }

    RuntimePreSnapshotActionScope(const RuntimePreSnapshotActionScope&) = delete;
    RuntimePreSnapshotActionScope& operator=(const RuntimePreSnapshotActionScope&) = delete;

    ~RuntimePreSnapshotActionScope() {
        trtmc::set_runtime_device_memory_qualification_pre_snapshot_action({});
    }
};

std::int64_t retained_process_bytes(const DeviceMemorySample& before,
                                    const DeviceMemorySample& after) {
    return static_cast<std::int64_t>(after.process_used_bytes) -
           static_cast<std::int64_t>(before.process_used_bytes);
}

std::int64_t retained_device_wide_bytes(const DeviceMemorySample& before,
                                        const DeviceMemorySample& after) {
    const auto before_used = before.total_bytes - before.free_bytes;
    const auto after_used = after.total_bytes - after.free_bytes;
    return static_cast<std::int64_t>(after_used) - static_cast<std::int64_t>(before_used);
}

class DeviceReservation {
  public:
    explicit DeviceReservation(std::uint64_t bytes,
                               std::uint64_t max_chunk_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL)
        : bytes_(0) {
        reserve_more(bytes, max_chunk_bytes);
    }

    void reserve_more(std::uint64_t bytes,
                      std::uint64_t max_chunk_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL) {
        if (bytes == 0 ||
            bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::invalid_argument("controlled reservation byte count is invalid");
        }
        if (max_chunk_bytes == 0)
            throw std::invalid_argument("controlled reservation chunk size is invalid");
        const auto updated_bytes =
            checked_add(bytes_, bytes, "controlled reservation accumulated bytes");
        const auto chunk_count_u64 = 1U + (bytes - 1U) / max_chunk_bytes;
        if (chunk_count_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) -
                                  addresses_.size()) {
            throw std::overflow_error("controlled reservation chunk count overflows size_t");
        }
        const auto chunk_count = static_cast<std::size_t>(chunk_count_u64);
        addresses_.reserve(addresses_.size() + chunk_count);
        chunk_bytes_.reserve(chunk_bytes_.size() + chunk_count);
        std::vector<void*> new_addresses;
        std::vector<std::uint64_t> new_chunk_bytes;
        new_addresses.reserve(chunk_count);
        new_chunk_bytes.reserve(chunk_count);
        auto remaining = bytes;
        while (remaining != 0) {
            const auto chunk_bytes = std::min(remaining, max_chunk_bytes);
            void* address = nullptr;
            const auto status = cudaMalloc(&address, static_cast<std::size_t>(chunk_bytes));
            if (status != cudaSuccess) {
                for (auto iterator = new_addresses.rbegin(); iterator != new_addresses.rend();
                     ++iterator) {
                    (void)cudaFree(*iterator);
                }
                throw std::runtime_error(std::string("controlled cudaMalloc failed after ") +
                                         std::to_string(new_addresses.size()) +
                                         " chunks: " + cudaGetErrorString(status));
            }
            new_addresses.push_back(address);
            new_chunk_bytes.push_back(chunk_bytes);
            remaining -= chunk_bytes;
        }
        addresses_.insert(addresses_.end(), new_addresses.begin(), new_addresses.end());
        chunk_bytes_.insert(chunk_bytes_.end(), new_chunk_bytes.begin(), new_chunk_bytes.end());
        bytes_ = updated_bytes;
    }

    DeviceReservation(const DeviceReservation&) = delete;
    DeviceReservation& operator=(const DeviceReservation&) = delete;

    ~DeviceReservation() { release_noexcept(); }

    void release() {
        while (!addresses_.empty()) {
            auto* address = addresses_.back();
            const auto status = cudaFree(address);
            if (status != cudaSuccess) {
                throw std::runtime_error(std::string("controlled cudaFree failed: ") +
                                         cudaGetErrorString(status));
            }
            bytes_ -= chunk_bytes_.back();
            addresses_.pop_back();
            chunk_bytes_.pop_back();
        }
        if (bytes_ != 0)
            throw std::logic_error("controlled reservation byte ledger did not reach zero");
    }

    std::uint64_t release_last() {
        if (addresses_.empty())
            throw std::logic_error("controlled reservation has no releasable tail allocation");
        auto* address = addresses_.back();
        const auto chunk_bytes = chunk_bytes_.back();
        const auto status = cudaFree(address);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("controlled cudaFree tail failed: ") +
                                     cudaGetErrorString(status));
        }
        addresses_.pop_back();
        chunk_bytes_.pop_back();
        bytes_ -= chunk_bytes;
        return chunk_bytes;
    }

    std::uint64_t bytes() const noexcept { return bytes_; }
    std::uint64_t tail_bytes() const noexcept {
        return chunk_bytes_.empty() ? 0 : chunk_bytes_.back();
    }
    std::uint64_t address() const noexcept {
        return addresses_.empty() ? 0 : reinterpret_cast<std::uintptr_t>(addresses_.front());
    }
    std::size_t allocation_count() const noexcept { return addresses_.size(); }
    json allocations_json() const {
        auto allocations = json::array();
        for (std::size_t index = 0; index < addresses_.size(); ++index) {
            allocations.push_back({
                {"index", index},
                {"address", reinterpret_cast<std::uintptr_t>(addresses_[index])},
                {"bytes", chunk_bytes_[index]},
            });
        }
        return allocations;
    }

  private:
    void release_noexcept() noexcept {
        while (!addresses_.empty()) {
            auto* address = addresses_.back();
            const auto chunk_bytes = chunk_bytes_.back();
            addresses_.pop_back();
            chunk_bytes_.pop_back();
            (void)cudaFree(address);
            bytes_ = chunk_bytes <= bytes_ ? bytes_ - chunk_bytes : 0;
        }
        bytes_ = 0;
    }

    std::vector<void*> addresses_;
    std::vector<std::uint64_t> chunk_bytes_;
    std::uint64_t bytes_{0};
};

struct QualificationCycle {
    trtmc::RuntimeMemoryQualificationResultV1 result;
    std::string model_id;
    std::string pipeline_type;
    json sequential_requests = trtmc::qualification::make_sequential_request_samples();
    DeviceMemorySample after_requests;
};

QualificationCycle run_cycle(const Arguments& args, const trtmc::LoadOptionsV2& options,
                             const trtmc::RuntimeMemoryQualificationRequestV1& request) {
    auto pipeline = trtmc::load(args.bundle, options);
    auto* qualifier = dynamic_cast<trtmc::IRuntimeMemoryQualificationV1*>(pipeline.get());
    if (qualifier == nullptr) {
        throw std::runtime_error(
            "loaded pipeline does not implement runtime-memory qualification V1");
    }
    if (qualifier->runtime_memory_qualification_api_version() !=
        trtmc::kRuntimeMemoryQualificationApiVersionV1) {
        throw std::runtime_error("runtime-memory qualification API version mismatch");
    }

    QualificationCycle cycle;
    cycle.model_id = pipeline->model_id();
    cycle.pipeline_type = pipeline->pipeline_type();
    std::uint64_t stable_allocation_id = 0;
    for (std::uint32_t index = 0; index < args.repeat; ++index) {
        const auto before = sample_device_memory();
        auto result = qualifier->qualify_runtime_memory(request);
        const auto after = sample_device_memory();
        const auto receipt = parse_receipt(result.runtime_memory_receipt_json);
        const auto allocation_id = receipt.at("kv_allocation_id").get<std::uint64_t>();
        if (index == 0)
            stable_allocation_id = allocation_id;
        else if (allocation_id != stable_allocation_id)
            throw std::logic_error("sequential qualification request replaced the KV allocation");
        cycle.sequential_requests.push_back({
            {"request_index", index},
            {"before", sample_json(before)},
            {"after", sample_json(after)},
            {"kv_allocation_id", allocation_id},
            {"final_kv_position", result.final_kv_position},
        });
        cycle.result = std::move(result);
    }
    cycle.after_requests = sample_device_memory();
    return cycle;
}

struct QualificationLifetime {
    QualificationCycle cycle;
    DeviceMemorySample before_load;
    DeviceMemorySample after_unload;
    json runtime_phase_memory_samples = trtmc::qualification::make_runtime_phase_memory_samples();
};

QualificationLifetime run_lifetime(const Arguments& args, const trtmc::LoadOptionsV2& options,
                                   const trtmc::RuntimeMemoryQualificationRequestV1& request,
                                   RuntimePhaseAfterSnapshot after_snapshot = {}) {
    QualificationLifetime lifetime;
    RuntimePhaseMemoryObserverScope observer(lifetime.runtime_phase_memory_samples,
                                             std::move(after_snapshot));
    lifetime.before_load = sample_device_memory();
    lifetime.cycle = run_cycle(args, options, request);
    lifetime.after_unload = sample_device_memory();
    return lifetime;
}

DeviceMemorySample require_phase_sample(const QualificationLifetime& lifetime,
                                        const char* required_phase) {
    const json* found = nullptr;
    for (const auto& sample : lifetime.runtime_phase_memory_samples) {
        if (sample.at("phase").get<std::string>() != required_phase)
            continue;
        if (found != nullptr) {
            throw std::runtime_error(std::string("runtime phase was sampled more than once: ") +
                                     required_phase);
        }
        found = &sample;
    }
    if (found == nullptr)
        throw std::runtime_error(std::string("runtime phase sample is missing: ") + required_phase);
    return {
        found->at("free_bytes").get<std::uint64_t>(),
        found->at("total_bytes").get<std::uint64_t>(),
        found->at("process_used_bytes").get<std::uint64_t>(),
    };
}

std::uint64_t positive_growth(std::uint64_t before, std::uint64_t after) {
    return after > before ? after - before : 0;
}

json lifetime_json(const QualificationLifetime& lifetime, const json& policy, const char* label,
                   bool measured) {
    if (!policy.is_object() || !policy.contains("kind"))
        throw std::invalid_argument("qualification lifetime requires a typed policy object");
    const auto receipt = parse_receipt(lifetime.cycle.result.runtime_memory_receipt_json);
    const auto process_growth =
        static_cast<std::int64_t>(lifetime.cycle.after_requests.process_used_bytes) -
        static_cast<std::int64_t>(lifetime.before_load.process_used_bytes);
    const auto before_device_used =
        lifetime.before_load.total_bytes - lifetime.before_load.free_bytes;
    const auto after_requests_device_used =
        lifetime.cycle.after_requests.total_bytes - lifetime.cycle.after_requests.free_bytes;
    json out{
        {"label", label},
        {"measured", measured},
        {"policy", policy},
        {"runtime_kv_capacity_tokens", receipt.at("runtime_kv_capacity_tokens")},
        {"kv_allocation_id", receipt.at("kv_allocation_id")},
        {"runtime_memory_receipt", receipt},
        {"prompt_tokens", lifetime.cycle.result.prompt_tokens},
        {"prefill_launches", lifetime.cycle.result.prefill_launches},
        {"decode_launches", lifetime.cycle.result.decode_launches},
        {"final_kv_position", lifetime.cycle.result.final_kv_position},
        {"selected_token_ids", lifetime.cycle.result.selected_token_ids},
        {"step_top1_token_ids", step_top1_token_ids(lifetime.cycle.result)},
        {"before_load", sample_json(lifetime.before_load)},
        {"after_requests", sample_json(lifetime.cycle.after_requests)},
        {"after_unload", sample_json(lifetime.after_unload)},
        {"process_growth_bytes", process_growth},
        {"device_wide_growth_bytes", static_cast<std::int64_t>(after_requests_device_used) -
                                         static_cast<std::int64_t>(before_device_used)},
        {"retained_bytes", retained_process_bytes(lifetime.before_load, lifetime.after_unload)},
        {"device_wide_retained_bytes",
         retained_device_wide_bytes(lifetime.before_load, lifetime.after_unload)},
    };
    trtmc::qualification::attach_runtime_phase_memory_samples(
        out, lifetime.runtime_phase_memory_samples);
    return out;
}

constexpr std::uint64_t kProfileSweepSecondSweepGrowthLimitBytes = 64ULL * 1024ULL * 1024ULL;

std::uint64_t device_used_bytes(const DeviceMemorySample& sample) {
    if (sample.free_bytes > sample.total_bytes)
        throw std::runtime_error("profile sweep received an invalid CUDA memory sample");
    return sample.total_bytes - sample.free_bytes;
}

json invocation_tuple_json(const trtmc::RuntimeMemoryQualificationResultV1& result) {
    auto tuples = json::array();
    for (const auto& trace : result.invocations) {
        tuples.push_back({
            {"role", trace.role},
            {"plan_id", trace.plan_id},
            {"profile_id", trace.profile_id},
        });
    }
    return tuples;
}

std::vector<std::int32_t>
observed_decode_profile_ids(const trtmc::RuntimeMemoryQualificationResultV1& result) {
    std::vector<std::int32_t> profiles;
    for (const auto& trace : result.invocations) {
        if (trace.role == "decode")
            profiles.push_back(trace.profile_id);
    }
    return profiles;
}

bool exact_decode_profile_observed(const trtmc::RuntimeMemoryQualificationResultV1& result,
                                   std::int32_t expected_profile) {
    const auto profiles = observed_decode_profile_ids(result);
    return result.decode_launches == 1 && profiles.size() == 1 &&
           profiles.front() == expected_profile;
}

json growth_from_baseline(const DeviceMemorySample& baseline, const DeviceMemorySample& sample,
                          std::uint64_t& process_high_water,
                          std::uint64_t& device_wide_high_water) {
    const auto process_growth =
        positive_growth(baseline.process_used_bytes, sample.process_used_bytes);
    const auto device_growth =
        positive_growth(device_used_bytes(baseline), device_used_bytes(sample));
    process_high_water = std::max(process_high_water, process_growth);
    device_wide_high_water = std::max(device_wide_high_water, device_growth);
    return {
        {"process_growth_bytes", process_growth},
        {"device_wide_growth_bytes", device_growth},
        {"cumulative_process_high_water_bytes", process_high_water},
        {"cumulative_device_wide_high_water_bytes", device_wide_high_water},
    };
}

struct ProfileSweepInput {
    std::string path;
    std::vector<std::int32_t> tokens;
    std::uint64_t lower_exclusive{0};
    std::uint64_t profile_limit{0};
};

std::vector<ProfileSweepInput> load_profile_sweep_inputs(const Arguments& args,
                                                         const trtmc::BundleInfo& bundle_info) {
    const auto& contract = bundle_info.runtime_memory;
    if (!contract.present || contract.active_kv_profile_limits.empty()) {
        throw std::runtime_error(
            "profile sweep requires a bundle with an active runtime_memory profile contract");
    }
    if (args.profile_sweep_token_files.size() != contract.active_kv_profile_limits.size()) {
        throw std::invalid_argument(
            "--profile-sweep-tokens count must exactly equal the bundle active profile count (" +
            std::to_string(contract.active_kv_profile_limits.size()) + ")");
    }

    std::vector<ProfileSweepInput> inputs;
    inputs.reserve(args.profile_sweep_token_files.size());
    std::uint64_t previous_limit = 0;
    for (std::size_t index = 0; index < args.profile_sweep_token_files.size(); ++index) {
        const auto profile_limit =
            static_cast<std::uint64_t>(contract.active_kv_profile_limits[index]);
        auto tokens = read_tokens(args.profile_sweep_token_files[index]);
        const auto token_count = static_cast<std::uint64_t>(tokens.size());
        if (tokens.empty() || token_count <= previous_limit || token_count > profile_limit) {
            throw std::invalid_argument(
                "profile sweep token file " + args.profile_sweep_token_files[index] +
                " must contain a prompt in (" + std::to_string(previous_limit) + ", " +
                std::to_string(profile_limit) + "] tokens");
        }
        if (args.max_sequence_length != 0 && token_count + 1U > args.max_sequence_length) {
            throw std::invalid_argument(
                "profile sweep request exceeds the explicit max sequence length");
        }
        inputs.push_back({std::filesystem::absolute(args.profile_sweep_token_files[index]).string(),
                          std::move(tokens), previous_limit, profile_limit});
        previous_limit = profile_limit;
    }
    return inputs;
}

int run_profile_sweep_mode(const Arguments& args, const trtmc::LoadOptionsV2& options) {
#if !TRTMC_HAS_NVML
    throw std::runtime_error("profile sweep requires independent NVML process-memory attribution");
#endif
    const auto bundle_info = trtmc::InspectBundle(args.bundle);
    const auto inputs = load_profile_sweep_inputs(args, bundle_info);
    const auto loading_mode = query_cuda_module_loading_mode();

    json runtime_phase_memory_samples = trtmc::qualification::make_runtime_phase_memory_samples();
    std::optional<DeviceMemorySample> after_kv_baseline;
    std::uint32_t after_kv_baseline_count = 0;
    DeviceMemorySample before_load;
    DeviceMemorySample after_sweep_a;
    DeviceMemorySample before_sweep_b;
    DeviceMemorySample after_sweep_b;
    DeviceMemorySample after_unload;
    std::string model_id;
    std::string pipeline_type;
    json sweep_a_rows = json::array();
    json sweep_b_rows = json::array();
    json reserve_rows = json::array();
    std::vector<trtmc::RuntimeMemoryQualificationResultV1> sweep_a_results;
    sweep_a_results.reserve(inputs.size());
    std::optional<trtmc::RuntimeMemoryQualificationResultV1> final_sweep_b_result;

    bool stable_allocation_id_gate = true;
    bool sweep_a_profile_coverage_gate = true;
    bool sweep_b_profile_coverage_gate = true;
    bool selected_ids_equivalent_gate = true;
    bool logits_equivalent_gate = true;
    bool invocation_tuples_equivalent_gate = true;
    std::optional<std::uint64_t> stable_allocation_id;
    std::uint64_t sweep_a_process_high_water = 0;
    std::uint64_t sweep_a_device_high_water = 0;
    std::uint64_t sweep_b_process_high_water = 0;
    std::uint64_t sweep_b_device_high_water = 0;

    const auto observe_allocation_id = [&](std::uint64_t allocation_id) {
        if (allocation_id == 0)
            stable_allocation_id_gate = false;
        if (!stable_allocation_id.has_value()) {
            stable_allocation_id = allocation_id;
        } else if (*stable_allocation_id != allocation_id) {
            stable_allocation_id_gate = false;
        }
    };

    {
        RuntimePhaseMemoryObserverScope observer(
            runtime_phase_memory_samples, [&](const char* phase, const DeviceMemorySample& sample) {
                if (std::string(phase) != "after runtime KV allocation")
                    return;
                ++after_kv_baseline_count;
                if (after_kv_baseline_count != 1) {
                    throw std::runtime_error(
                        "profile sweep observed more than one runtime KV allocation phase");
                }
                after_kv_baseline = sample;
            });

        before_load = sample_device_memory();
        auto pipeline = trtmc::load(args.bundle, options);
        if (!after_kv_baseline.has_value() || after_kv_baseline_count != 1) {
            throw std::runtime_error(
                "profile sweep did not observe the exact after-runtime-KV-allocation baseline");
        }
        auto* qualifier = dynamic_cast<trtmc::IRuntimeMemoryQualificationV1*>(pipeline.get());
        if (qualifier == nullptr || qualifier->runtime_memory_qualification_api_version() !=
                                        trtmc::kRuntimeMemoryQualificationApiVersionV1) {
            throw std::runtime_error(
                "profile sweep requires runtime-memory qualification interface V1");
        }
        model_id = pipeline->model_id();
        pipeline_type = pipeline->pipeline_type();

        for (std::size_t index = 0; index < inputs.size(); ++index) {
            trtmc::RuntimeMemoryQualificationRequestV1 request;
            request.input_ids = inputs[index].tokens;
            request.max_new_tokens = 1;
            auto result = qualifier->qualify_runtime_memory(request);
            const auto sample = sample_device_memory();
            const auto receipt = parse_receipt(result.runtime_memory_receipt_json);
            const auto allocation_id = receipt.at("kv_allocation_id").get<std::uint64_t>();
            observe_allocation_id(allocation_id);
            const auto expected_profile = static_cast<std::int32_t>(index);
            const bool profile_match = exact_decode_profile_observed(result, expected_profile);
            sweep_a_profile_coverage_gate = sweep_a_profile_coverage_gate && profile_match;
            auto growth = growth_from_baseline(
                *after_kv_baseline, sample, sweep_a_process_high_water, sweep_a_device_high_water);
            reserve_rows.push_back({
                {"profile_id", expected_profile},
                {"covering_profile_limit", inputs[index].profile_limit},
                {"cumulative_process_first_use_bytes",
                 growth.at("cumulative_process_high_water_bytes")},
                {"cumulative_device_wide_first_use_bytes",
                 growth.at("cumulative_device_wide_high_water_bytes")},
            });
            sweep_a_rows.push_back({
                {"row_index", index},
                {"token_file", inputs[index].path},
                {"prompt_tokens", result.prompt_tokens},
                {"max_new_tokens", 1},
                {"expected_profile_id", expected_profile},
                {"expected_profile_limit", inputs[index].profile_limit},
                {"expected_prompt_lower_exclusive", inputs[index].lower_exclusive},
                {"profile_match", profile_match},
                {"observed_decode_profile_ids", observed_decode_profile_ids(result)},
                {"selected_token_ids", result.selected_token_ids},
                {"step_top1_token_ids", step_top1_token_ids(result)},
                {"kv_allocation_id", allocation_id},
                {"runtime_memory_receipt", receipt},
                {"invocation_tuples", invocation_tuple_json(result)},
                {"invocations", invocation_json(result)},
                {"after_request", sample_json(sample)},
                {"cumulative_first_use_growth", std::move(growth)},
            });
            after_sweep_a = sample;
            sweep_a_results.push_back(std::move(result));
        }

        before_sweep_b = sample_device_memory();
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            trtmc::RuntimeMemoryQualificationRequestV1 request;
            request.input_ids = inputs[index].tokens;
            request.max_new_tokens = 1;
            auto result = qualifier->qualify_runtime_memory(request);
            const auto sample = sample_device_memory();
            const auto receipt = parse_receipt(result.runtime_memory_receipt_json);
            const auto allocation_id = receipt.at("kv_allocation_id").get<std::uint64_t>();
            observe_allocation_id(allocation_id);

            const auto expected_profile = static_cast<std::int32_t>(index);
            const bool profile_match = exact_decode_profile_observed(result, expected_profile);
            sweep_b_profile_coverage_gate = sweep_b_profile_coverage_gate && profile_match;
            const auto& reference = sweep_a_results[index];
            const bool selected_ids_equal =
                reference.selected_token_ids == result.selected_token_ids;
            const bool logits_equal = trtmc::qualification::float32_logits_bitwise_equal(
                reference.step_logits, result.step_logits);
            const bool invocation_tuples_equal =
                invocation_tuple_json(reference) == invocation_tuple_json(result);
            selected_ids_equivalent_gate = selected_ids_equivalent_gate && selected_ids_equal;
            logits_equivalent_gate = logits_equivalent_gate && logits_equal;
            invocation_tuples_equivalent_gate =
                invocation_tuples_equivalent_gate && invocation_tuples_equal;
            auto growth = growth_from_baseline(before_sweep_b, sample, sweep_b_process_high_water,
                                               sweep_b_device_high_water);
            sweep_b_rows.push_back({
                {"row_index", index},
                {"token_file", inputs[index].path},
                {"prompt_tokens", result.prompt_tokens},
                {"max_new_tokens", 1},
                {"expected_profile_id", expected_profile},
                {"expected_profile_limit", inputs[index].profile_limit},
                {"expected_prompt_lower_exclusive", inputs[index].lower_exclusive},
                {"profile_match", profile_match},
                {"observed_decode_profile_ids", observed_decode_profile_ids(result)},
                {"selected_token_ids", result.selected_token_ids},
                {"step_top1_token_ids", step_top1_token_ids(result)},
                {"kv_allocation_id", allocation_id},
                {"runtime_memory_receipt", receipt},
                {"invocation_tuples", invocation_tuple_json(result)},
                {"invocations", invocation_json(result)},
                {"after_request", sample_json(sample)},
                {"incremental_growth_from_sweep_b_baseline", std::move(growth)},
                {"equivalence_to_sweep_a",
                 {
                     {"selected_token_ids_bitwise_equal", selected_ids_equal},
                     {"complete_float32_logits_bitwise_equal", logits_equal},
                     {"invocation_tuples_equal", invocation_tuples_equal},
                     {"passed", selected_ids_equal && logits_equal && invocation_tuples_equal},
                 }},
            });
            after_sweep_b = sample;
            if (index + 1 == inputs.size())
                final_sweep_b_result = std::move(result);
        }

        pipeline.reset();
        after_unload = sample_device_memory();
    }

    if (!final_sweep_b_result.has_value())
        throw std::logic_error("profile sweep produced no Sweep B result");
    write_logits(args.logits_file, *final_sweep_b_result);

    const bool sweep_b_incremental_growth_gate =
        sweep_b_process_high_water <= kProfileSweepSecondSweepGrowthLimitBytes;
    const bool passed = stable_allocation_id_gate && sweep_a_profile_coverage_gate &&
                        sweep_b_profile_coverage_gate && selected_ids_equivalent_gate &&
                        logits_equivalent_gate && invocation_tuples_equivalent_gate &&
                        sweep_b_incremental_growth_gate;
    json blockers = json::array();
    const auto add_blocker = [&](bool gate, const char* name) {
        if (!gate)
            blockers.push_back(name);
    };
    add_blocker(stable_allocation_id_gate, "stable_kv_allocation_id");
    add_blocker(sweep_a_profile_coverage_gate, "sweep_a_exact_profile_coverage");
    add_blocker(sweep_b_profile_coverage_gate, "sweep_b_exact_profile_coverage");
    add_blocker(selected_ids_equivalent_gate, "selected_token_ids_bitwise_equivalent");
    add_blocker(logits_equivalent_gate, "complete_float32_logits_bitwise_equivalent");
    add_blocker(invocation_tuples_equivalent_gate, "invocation_tuples_equivalent");
    add_blocker(sweep_b_incremental_growth_gate,
                "sweep_b_process_incremental_high_water_within_limit");

    json input_manifest = json::array();
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        input_manifest.push_back({
            {"row_index", index},
            {"token_file", inputs[index].path},
            {"prompt_tokens", inputs[index].tokens.size()},
            {"expected_profile_id", index},
            {"expected_profile_limit", inputs[index].profile_limit},
            {"expected_prompt_lower_exclusive", inputs[index].lower_exclusive},
        });
    }

    json output{
        {"schema_version", 1},
        {"mode", "all_profile_two_sweep"},
        {"status", passed ? "ok" : "error"},
        {"error_type", passed ? json(nullptr) : json("qualification_gate")},
        {"passed", passed},
        {"qualification_blockers", std::move(blockers)},
        {"qualification_api_version", trtmc::kRuntimeMemoryQualificationApiVersionV1},
        {"model_id", model_id},
        {"pipeline_type", pipeline_type},
        {"cuda_module_loading",
         {
             {"source", "cuModuleGetLoadingMode"},
             {"mode", loading_mode.mode},
             {"driver_value", loading_mode.driver_value},
         }},
        {"protocol",
         {
             {"schema_version", 1},
             {"execution_order", {"sweep_a", "sweep_b"}},
             {"pipeline_load_count", 1},
             {"kv_allocation_count", 1},
             {"max_new_tokens_per_request", 1},
             {"profile_order", "ascending"},
             {"second_sweep_process_growth_limit_bytes", kProfileSweepSecondSweepGrowthLimitBytes},
         }},
        {"input_manifest", std::move(input_manifest)},
        {"stable_kv_allocation_id",
         stable_allocation_id.has_value() ? json(*stable_allocation_id) : json(nullptr)},
        {"memory",
         {
             {"before_load", sample_json(before_load)},
             {"after_runtime_kv_allocation", sample_json(*after_kv_baseline)},
             {"after_sweep_a", sample_json(after_sweep_a)},
             {"before_sweep_b", sample_json(before_sweep_b)},
             {"after_sweep_b", sample_json(after_sweep_b)},
             {"after_unload", sample_json(after_unload)},
             {"retained_process_bytes", retained_process_bytes(before_load, after_unload)},
             {"retained_device_wide_bytes", retained_device_wide_bytes(before_load, after_unload)},
         }},
        {"profile_reserve_rows", std::move(reserve_rows)},
        {"sweep_a",
         {
             {"rows", std::move(sweep_a_rows)},
             {"cumulative_process_first_use_high_water_bytes", sweep_a_process_high_water},
             {"cumulative_device_wide_first_use_high_water_bytes", sweep_a_device_high_water},
         }},
        {"sweep_b",
         {
             {"rows", std::move(sweep_b_rows)},
             {"incremental_process_high_water_bytes", sweep_b_process_high_water},
             {"incremental_device_wide_high_water_bytes", sweep_b_device_high_water},
             {"process_growth_limit_bytes", kProfileSweepSecondSweepGrowthLimitBytes},
             {"process_growth_within_limit", sweep_b_incremental_growth_gate},
         }},
        {"gates",
         {
             {"stable_kv_allocation_id", stable_allocation_id_gate},
             {"sweep_a_exact_profile_coverage", sweep_a_profile_coverage_gate},
             {"sweep_b_exact_profile_coverage", sweep_b_profile_coverage_gate},
             {"selected_token_ids_bitwise_equivalent", selected_ids_equivalent_gate},
             {"complete_float32_logits_bitwise_equivalent", logits_equivalent_gate},
             {"invocation_tuples_equivalent", invocation_tuples_equivalent_gate},
             {"sweep_b_process_incremental_high_water_within_limit",
              sweep_b_incremental_growth_gate},
         }},
        {"runtime_phase_memory_samples", std::move(runtime_phase_memory_samples)},
        {"logits_artifact", logits_artifact_json(args.logits_file, *final_sweep_b_result)},
    };
#if TRTMC_HAS_NVML
    output["memory_sampler"] = process_memory_sampler().metadata();
#endif
    std::cout << output.dump() << '\n';
    return passed ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_arguments(argc, argv);
        if (args.query_product_identity) {
            (void)kProductIdentityMarker;
            std::cout << trtmc::qualification::make_product_identity_evidence(
                             TRTMC_INTERNAL_PRODUCT_VERSION,
                             TRTMC_INTERNAL_CALIBRATOR_BUILD_IDENTITY)
                             .dump()
                      << '\n';
            return 0;
        }
        if (args.query_module_loading_mode) {
            const auto loading_mode = query_cuda_module_loading_mode();
            std::cout << trtmc::qualification::make_cuda_module_loading_mode_evidence(
                             loading_mode.mode, loading_mode.driver_value)
                             .dump()
                      << '\n';
            return 0;
        }
        // The final product runtime rejects external-manifest evidence. This
        // product-owned helper alone may load the ephemeral one-byte-reserve
        // bootstrap used to measure and create embedded calibration evidence.
        trtmc::InternalRuntimeMemoryCalibrationBootstrapScope bootstrap_scope;
        trtmc::LoadOptionsV2 options;
        append_default_search_paths(argv[0], options);
        options.backend_search_paths.insert(options.backend_search_paths.end(),
                                            args.backend_dirs.begin(), args.backend_dirs.end());
        options.model_plugin_search_paths.insert(options.model_plugin_search_paths.end(),
                                                 args.model_plugin_dirs.begin(),
                                                 args.model_plugin_dirs.end());
        options.max_sequence_length = args.max_sequence_length;
        options.max_sequence_length_explicit = args.max_sequence_length != 0 ? 1U : 0U;
        if (args.kv_cache_bytes != 0) {
            options.kv_cache_memory_policy = trtmc::KvCacheMemoryPolicy::kBytes;
            options.kv_cache_memory_bytes = args.kv_cache_bytes;
        } else if (args.kv_cache_fraction != 0.0) {
            options.kv_cache_memory_policy = trtmc::KvCacheMemoryPolicy::kFraction;
            options.kv_cache_memory_fraction = args.kv_cache_fraction;
        } else if (args.max_sequence_length != 0) {
            options.kv_cache_memory_policy = trtmc::KvCacheMemoryPolicy::kAuto;
        }

        if (!args.profile_sweep_token_files.empty())
            return run_profile_sweep_mode(args, options);

        trtmc::RuntimeMemoryQualificationRequestV1 request;
        request.input_ids = read_tokens(args.token_file);
        request.max_new_tokens = args.max_new_tokens;
        if (args.controlled_reservation_target_tokens != 0) {
            // The first TensorRT lifetime can retain process-global lazy
            // initialization state.  Calibrate the exact target allocation
            // after that warm-up, then run both auto lifetimes from the same
            // warmed process boundary.
            const auto warmup = run_lifetime(args, options, request);
            auto calibration_options = options;
            calibration_options.max_sequence_length = args.controlled_reservation_target_tokens;
            calibration_options.max_sequence_length_explicit = 1U;
            calibration_options.kv_cache_memory_policy = trtmc::KvCacheMemoryPolicy::kAuto;
            const auto calibration = run_lifetime(args, calibration_options, request);
            const auto baseline = run_lifetime(args, options, request);
            const auto calibration_receipt =
                parse_receipt(calibration.cycle.result.runtime_memory_receipt_json);
            const auto baseline_receipt =
                parse_receipt(baseline.cycle.result.runtime_memory_receipt_json);
            const auto calibration_r =
                calibration_receipt.at("runtime_kv_capacity_tokens").get<std::uint64_t>();
            const auto baseline_r =
                baseline_receipt.at("runtime_kv_capacity_tokens").get<std::uint64_t>();
            const auto bytes_per_token =
                baseline_receipt.at("kv_bytes_per_token").get<std::uint64_t>();
            const auto baseline_kv_bytes =
                baseline_receipt.at("kv_reserved_bytes").get<std::uint64_t>();
            const auto calibration_bytes_per_token =
                calibration_receipt.at("kv_bytes_per_token").get<std::uint64_t>();
            const auto calibration_kv_bytes =
                calibration_receipt.at("kv_reserved_bytes").get<std::uint64_t>();
            if (calibration_r != args.controlled_reservation_target_tokens) {
                throw std::runtime_error(
                    "controlled reservation target calibration did not allocate exact R");
            }
            if (calibration_bytes_per_token != bytes_per_token ||
                calibration_r > std::numeric_limits<std::uint64_t>::max() / bytes_per_token ||
                calibration_kv_bytes != calibration_r * bytes_per_token) {
                throw std::runtime_error(
                    "controlled reservation target calibration has invalid KV accounting");
            }
            if (args.controlled_reservation_target_tokens >= baseline_r) {
                throw std::invalid_argument(
                    "controlled reservation target must be below auto baseline R");
            }
            const auto request_tokens = static_cast<std::uint64_t>(request.input_ids.size()) +
                                        static_cast<std::uint64_t>(request.max_new_tokens);
            if (args.controlled_reservation_target_tokens < request_tokens) {
                throw std::invalid_argument("controlled reservation target cannot fit the request");
            }
            if (args.controlled_reservation_target_tokens >
                std::numeric_limits<std::uint64_t>::max() / bytes_per_token) {
                throw std::overflow_error("controlled reservation target KV bytes overflow");
            }

            constexpr const char* kBeforePlanningPhase = "before runtime KV planning";
            constexpr const char* kAfterOverheadPhase =
                "after shared context and output allocation";
            constexpr const char* kAfterKvPhase = "after runtime KV allocation";
            constexpr const char* kAfterRequestPhase =
                "after successful runtime-memory request completion";
            const auto calibration_before_planning =
                require_phase_sample(calibration, kBeforePlanningPhase);
            const auto calibration_after_overhead =
                require_phase_sample(calibration, kAfterOverheadPhase);
            const auto calibration_after_kv = require_phase_sample(calibration, kAfterKvPhase);
            const auto calibration_after_request =
                require_phase_sample(calibration, kAfterRequestPhase);
            if (calibration_before_planning.free_bytes < calibration_after_overhead.free_bytes) {
                throw std::runtime_error(
                    "target calibration context/output phase increased free memory");
            }
            const auto measured_context_output_bytes =
                calibration_before_planning.free_bytes - calibration_after_overhead.free_bytes;
            const auto request_completion_device_bytes = positive_growth(
                calibration_after_kv.total_bytes - calibration_after_kv.free_bytes,
                calibration_after_request.total_bytes - calibration_after_request.free_bytes);
            const auto request_completion_process_bytes =
                positive_growth(calibration_after_kv.process_used_bytes,
                                calibration_after_request.process_used_bytes);
            const auto request_completion_headroom_bytes =
                std::max(request_completion_device_bytes, request_completion_process_bytes);
            const auto request_completion_external_delta_bytes =
                static_cast<std::int64_t>(request_completion_device_bytes) -
                static_cast<std::int64_t>(request_completion_process_bytes);

            const auto baseline_process_growth =
                static_cast<std::uint64_t>(baseline.cycle.after_requests.process_used_bytes -
                                           baseline.before_load.process_used_bytes);
            const auto non_kv_growth = baseline_process_growth > baseline_kv_bytes
                                           ? baseline_process_growth - baseline_kv_bytes
                                           : baseline_process_growth;
            const auto baseline_pre_load_free_bytes =
                baseline_receipt.at("pre_load_free_bytes").get<std::uint64_t>();
            const auto baseline_post_load_free_bytes =
                baseline_receipt.at("post_load_free_bytes").get<std::uint64_t>();
            const auto baseline_engine_load_device_bytes =
                baseline_pre_load_free_bytes > baseline_post_load_free_bytes
                    ? baseline_pre_load_free_bytes - baseline_post_load_free_bytes
                    : 0;
            const auto safety_reserve_bytes =
                baseline_receipt.at("safety_reserve_bytes").get<std::uint64_t>();
            const auto calibration_safety_reserve_bytes =
                calibration_receipt.at("safety_reserve_bytes").get<std::uint64_t>();
            if (calibration_safety_reserve_bytes != safety_reserve_bytes) {
                throw std::runtime_error("target calibration changed the runtime safety reserve");
            }
            const auto auto_fraction = baseline_receipt.at("policy_fraction").get<double>();
            const auto calibration_auto_fraction =
                calibration_receipt.at("policy_fraction").get<double>();
            if (baseline_receipt.at("policy") != "auto" ||
                calibration_receipt.at("policy") != "auto" ||
                std::abs(auto_fraction - 0.9) > 1e-12 ||
                std::abs(calibration_auto_fraction - auto_fraction) > 1e-12) {
                throw std::runtime_error(
                    "controlled reservation requires measured auto fraction 0.9");
            }
            const auto policy_safe_bytes =
                ceil_fraction_denominator(calibration_kv_bytes, auto_fraction);
            const auto policy_fraction_headroom_bytes = policy_safe_bytes - calibration_kv_bytes;
            constexpr std::uint64_t kReservationAlignment =
                trtmc::qualification::kControlledReservationAlignmentBytes;
            const auto calibration_context_bytes =
                calibration_receipt.at("context_device_memory_bytes").get<std::uint64_t>();
            const auto calibration_ordinary_input_bytes =
                calibration_receipt.at("ordinary_device_input_bytes").get<std::uint64_t>();
            const auto calibration_ordinary_output_bytes =
                calibration_receipt.at("ordinary_device_output_bytes").get<std::uint64_t>();
            const auto calibration_external_output_bytes =
                calibration_receipt.at("external_device_output_bytes").get<std::uint64_t>();
            const auto calibration_graph_bytes =
                calibration_receipt.at("graph_private_device_bytes").get<std::uint64_t>();
            auto logical_context_output_bytes =
                checked_add(calibration_context_bytes, calibration_ordinary_input_bytes,
                            "controlled logical context/output bytes");
            logical_context_output_bytes =
                checked_add(logical_context_output_bytes, calibration_ordinary_output_bytes,
                            "controlled logical context/output bytes");
            logical_context_output_bytes =
                checked_add(logical_context_output_bytes, calibration_external_output_bytes,
                            "controlled logical context/output bytes");
            logical_context_output_bytes =
                checked_add(logical_context_output_bytes, calibration_graph_bytes,
                            "controlled logical context/output bytes");
            const auto final_free_lower_bound_bytes =
                checked_add(safety_reserve_bytes, policy_safe_bytes,
                            "controlled final visible-free lower bound");
            const auto final_free_upper_bound_bytes =
                checked_add(final_free_lower_bound_bytes, kReservationAlignment,
                            "controlled final visible-free upper bound");
            constexpr auto kPreplanningHeadroomBytes =
                trtmc::qualification::kControlledPreplanningHeadroomBytes;
            const auto required_visible_post_load_free_bytes =
                checked_add(logical_context_output_bytes,
                            checked_add(final_free_upper_bound_bytes, kPreplanningHeadroomBytes,
                                        "controlled visible post-load bytes"),
                            "controlled visible post-load bytes");
            const auto guard_bytes =
                align_up(checked_add(calibration_kv_bytes, request_completion_headroom_bytes,
                                     "controlled contiguous guard"),
                         kReservationAlignment);
            constexpr auto max_capacity_rounding_rows =
                trtmc::qualification::kControlledTargetToleranceRows;

            DeviceMemorySample before_reservation;
            DeviceMemorySample after_guard_allocation;
            DeviceMemorySample after_reservation;
            DeviceMemorySample before_guard_release;
            DeviceMemorySample after_guard_release;
            DeviceMemorySample final_feedback_sample;
            std::uint64_t bulk_reservation_bytes = 0;
            std::uint64_t initial_bulk_reservation_bytes = 0;
            std::uint64_t bulk_correction_allocated_bytes = 0;
            std::uint64_t bulk_correction_released_bytes = 0;
            std::uint64_t bulk_correction_attempts = 0;
            json bulk_correction_evidence = json::array();
            std::uint64_t guard_address = 0;
            std::size_t guard_allocation_count = 0;
            json guard_allocations = json::array();
            std::unique_ptr<DeviceReservation> guard;
            std::unique_ptr<DeviceReservation> bulk_reservation;
            bool reservation_action_invoked = false;
            bool final_feedback_action_invoked = false;
            bool final_feedback_converged = false;
            bool guard_release_action_invoked = false;
            auto correction_diagnostic = [&](const char* reason,
                                             const DeviceMemorySample& current_sample) {
                return json{
                    {"mode", "controlled_final_free_feedback"},
                    {"reason", reason},
                    {"reservation_alignment_bytes", kReservationAlignment},
                    {"max_correction_attempts",
                     trtmc::qualification::kMaxControlledBulkCorrectionAttempts},
                    {"final_free_lower_bound_bytes", final_free_lower_bound_bytes},
                    {"final_free_upper_bound_bytes", final_free_upper_bound_bytes},
                    {"required_visible_post_load_free_bytes",
                     required_visible_post_load_free_bytes},
                    {"initial_bulk_reservation_bytes", initial_bulk_reservation_bytes},
                    {"bulk_reservation_bytes", bulk_reservation_bytes},
                    {"bulk_correction_allocated_bytes", bulk_correction_allocated_bytes},
                    {"bulk_correction_released_bytes", bulk_correction_released_bytes},
                    {"bulk_correction_attempts", bulk_correction_attempts},
                    {"current_sample", sample_json(current_sample)},
                    {"corrections", bulk_correction_evidence},
                };
            };
            QualificationLifetime constrained;
            {
                RuntimePreSnapshotActionScope pre_snapshot_action([&](const char* phase) {
                    const std::string phase_name = phase;
                    if (phase_name == kBeforePlanningPhase) {
                        if (reservation_action_invoked) {
                            throw std::logic_error(
                                "controlled post-load reservation action ran more than once");
                        }
                        before_reservation = sample_device_memory();
                        const auto required_before_bulk =
                            checked_add(guard_bytes,
                                        checked_add(required_visible_post_load_free_bytes,
                                                    kReservationAlignment,
                                                    "controlled reservation required bytes"),
                                        "controlled reservation required bytes");
                        if (before_reservation.free_bytes <= required_before_bulk) {
                            throw std::runtime_error("insufficient post-load free memory for "
                                                     "controlled reservation");
                        }
                        guard = std::make_unique<DeviceReservation>(guard_bytes, guard_bytes);
                        guard_address = guard->address();
                        guard_allocation_count = guard->allocation_count();
                        guard_allocations = guard->allocations_json();
                        after_guard_allocation = sample_device_memory();
                        if (after_guard_allocation.free_bytes <=
                            required_visible_post_load_free_bytes + kReservationAlignment) {
                            throw std::runtime_error(
                                "contiguous guard leaves insufficient memory for bulk pressure");
                        }
                        initial_bulk_reservation_bytes = ((after_guard_allocation.free_bytes -
                                                           required_visible_post_load_free_bytes) /
                                                          kReservationAlignment) *
                                                         kReservationAlignment;
                        if (initial_bulk_reservation_bytes == 0) {
                            throw std::runtime_error(
                                "controlled bulk reservation resolved to zero bytes");
                        }
                        bulk_reservation = std::make_unique<DeviceReservation>(
                            initial_bulk_reservation_bytes,
                            trtmc::qualification::kControlledInitialBulkChunkBytes);
                        bulk_reservation_bytes = bulk_reservation->bytes();
                        after_reservation = sample_device_memory();
                        const auto minimum_initial_free =
                            checked_add(logical_context_output_bytes, final_free_upper_bound_bytes,
                                        "controlled minimum initial visible free bytes");
                        if (after_reservation.free_bytes < minimum_initial_free) {
                            throw QualificationDiagnosticError(
                                "controlled preplanning reservation left insufficient target "
                                "headroom",
                                correction_diagnostic("preplanning_headroom_exhausted",
                                                      after_reservation));
                        }
                        reservation_action_invoked = true;
                        return;
                    }
                    if (phase_name != kAfterOverheadPhase)
                        return;
                    if (!reservation_action_invoked || !bulk_reservation || !guard) {
                        throw std::logic_error(
                            "controlled final feedback ran before preplanning reservation");
                    }
                    if (final_feedback_action_invoked) {
                        throw std::logic_error(
                            "controlled final feedback action ran more than once");
                    }

                    auto current = sample_device_memory();
                    std::set<std::pair<std::uint64_t, std::uint64_t>> visited_states;
                    while (true) {
                        const auto action =
                            trtmc::qualification::decide_controlled_free_window_action(
                                current.free_bytes, final_free_lower_bound_bytes,
                                final_free_upper_bound_bytes, kReservationAlignment,
                                bulk_reservation->tail_bytes());
                        if (action.kind ==
                            trtmc::qualification::ControlledFreeWindowActionKind::kInWindow) {
                            final_feedback_sample = current;
                            final_feedback_converged = true;
                            break;
                        }
                        if (bulk_correction_attempts >=
                            trtmc::qualification::kMaxControlledBulkCorrectionAttempts) {
                            throw QualificationDiagnosticError(
                                "controlled final feedback did not converge within its finite "
                                "attempt limit",
                                correction_diagnostic("correction_attempt_limit", current));
                        }
                        const auto state =
                            std::make_pair(current.free_bytes, bulk_reservation->bytes());
                        if (!visited_states.insert(state).second) {
                            throw QualificationDiagnosticError(
                                "controlled final feedback repeated a prior state",
                                correction_diagnostic("repeated_feedback_state", current));
                        }

                        const auto direction =
                            action.kind ==
                                    trtmc::qualification::ControlledFreeWindowActionKind::kAllocate
                                ? "allocate"
                                : "release";
                        const auto reserved_before = bulk_reservation->bytes();
                        const auto allocated_bytes =
                            action.kind ==
                                    trtmc::qualification::ControlledFreeWindowActionKind::kAllocate
                                ? action.bytes
                                : 0;
                        const auto released_bytes =
                            action.kind ==
                                    trtmc::qualification::ControlledFreeWindowActionKind::kRelease
                                ? action.bytes
                                : 0;
                        const auto reserved_after =
                            action.kind ==
                                    trtmc::qualification::ControlledFreeWindowActionKind::kAllocate
                                ? checked_add(reserved_before, action.bytes,
                                              "controlled final-feedback reservation")
                                : reserved_before - action.bytes;
                        const auto attempt_index = bulk_correction_attempts++;
                        bulk_correction_evidence.push_back({
                            {"attempt_index", attempt_index},
                            {"direction", direction},
                            {"before", sample_json(current)},
                            {"after", nullptr},
                            {"deficit_bytes", action.deficit_bytes},
                            {"excess_bytes", action.excess_bytes},
                            {"allocated_bytes", allocated_bytes},
                            {"released_bytes", released_bytes},
                            {"cumulative_reserved_bytes_before", reserved_before},
                            {"cumulative_reserved_bytes_after", reserved_after},
                            {"status", "applying"},
                        });
                        try {
                            if (action.kind ==
                                trtmc::qualification::ControlledFreeWindowActionKind::kAllocate) {
                                bulk_reservation->reserve_more(action.bytes, kReservationAlignment);
                                bulk_correction_allocated_bytes =
                                    checked_add(bulk_correction_allocated_bytes, action.bytes,
                                                "controlled final-feedback allocated bytes");
                            } else {
                                const auto actual_released = bulk_reservation->release_last();
                                if (actual_released != action.bytes) {
                                    throw std::logic_error(
                                        "controlled final-feedback tail release changed size");
                                }
                                bulk_correction_released_bytes =
                                    checked_add(bulk_correction_released_bytes, actual_released,
                                                "controlled final-feedback released bytes");
                            }
                            bulk_reservation_bytes = bulk_reservation->bytes();
                            if (bulk_reservation_bytes != reserved_after) {
                                throw std::logic_error(
                                    "controlled final-feedback byte ledger mismatch");
                            }
                            const auto after_action = sample_device_memory();
                            auto& attempt = bulk_correction_evidence.back();
                            attempt["after"] = sample_json(after_action);
                            attempt["status"] = "completed";
                            if (after_action.free_bytes == current.free_bytes) {
                                throw QualificationDiagnosticError(
                                    "controlled final feedback made no visible-free progress",
                                    correction_diagnostic("no_visible_free_progress",
                                                          after_action));
                            }
                            current = after_action;
                        } catch (const QualificationDiagnosticError&) {
                            throw;
                        } catch (const std::exception& error) {
                            auto& attempt = bulk_correction_evidence.back();
                            attempt["status"] = "failed";
                            attempt["failure"] = error.what();
                            throw QualificationDiagnosticError(
                                "controlled final-feedback allocation, release, or sampling "
                                "failed",
                                correction_diagnostic("correction_action_failed", current));
                        }
                    }
                    final_feedback_action_invoked = true;
                });
                constrained = run_lifetime(
                    args, options, request,
                    [&](const char* phase, const DeviceMemorySample& sample) {
                        if (std::string(phase) != kAfterOverheadPhase)
                            return;
                        if (guard_release_action_invoked) {
                            throw std::logic_error(
                                "controlled contiguous guard release ran more than once");
                        }
                        if (!guard || guard->allocation_count() != 1) {
                            throw std::logic_error(
                                "controlled contiguous guard was not held through final snapshot");
                        }
                        if (!final_feedback_action_invoked || !final_feedback_converged) {
                            throw std::logic_error(
                                "controlled final snapshot preceded converged feedback");
                        }
                        before_guard_release = sample;
                        if (sample.free_bytes < final_free_lower_bound_bytes ||
                            sample.free_bytes >= final_free_upper_bound_bytes) {
                            throw QualificationDiagnosticError(
                                "actual controlled final snapshot left the exact target window",
                                correction_diagnostic("actual_final_snapshot_outside_window",
                                                      sample));
                        }
                        guard->release();
                        after_guard_release = sample_device_memory();
                        guard_release_action_invoked = true;
                    });
            }
            if (!reservation_action_invoked || !bulk_reservation) {
                throw std::logic_error("controlled post-load reservation action did not run");
            }
            if (!final_feedback_action_invoked || !final_feedback_converged) {
                throw std::logic_error("controlled final feedback did not converge");
            }
            if (!guard_release_action_invoked || !guard || guard->allocation_count() != 0) {
                throw std::logic_error(
                    "controlled contiguous guard was not released after final snapshot");
            }
            const auto bulk_reservation_address = bulk_reservation->address();
            const auto bulk_reservation_allocation_count = bulk_reservation->allocation_count();
            const auto bulk_reservation_allocations = bulk_reservation->allocations_json();
            const auto constrained_receipt =
                parse_receipt(constrained.cycle.result.runtime_memory_receipt_json);
            const auto constrained_r =
                constrained_receipt.at("runtime_kv_capacity_tokens").get<std::uint64_t>();
            const auto constrained_kv_bytes =
                constrained_receipt.at("kv_reserved_bytes").get<std::uint64_t>();
            const auto controlled_policy_is_auto = baseline_receipt.at("policy") == "auto" &&
                                                   calibration_receipt.at("policy") == "auto" &&
                                                   constrained_receipt.at("policy") == "auto";
            const auto controlled_reduced_r = constrained_r < baseline_r;
            const auto controlled_target_bounded =
                constrained_r >= calibration_r &&
                constrained_r - calibration_r <= max_capacity_rounding_rows;
            const auto controlled_kv_is_exact =
                constrained_r <= std::numeric_limits<std::uint64_t>::max() / bytes_per_token &&
                constrained_kv_bytes == constrained_r * bytes_per_token;
            const auto guard_covers_calibrated_target =
                guard_bytes >= checked_add(calibration_kv_bytes, request_completion_headroom_bytes,
                                           "controlled guard coverage");
            const auto controlled_request_fits =
                constrained.cycle.result.final_kv_position <= constrained_r;
            const auto receipt_final_free_bytes =
                constrained_receipt.at("final_free_bytes").get<std::uint64_t>();
            const auto controlled_final_snapshot_bounded =
                before_guard_release.free_bytes >= final_free_lower_bound_bytes &&
                before_guard_release.free_bytes < final_free_upper_bound_bytes;
            const auto controlled_receipt_binds_final_snapshot =
                receipt_final_free_bytes == before_guard_release.free_bytes;
            const auto expected_final_r =
                std::min(baseline_r, trtmc::qualification::controlled_auto_capacity_from_final_free(
                                         receipt_final_free_bytes, safety_reserve_bytes,
                                         auto_fraction, bytes_per_token));
            const auto controlled_final_formula_exact = constrained_r == expected_final_r;
            const auto controlled_passed =
                controlled_policy_is_auto && controlled_reduced_r && controlled_target_bounded &&
                controlled_kv_is_exact && guard_covers_calibrated_target &&
                controlled_request_fits && controlled_final_snapshot_bounded &&
                controlled_receipt_binds_final_snapshot && controlled_final_formula_exact;
            std::string controlled_failure;
            if (!controlled_policy_is_auto) {
                controlled_failure = "controlled reservation did not use auto policy";
            } else if (!controlled_reduced_r) {
                controlled_failure = "controlled reservation did not reduce runtime R";
            } else if (!controlled_target_bounded) {
                controlled_failure =
                    "controlled reservation resolved outside target alignment tolerance";
            } else if (!controlled_kv_is_exact) {
                controlled_failure = "controlled reservation KV bytes do not equal R times B";
            } else if (!guard_covers_calibrated_target) {
                controlled_failure =
                    "controlled contiguous guard does not cover calibrated target and request";
            } else if (!controlled_request_fits) {
                controlled_failure = "controlled reservation request did not fit R";
            } else if (!controlled_final_snapshot_bounded) {
                controlled_failure = "controlled final snapshot missed the exact 2MiB window";
            } else if (!controlled_receipt_binds_final_snapshot) {
                controlled_failure = "controlled receipt did not bind the actual final snapshot";
            } else if (!controlled_final_formula_exact) {
                controlled_failure =
                    "controlled R did not match the runtime formula at the final snapshot";
            }
            const auto after_constrained_unload = constrained.after_unload;
            bulk_reservation.reset();
            const auto after_release = sample_device_memory();

            auto auto_lifetime_json = [](const QualificationLifetime& lifetime, const char* label,
                                         bool measured) {
                auto out = lifetime_json(lifetime, trtmc::qualification::make_auto_policy(), label,
                                         measured);
                out["invocations"] = invocation_json(lifetime.cycle.result);
                return out;
            };
            auto calibration_json =
                lifetime_json(calibration,
                              trtmc::qualification::make_max_sequence_length_policy(
                                  args.controlled_reservation_target_tokens),
                              "measured-explicit-target-calibration", true);
            calibration_json["invocations"] = invocation_json(calibration.cycle.result);
            write_logits(args.logits_file, constrained.cycle.result);
            auto output = success_json(constrained.cycle.model_id, constrained.cycle.pipeline_type,
                                       constrained.cycle.result, args.logits_file);
            output["mode"] = "same_process_controlled_external_reservation";
            output["controlled_reservation"] = {
                {"target_tokens", args.controlled_reservation_target_tokens},
                {"sizing",
                 {
                     {"baseline_process_growth_bytes", baseline_process_growth},
                     {"baseline_kv_reserved_bytes", baseline_kv_bytes},
                     {"estimated_non_kv_growth_bytes", non_kv_growth},
                     {"baseline_pre_load_free_bytes", baseline_pre_load_free_bytes},
                     {"baseline_post_load_free_bytes", baseline_post_load_free_bytes},
                     {"baseline_engine_load_device_bytes", baseline_engine_load_device_bytes},
                     {"warmup_retained_process_bytes",
                      retained_process_bytes(warmup.before_load, warmup.after_unload)},
                     {"warmup_retained_device_wide_bytes",
                      retained_device_wide_bytes(warmup.before_load, warmup.after_unload)},
                     {"required_free_basis",
                      "calibration receipt logical context/output bytes plus exact final target "
                      "window and preplanning headroom"},
                     {"auto_fraction", auto_fraction},
                     {"calibration_context_device_memory_bytes",
                      calibration_receipt.at("context_device_memory_bytes")},
                     {"calibration_ordinary_device_input_bytes",
                      calibration_receipt.at("ordinary_device_input_bytes")},
                     {"calibration_ordinary_device_output_bytes",
                      calibration_receipt.at("ordinary_device_output_bytes")},
                     {"calibration_external_device_output_bytes",
                      calibration_receipt.at("external_device_output_bytes")},
                     {"calibration_graph_private_device_bytes",
                      calibration_receipt.at("graph_private_device_bytes")},
                     {"logical_context_output_bytes", logical_context_output_bytes},
                     {"measured_context_output_bytes", measured_context_output_bytes},
                     {"request_completion_device_bytes", request_completion_device_bytes},
                     {"request_completion_process_bytes", request_completion_process_bytes},
                     {"request_completion_external_delta_bytes",
                      request_completion_external_delta_bytes},
                     {"request_completion_headroom_bytes", request_completion_headroom_bytes},
                     {"request_completion_guard_basis",
                      "max(calibration device-wide free delta, calibration per-process NVML "
                      "delta)"},
                     {"target_kv_bytes", calibration_kv_bytes},
                     {"safety_reserve_bytes", safety_reserve_bytes},
                     {"policy_safe_bytes", policy_safe_bytes},
                     {"policy_fraction_headroom_bytes", policy_fraction_headroom_bytes},
                     {"reservation_alignment_bytes", kReservationAlignment},
                     {"max_capacity_rounding_rows", max_capacity_rounding_rows},
                     {"target_tolerance_rows", max_capacity_rounding_rows},
                     {"final_free_lower_bound_bytes", final_free_lower_bound_bytes},
                     {"final_free_upper_bound_bytes", final_free_upper_bound_bytes},
                     {"preplanning_headroom_bytes", kPreplanningHeadroomBytes},
                     {"guard_bytes", guard_bytes},
                     {"required_visible_post_load_free_bytes",
                      required_visible_post_load_free_bytes},
                     {"visible_free_formula",
                      "logical_context_output_bytes + final_free_upper_bound_bytes + "
                      "preplanning_headroom_bytes"},
                 }},
                {"before_reservation", sample_json(before_reservation)},
                {"after_reservation", sample_json(after_reservation)},
                {"guard",
                 {
                     {"allocation_phase", kBeforePlanningPhase},
                     {"release_after_snapshot_phase", kAfterOverheadPhase},
                     {"bytes", guard_bytes},
                     {"address", guard_address},
                     {"allocation_count", guard_allocation_count},
                     {"allocations", guard_allocations},
                     {"before_allocation", sample_json(before_reservation)},
                     {"after_allocation", sample_json(after_guard_allocation)},
                     {"before_release", sample_json(before_guard_release)},
                     {"after_release", sample_json(after_guard_release)},
                 }},
                {"bulk",
                 {
                     {"allocation_phase", kBeforePlanningPhase},
                     {"final_feedback_phase", kAfterOverheadPhase},
                     {"release_phase", "after constrained pipeline unload"},
                     {"bytes", bulk_reservation_bytes},
                     {"initial_bytes", initial_bulk_reservation_bytes},
                     {"correction_bytes", bulk_correction_allocated_bytes},
                     {"released_correction_bytes", bulk_correction_released_bytes},
                     {"correction_attempts", bulk_correction_attempts},
                     {"max_correction_attempts",
                      trtmc::qualification::kMaxControlledBulkCorrectionAttempts},
                     {"corrections", bulk_correction_evidence},
                     {"final_feedback",
                      {
                          {"phase", kAfterOverheadPhase},
                          {"lower_bound_bytes", final_free_lower_bound_bytes},
                          {"upper_bound_bytes", final_free_upper_bound_bytes},
                          {"max_attempts",
                           trtmc::qualification::kMaxControlledBulkCorrectionAttempts},
                          {"attempts", bulk_correction_attempts},
                          {"allocated_bytes", bulk_correction_allocated_bytes},
                          {"released_bytes", bulk_correction_released_bytes},
                          {"converged", final_feedback_converged},
                          {"controller_final_sample", sample_json(final_feedback_sample)},
                          {"actual_final_snapshot", sample_json(before_guard_release)},
                      }},
                     {"address", bulk_reservation_address},
                     {"allocation_count", bulk_reservation_allocation_count},
                     {"allocations", bulk_reservation_allocations},
                     {"before_allocation", sample_json(after_guard_allocation)},
                     {"after_allocation", sample_json(after_reservation)},
                     {"before_release", sample_json(after_constrained_unload)},
                     {"after_release", sample_json(after_release)},
                 }},
                {"warmup", auto_lifetime_json(warmup, "unmeasured-auto-warmup", false)},
                {"calibration", std::move(calibration_json)},
                {"baseline", auto_lifetime_json(baseline, "measured-auto-baseline", true)},
                {"constrained",
                 auto_lifetime_json(constrained, "measured-auto-with-reservation", true)},
                {"baseline_r", baseline_r},
                {"constrained_r", constrained_r},
                {"r_delta",
                 static_cast<std::int64_t>(baseline_r) - static_cast<std::int64_t>(constrained_r)},
                {"passed", controlled_passed},
                {"diagnostic",
                 {
                     {"policy_is_auto", controlled_policy_is_auto},
                     {"reduced_r", controlled_reduced_r},
                     {"target_bounded", controlled_target_bounded},
                     {"kv_is_exact", controlled_kv_is_exact},
                     {"guard_covers_calibrated_target", guard_covers_calibrated_target},
                     {"request_fits", controlled_request_fits},
                     {"final_snapshot_bounded", controlled_final_snapshot_bounded},
                     {"receipt_binds_final_snapshot", controlled_receipt_binds_final_snapshot},
                     {"final_formula_exact", controlled_final_formula_exact},
                     {"expected_final_r", expected_final_r},
                     {"baseline_r", baseline_r},
                     {"calibration_r", calibration_r},
                     {"constrained_r", constrained_r},
                     {"kv_bytes_per_token", bytes_per_token},
                     {"safety_reserve_bytes", constrained_receipt.at("safety_reserve_bytes")},
                     {"context_device_memory_bytes",
                      constrained_receipt.at("context_device_memory_bytes")},
                     {"constrained_pre_load_free_bytes",
                      constrained_receipt.at("pre_load_free_bytes")},
                     {"constrained_post_load_free_bytes",
                      constrained_receipt.at("post_load_free_bytes")},
                 }},
                {"after_constrained_unload", sample_json(after_constrained_unload)},
                {"after_release", sample_json(after_release)},
                {"release_recovery_process_bytes",
                 static_cast<std::int64_t>(after_release.process_used_bytes) -
                     static_cast<std::int64_t>(constrained.before_load.process_used_bytes)},
                {"release_recovery_device_wide_bytes",
                 retained_device_wide_bytes(constrained.before_load, after_release)},
            };
#if TRTMC_HAS_NVML
            output["memory_sampler"] = process_memory_sampler().metadata();
#else
            output["memory_sampler"] = {{"source", "cudaMemGetInfo-device-wide"}};
#endif
            if (!controlled_passed) {
                output["status"] = "error";
                output["error_type"] = "qualification_gate";
                output["message"] = controlled_failure;
            }
            std::cout << output.dump() << '\n';
            return controlled_passed ? 0 : 1;
        }
        if (args.second_max_sequence_length != 0) {
            auto large_options = options;
            large_options.max_sequence_length = args.second_max_sequence_length;

            json slope_warmup;
            {
                const auto warmup = run_lifetime(args, large_options, request);
                slope_warmup = lifetime_json(warmup,
                                             trtmc::qualification::make_max_sequence_length_policy(
                                                 args.second_max_sequence_length),
                                             "unmeasured-r2-warmup", false);
            }
            const auto small = run_lifetime(args, options, request);
            const auto large = run_lifetime(args, large_options, request);

            write_logits(args.logits_file, large.cycle.result);
            auto output = success_json(large.cycle.model_id, large.cycle.pipeline_type,
                                       large.cycle.result, args.logits_file);
            output["mode"] = "same_process_two_r_allocation_slope";
            output["allocation_slope_warmup"] = std::move(slope_warmup);
            output["allocation_slope_lifetimes"] = json::array({
                lifetime_json(
                    small,
                    trtmc::qualification::make_max_sequence_length_policy(args.max_sequence_length),
                    "measured-r1", true),
                lifetime_json(large,
                              trtmc::qualification::make_max_sequence_length_policy(
                                  args.second_max_sequence_length),
                              "measured-r2", true),
            });
#if TRTMC_HAS_NVML
            output["memory_sampler"] = process_memory_sampler().metadata();
#else
            output["memory_sampler"] = {{"source", "cudaMemGetInfo-device-wide"}};
#endif
            std::cout << output.dump() << '\n';
            return 0;
        }
        json load_cycle_warmup = nullptr;
        trtmc::RuntimeMemoryQualificationResultV1 explicit_warmup_result;
        const auto lifetime_policy = requested_policy_json(args);
        const bool run_load_cycle_warmup = args.warmup_load_cycle || args.load_cycles > 1;
        if (run_load_cycle_warmup) {
            auto warmup = run_lifetime(args, options, request);
            load_cycle_warmup =
                lifetime_json(warmup, lifetime_policy, "unmeasured-load-cycle-warmup", false);
            trtmc::qualification::attach_lifetime_execution_evidence(load_cycle_warmup, 0, "warmup",
                                                                     false);
            if (args.warmup_load_cycle)
                explicit_warmup_result = std::move(warmup.cycle.result);
        }
        json load_cycle_samples = json::array();
        QualificationCycle last_cycle;
        for (std::uint32_t index = 0; index < args.load_cycles; ++index) {
            auto lifetime = run_lifetime(args, options, request);
            auto sample = lifetime_json(lifetime, lifetime_policy, "measured-load-cycle", true);
            sample["cycle_index"] = index;
            trtmc::qualification::attach_lifetime_execution_evidence(
                sample, run_load_cycle_warmup ? index + 1U : index, "measured", true);
            load_cycle_samples.push_back(std::move(sample));
            last_cycle = std::move(lifetime.cycle);
        }

        write_logits(args.logits_file, last_cycle.result);
        auto output = success_json(last_cycle.model_id, last_cycle.pipeline_type, last_cycle.result,
                                   args.logits_file);
        output["sequential_request_count"] = args.repeat;
        output["sequential_requests"] = std::move(last_cycle.sequential_requests);
        output["load_cycle_warmup"] = std::move(load_cycle_warmup);
        output["load_cycle_count"] = args.load_cycles;
        output["load_cycles"] = std::move(load_cycle_samples);
        bool output_equivalence_passed = true;
        if (args.warmup_load_cycle) {
            const auto cold_start_logits_path = args.logits_file + ".cold-start.bin";
            write_logits(cold_start_logits_path, explicit_warmup_result);
            output["cold_start_logits_artifact"] =
                logits_artifact_json(cold_start_logits_path, explicit_warmup_result);
            output["lifetime_protocol"] =
                trtmc::qualification::make_single_warmup_lifetime_protocol();
            auto equivalence =
                cold_warm_output_equivalence(explicit_warmup_result, last_cycle.result);
            output_equivalence_passed = equivalence.at("passed").get<bool>();
            output["cold_warm_output_equivalence"] = std::move(equivalence);
            if (!output_equivalence_passed) {
                output["status"] = "error";
                output["error_type"] = "qualification_gate";
                output["message"] =
                    "cold and warm load cycles produced non-identical qualification outputs";
            }
        }
#if TRTMC_HAS_NVML
        output["memory_sampler"] = process_memory_sampler().metadata();
#else
        output["memory_sampler"] = {{"source", "cudaMemGetInfo-device-wide"}};
#endif
        std::cout << output.dump() << '\n';
        return output_equivalence_passed ? 0 : 1;
    } catch (const trtmc::RuntimeMemoryQualificationAdmissionError& error) {
        const auto execution_ledger = trtmc::qualification::make_attention_execution_ledger(
            error.execution_attempt_source(), error.execution_attempt_available(),
            error.execution_attempt_module_count(), error.execution_attempt_before(),
            error.execution_attempt_after(), error.execution_attempt_delta());
        const bool rejected_before_attention =
            trtmc::qualification::attention_execution_ledger_proves_before_attention(
                execution_ledger);
        if (!rejected_before_attention) {
            std::cout
                << json{
                       {"status", "error"},
                       {"error_type", "qualification_gate"},
                       {"stage", "after_execution_attempt"},
                       {"attention_started", true},
                       {"attention_execution_ledger", execution_ledger},
                       {"message",
                        "admission error was raised after an execution attempt: " +
                            std::string(error.what())},
                   }
                       .dump()
                << '\n';
            return 1;
        }
        std::cout << json{
                         {"status", "rejected"},
                         {"error_type", "admission"},
                         {"stage", "before_attention"},
                         {"attention_started", false},
                         {"prefill_launches", 0},
                         {"decode_launches", 0},
                         {"final_kv_position", 0},
                         {"invocations", json::array()},
                         {"selected_token_ids", json::array()},
                         {"step_top1_token_ids", json::array()},
                         {"attention_execution_ledger", execution_ledger},
                         {"message", error.what()},
                     }
                         .dump()
                  << '\n';
        return 3;
    } catch (const QualificationDiagnosticError& error) {
        std::cout << json{
                         {"status", "error"},
                         {"error_type", "qualification_gate"},
                         {"message", error.what()},
                         {"diagnostic", error.diagnostic()},
                     }
                         .dump()
                  << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cout << json{
                         {"status", "error"},
                         {"error_type", "runtime"},
                         {"message", error.what()},
                     }
                         .dump()
                  << '\n';
        return 1;
    }
}
