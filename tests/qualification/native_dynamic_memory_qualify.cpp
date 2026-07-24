/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Developer-only runner for Section 9.2 of the native dynamic-memory plan.
// It loads the real bundle through the same public factory as `trtmc run`,
// then discovers the independently versioned private qualification interface.

#include "native_dynamic_memory_qualify_schema.h"
#include "runtime/domains/text/dynamic_memory/runtime_kv_setup.h"
#include "runtime/domains/text/dynamic_memory/runtime_memory_qualification.h"
#include "trtmc/pipeline.h"

#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#ifndef TRTMC_HAS_NVML
#define TRTMC_HAS_NVML 0
#endif
#if TRTMC_HAS_NVML
#include <nvml.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

struct Arguments {
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
    std::vector<std::string> backend_dirs;
    std::vector<std::string> model_plugin_dirs;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message + "\nusage: trtmc_dynamic_memory_qualify --bundle MODEL.trtfb "
                  "--tokens IDS.txt --logits LOGITS.bin [--max-new-tokens N] "
                  "[--max-sequence-length N] [--kv-cache-bytes N | --kv-cache-fraction F] "
                  "[--second-max-sequence-length N] "
                  "[--controlled-reservation-target-tokens N] "
                  "[--repeat N | --load-cycles N] "
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
        if (flag == "--bundle")
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
        else if (flag == "--backend-dir")
            out.backend_dirs.push_back(require_value());
        else if (flag == "--model-plugin-dir")
            out.model_plugin_dirs.push_back(require_value());
        else
            usage_error("unknown argument: " + flag);
    }
    if (out.bundle.empty() || out.token_file.empty() || out.logits_file.empty())
        usage_error("--bundle, --tokens, and --logits are required");
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
    return out;
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

json parse_receipt(const std::string& receipt) {
    if (receipt.empty())
        throw std::runtime_error("runtime-memory qualification returned an empty receipt");
    return json::parse(receipt);
}

void append_default_search_paths(const char* argv0, trtmc::LoadOptionsV2& options) {
    const auto executable =
        std::filesystem::absolute(std::filesystem::path(argv0)).lexically_normal();
    const auto build_dir = executable.parent_path();
    options.backend_search_paths.push_back(build_dir.string());
    options.model_plugin_search_paths.push_back((build_dir / "models/qwen").string());
    options.model_plugin_search_paths.push_back((build_dir / "models/llama").string());
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
        {"logits_artifact",
         {
             {"path", std::filesystem::absolute(logits_path).string()},
             {"format", "trtmc-qualification-logits-v1"},
             {"dtype", "float32"},
             {"rows", result.step_logits.size()},
             {"vocab_size", result.step_logits.front().size()},
         }},
    };
    out["invocations"] = json::array();
    for (const auto& trace : result.invocations) {
        out["invocations"].push_back({
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
    std::vector<std::int32_t> top1;
    top1.reserve(result.step_logits.size());
    for (const auto& logits : result.step_logits) {
        top1.push_back(static_cast<std::int32_t>(
            std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()))));
    }
    out["step_top1_token_ids"] = std::move(top1);
    return out;
}

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
        check(nvmlDeviceGetIndex(device_, &physical_device_index_), "nvmlDeviceGetIndex");
        char uuid[NVML_DEVICE_UUID_V2_BUFFER_SIZE]{};
        check(nvmlDeviceGetUUID(device_, uuid, sizeof(uuid)), "nvmlDeviceGetUUID");
        gpu_uuid_ = uuid;
        pid_ = static_cast<unsigned int>(getpid());
    }

    std::uint64_t sample_process_used_bytes() const {
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
            for (unsigned int index = 0; index < count; ++index) {
                if (processes[index].pid != pid_)
                    continue;
                if (processes[index].usedGpuMemory == NVML_VALUE_NOT_AVAILABLE) {
                    throw std::runtime_error("NVML did not report process GPU memory");
                }
                return processes[index].usedGpuMemory;
            }
            throw std::runtime_error("NVML did not list the qualification runner process");
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
};

DeviceMemorySample sample_device_memory() {
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
    std::uint64_t process_used_bytes = 0;
#if TRTMC_HAS_NVML
    process_used_bytes = process_memory_sampler().sample_process_used_bytes();
#else
    process_used_bytes = static_cast<std::uint64_t>(total_bytes - free_bytes);
#endif
    return {static_cast<std::uint64_t>(free_bytes), static_cast<std::uint64_t>(total_bytes),
            process_used_bytes};
}

json sample_json(const DeviceMemorySample& sample) {
    return {
        {"free_bytes", sample.free_bytes},
        {"total_bytes", sample.total_bytes},
        {"used_bytes", sample.total_bytes - sample.free_bytes},
        {"process_used_bytes", sample.process_used_bytes},
    };
}

class RuntimePhaseMemoryObserverScope {
  public:
    explicit RuntimePhaseMemoryObserverScope(json& samples) : samples_(samples) {
        trtmc::set_runtime_device_memory_qualification_observer(
            [this](const char* phase, const trtmc::RuntimeDeviceMemorySnapshot& snapshot) {
                std::uint64_t process_used_bytes = snapshot.total_bytes - snapshot.free_bytes;
#if TRTMC_HAS_NVML
                process_used_bytes = process_memory_sampler().sample_process_used_bytes();
#endif
                samples_.push_back(trtmc::qualification::make_runtime_phase_memory_sample(
                    phase, snapshot.device, snapshot.free_bytes, snapshot.total_bytes,
                    process_used_bytes));
            });
    }

    RuntimePhaseMemoryObserverScope(const RuntimePhaseMemoryObserverScope&) = delete;
    RuntimePhaseMemoryObserverScope& operator=(const RuntimePhaseMemoryObserverScope&) = delete;

    ~RuntimePhaseMemoryObserverScope() {
        trtmc::set_runtime_device_memory_qualification_observer({});
    }

  private:
    json& samples_;
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
    explicit DeviceReservation(std::uint64_t bytes) : bytes_(bytes) {
        if (bytes == 0 ||
            bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::invalid_argument("controlled reservation byte count is invalid");
        }
        constexpr std::uint64_t kMaxChunkBytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
        auto remaining = bytes;
        while (remaining != 0) {
            const auto chunk_bytes = std::min(remaining, kMaxChunkBytes);
            void* address = nullptr;
            const auto status = cudaMalloc(&address, static_cast<std::size_t>(chunk_bytes));
            if (status != cudaSuccess) {
                release_noexcept();
                throw std::runtime_error(std::string("controlled cudaMalloc failed after ") +
                                         std::to_string(addresses_.size()) +
                                         " chunks: " + cudaGetErrorString(status));
            }
            addresses_.push_back(address);
            chunk_bytes_.push_back(chunk_bytes);
            remaining -= chunk_bytes;
        }
    }

    DeviceReservation(const DeviceReservation&) = delete;
    DeviceReservation& operator=(const DeviceReservation&) = delete;

    ~DeviceReservation() { release_noexcept(); }

    void release() {
        while (!addresses_.empty()) {
            auto* address = addresses_.back();
            addresses_.pop_back();
            chunk_bytes_.pop_back();
            const auto status = cudaFree(address);
            if (status != cudaSuccess) {
                throw std::runtime_error(std::string("controlled cudaFree failed: ") +
                                         cudaGetErrorString(status));
            }
        }
    }

    std::uint64_t bytes() const noexcept { return bytes_; }
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
            addresses_.pop_back();
            chunk_bytes_.pop_back();
            (void)cudaFree(address);
        }
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
                                   const trtmc::RuntimeMemoryQualificationRequestV1& request) {
    QualificationLifetime lifetime;
    RuntimePhaseMemoryObserverScope observer(lifetime.runtime_phase_memory_samples);
    lifetime.before_load = sample_device_memory();
    lifetime.cycle = run_cycle(args, options, request);
    lifetime.after_unload = sample_device_memory();
    return lifetime;
}

json lifetime_json(const QualificationLifetime& lifetime, std::uint64_t requested_tokens,
                   const char* label, bool measured) {
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
        {"policy",
         {
             {"kind", "max_sequence_length"},
             {"requested_tokens", requested_tokens},
         }},
        {"runtime_kv_capacity_tokens", receipt.at("runtime_kv_capacity_tokens")},
        {"kv_allocation_id", receipt.at("kv_allocation_id")},
        {"runtime_memory_receipt", receipt},
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

} // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_arguments(argc, argv);
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

        trtmc::RuntimeMemoryQualificationRequestV1 request;
        request.input_ids = read_tokens(args.token_file);
        request.max_new_tokens = args.max_new_tokens;
        if (args.controlled_reservation_target_tokens != 0) {
            // The first TensorRT lifetime can retain process-global lazy
            // initialization state.  It is intentionally excluded from the
            // controlled-reservation comparison so that baseline and
            // constrained lifetimes use the same warmed process boundary.
            const auto warmup = run_lifetime(args, options, request);
            const auto baseline = run_lifetime(args, options, request);
            const auto baseline_receipt =
                parse_receipt(baseline.cycle.result.runtime_memory_receipt_json);
            const auto baseline_r =
                baseline_receipt.at("runtime_kv_capacity_tokens").get<std::uint64_t>();
            const auto bytes_per_token =
                baseline_receipt.at("kv_bytes_per_token").get<std::uint64_t>();
            const auto baseline_kv_bytes =
                baseline_receipt.at("kv_reserved_bytes").get<std::uint64_t>();
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
            const auto target_kv_bytes =
                args.controlled_reservation_target_tokens * bytes_per_token;
            const auto safety_reserve_bytes =
                baseline_receipt.at("safety_reserve_bytes").get<std::uint64_t>();
            const auto context_device_memory_bytes =
                baseline_receipt.at("context_device_memory_bytes").get<std::uint64_t>();
            const auto external_device_output_bytes =
                baseline_receipt.at("external_device_output_bytes").get<std::uint64_t>();
            const auto graph_private_device_bytes =
                baseline_receipt.at("graph_private_device_bytes").get<std::uint64_t>();
            const auto post_load_runtime_overhead_bytes = context_device_memory_bytes +
                                                          external_device_output_bytes +
                                                          graph_private_device_bytes;
            constexpr std::uint64_t kMinimumTargetHeadroomBytes = 8ULL * 1024ULL * 1024ULL;
            const auto target_fraction_headroom_bytes =
                target_kv_bytes / 8U + (target_kv_bytes % 8U == 0 ? 0U : 1U);
            const auto target_headroom_bytes =
                std::max(kMinimumTargetHeadroomBytes, target_fraction_headroom_bytes);
            const auto required_post_load_free_bytes = target_kv_bytes + safety_reserve_bytes +
                                                       post_load_runtime_overhead_bytes +
                                                       target_headroom_bytes;
            constexpr std::uint64_t kReservationAlignment = 2ULL * 1024ULL * 1024ULL;
            DeviceMemorySample before_reservation;
            DeviceMemorySample after_reservation;
            std::uint64_t reservation_bytes = 0;
            std::unique_ptr<DeviceReservation> reservation;
            bool reservation_action_invoked = false;
            trtmc::set_runtime_device_memory_qualification_pre_snapshot_action(
                [&](const char* phase) {
                    if (std::string(phase) != "before runtime KV planning") {
                        return;
                    }
                    if (reservation_action_invoked) {
                        throw std::logic_error("controlled post-load reservation action "
                                               "ran more than once");
                    }
                    before_reservation = sample_device_memory();
                    if (before_reservation.free_bytes <=
                        required_post_load_free_bytes + kReservationAlignment) {
                        throw std::runtime_error("insufficient post-load free memory for "
                                                 "controlled reservation");
                    }
                    reservation_bytes =
                        ((before_reservation.free_bytes - required_post_load_free_bytes) /
                         kReservationAlignment) *
                        kReservationAlignment;
                    reservation = std::make_unique<DeviceReservation>(reservation_bytes);
                    after_reservation = sample_device_memory();
                    reservation_action_invoked = true;
                });
            QualificationLifetime constrained;
            try {
                constrained = run_lifetime(args, options, request);
            } catch (...) {
                trtmc::set_runtime_device_memory_qualification_pre_snapshot_action({});
                throw;
            }
            trtmc::set_runtime_device_memory_qualification_pre_snapshot_action({});
            if (!reservation_action_invoked || !reservation) {
                throw std::logic_error("controlled post-load reservation action did not run");
            }
            const auto reservation_address = reservation->address();
            const auto reservation_allocation_count = reservation->allocation_count();
            const auto reservation_allocations = reservation->allocations_json();
            const auto constrained_receipt =
                parse_receipt(constrained.cycle.result.runtime_memory_receipt_json);
            const auto constrained_r =
                constrained_receipt.at("runtime_kv_capacity_tokens").get<std::uint64_t>();
            const auto controlled_policy_is_auto = baseline_receipt.at("policy") == "auto" &&
                                                   constrained_receipt.at("policy") == "auto";
            const auto controlled_reduced_r = constrained_r < baseline_r;
            const auto controlled_request_fits =
                constrained.cycle.result.final_kv_position <= constrained_r;
            const auto controlled_passed =
                controlled_policy_is_auto && controlled_reduced_r && controlled_request_fits;
            std::string controlled_failure;
            if (!controlled_policy_is_auto) {
                controlled_failure = "controlled reservation did not use auto policy";
            } else if (!controlled_reduced_r) {
                controlled_failure = "controlled reservation did not reduce runtime R";
            } else if (!controlled_request_fits) {
                controlled_failure = "controlled reservation request did not fit R";
            }
            const auto after_constrained_unload = constrained.after_unload;
            reservation.reset();
            const auto after_release = sample_device_memory();

            auto auto_lifetime_json = [](const QualificationLifetime& lifetime, const char* label,
                                         bool measured) {
                auto out = lifetime_json(lifetime, 0, label, measured);
                out["policy"] = {{"kind", "auto"}};
                return out;
            };
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
                     {"required_free_basis", "post-load runtime overhead before KV planning"},
                     {"context_device_memory_bytes", context_device_memory_bytes},
                     {"external_device_output_bytes", external_device_output_bytes},
                     {"graph_private_device_bytes", graph_private_device_bytes},
                     {"post_load_runtime_overhead_bytes", post_load_runtime_overhead_bytes},
                     {"target_kv_bytes", target_kv_bytes},
                     {"safety_reserve_bytes", safety_reserve_bytes},
                     {"minimum_target_headroom_bytes", kMinimumTargetHeadroomBytes},
                     {"target_fraction_denominator", 8},
                     {"target_fraction_headroom_bytes", target_fraction_headroom_bytes},
                     {"target_headroom_bytes", target_headroom_bytes},
                     {"margin_formula", "safety_reserve_bytes + "
                                        "max(8MiB, ceil(target_kv_bytes/8))"},
                     {"required_post_load_free_bytes", required_post_load_free_bytes},
                 }},
                {"reservation_phase", "before runtime KV planning"},
                {"reservation_bytes", reservation_bytes},
                {"reservation_address", reservation_address},
                {"reservation_allocation_count", reservation_allocation_count},
                {"reservation_allocations", reservation_allocations},
                {"before_reservation", sample_json(before_reservation)},
                {"after_reservation", sample_json(after_reservation)},
                {"warmup", auto_lifetime_json(warmup, "unmeasured-auto-warmup", false)},
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
                     {"request_fits", controlled_request_fits},
                     {"baseline_r", baseline_r},
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
                slope_warmup = lifetime_json(warmup, args.second_max_sequence_length,
                                             "unmeasured-r2-warmup", false);
            }
            const auto small = run_lifetime(args, options, request);
            const auto large = run_lifetime(args, large_options, request);

            write_logits(args.logits_file, large.cycle.result);
            auto output = success_json(large.cycle.model_id, large.cycle.pipeline_type,
                                       large.cycle.result, args.logits_file);
            output["mode"] = "same_process_two_r_allocation_slope";
            output["allocation_slope_warmup"] = std::move(slope_warmup);
            output["allocation_slope_lifetimes"] = json::array(
                {lifetime_json(small, args.max_sequence_length, "measured-r1", true),
                 lifetime_json(large, args.second_max_sequence_length, "measured-r2", true)});
#if TRTMC_HAS_NVML
            output["memory_sampler"] = process_memory_sampler().metadata();
#else
            output["memory_sampler"] = {{"source", "cudaMemGetInfo-device-wide"}};
#endif
            std::cout << output.dump() << '\n';
            return 0;
        }
        json load_cycle_warmup = nullptr;
        if (args.load_cycles > 1) {
            const auto warmup = run_lifetime(args, options, request);
            load_cycle_warmup = lifetime_json(warmup, args.max_sequence_length,
                                              "unmeasured-load-cycle-warmup", false);
        }
        json load_cycle_samples = json::array();
        QualificationCycle last_cycle;
        for (std::uint32_t index = 0; index < args.load_cycles; ++index) {
            auto lifetime = run_lifetime(args, options, request);
            auto sample =
                lifetime_json(lifetime, args.max_sequence_length, "measured-load-cycle", true);
            sample["cycle_index"] = index;
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
#if TRTMC_HAS_NVML
        output["memory_sampler"] = process_memory_sampler().metadata();
#else
        output["memory_sampler"] = {{"source", "cudaMemGetInfo-device-wide"}};
#endif
        std::cout << output.dump() << '\n';
        return 0;
    } catch (const trtmc::RuntimeMemoryQualificationAdmissionError& error) {
        std::cout << json{
                         {"status", "rejected"},
                         {"error_type", "admission"},
                         {"stage", "before_attention"},
                         {"prefill_launches", 0},
                         {"decode_launches", 0},
                         {"message", error.what()},
                     }
                         .dump()
                  << '\n';
        return 3;
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
