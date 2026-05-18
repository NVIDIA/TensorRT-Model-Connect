#include "trtmc/runtime/distributed_runtime.h"

#include "runtime/core/cuda_common.h"
#include "utils/json_helpers.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

namespace trtmc {

namespace {

struct NcclUniqueId {
    char internal[128];
};

using NcclComm = void*;
using NcclResult = int;
using NcclGetUniqueIdFn = NcclResult (*)(NcclUniqueId*);
using NcclCommInitRankFn = NcclResult (*)(NcclComm*, int, NcclUniqueId, int);
using NcclCommDestroyFn = NcclResult (*)(NcclComm);
using NcclGetErrorStringFn = const char* (*)(NcclResult);

int env_int(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
        return fallback;
    char* end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (end == raw)
        return fallback;
    return static_cast<int>(value);
}

std::string env_string(const char* name, const std::string& fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
        return fallback;
    return raw;
}

int detect_world_size() {
    int size = env_int("OMPI_COMM_WORLD_SIZE", -1);
    if (size > 0)
        return size;
    size = env_int("PMI_SIZE", -1);
    if (size > 0)
        return size;
    return env_int("WORLD_SIZE", 1);
}

int detect_rank() {
    int rank = env_int("OMPI_COMM_WORLD_RANK", -1);
    if (rank >= 0)
        return rank;
    rank = env_int("PMI_RANK", -1);
    if (rank >= 0)
        return rank;
    return env_int("RANK", 0);
}

int detect_local_rank() {
    int rank = env_int("OMPI_COMM_WORLD_LOCAL_RANK", -1);
    if (rank >= 0)
        return rank;
    rank = env_int("MPI_LOCALRANKID", -1);
    if (rank >= 0)
        return rank;
    return env_int("LOCAL_RANK", 0);
}

std::filesystem::path rendezvous_path() {
    std::string base = env_string("TRTMC_NCCL_RENDEZVOUS", "");
    if (!base.empty())
        return std::filesystem::path(base);
    std::string job = env_string("OMPI_COMM_WORLD_JOBID", "");
    if (job.empty())
        job = env_string("PMIX_NAMESPACE", "");
    if (job.empty())
        job = std::to_string(getppid());
    return std::filesystem::temp_directory_path() /
           ("trtmc_nccl_" + std::to_string(getuid()) + "_" + job + ".bin");
}

class NcclRuntime {
  public:
    NcclRuntime() {
        handle_ = dlopen("libnccl.so.2", RTLD_NOW | RTLD_LOCAL);
        if (handle_ == nullptr)
            handle_ = dlopen("libnccl.so", RTLD_NOW | RTLD_LOCAL);
        if (handle_ == nullptr) {
            throw std::runtime_error(
                std::string("Failed to load NCCL for tensor parallel runtime: ") + dlerror());
        }
        get_unique_id_ = load<NcclGetUniqueIdFn>("ncclGetUniqueId");
        comm_init_rank_ = load<NcclCommInitRankFn>("ncclCommInitRank");
        comm_destroy_ = load<NcclCommDestroyFn>("ncclCommDestroy");
        get_error_string_ = load<NcclGetErrorStringFn>("ncclGetErrorString");
    }

    ~NcclRuntime() {
        if (comm_ != nullptr) {
            if (env_int("TRTMC_NCCL_SKIP_DESTROY", 0) != 0) {
                // Escape hatch for process-exit hangs in external runtimes.
                // Normal teardown should destroy the communicator after TRT
                // contexts and engines have released it.
                comm_ = nullptr;
                handle_ = nullptr;
            } else {
                comm_destroy_(comm_);
                comm_ = nullptr;
            }
        }
        if (handle_ != nullptr)
            dlclose(handle_);
    }

    void init(int tp_size, int rank, const NcclUniqueId& id) {
        check(comm_init_rank_(&comm_, tp_size, id, rank), "ncclCommInitRank");
    }

    NcclUniqueId unique_id() {
        NcclUniqueId id{};
        check(get_unique_id_(&id), "ncclGetUniqueId");
        return id;
    }

    void* communicator() const { return comm_; }

  private:
    template <typename T>
    T load(const char* symbol) {
        dlerror();
        void* raw = dlsym(handle_, symbol);
        const char* err = dlerror();
        if (err != nullptr || raw == nullptr)
            throw std::runtime_error(std::string("Failed to resolve NCCL symbol ") + symbol);
        return reinterpret_cast<T>(raw);
    }

    void check(NcclResult result, const char* op) const {
        if (result == 0)
            return;
        const char* msg = get_error_string_ ? get_error_string_(result) : "unknown NCCL error";
        throw std::runtime_error(std::string(op) + " failed: " + msg);
    }

    void* handle_{nullptr};
    NcclComm comm_{nullptr};
    NcclGetUniqueIdFn get_unique_id_{nullptr};
    NcclCommInitRankFn comm_init_rank_{nullptr};
    NcclCommDestroyFn comm_destroy_{nullptr};
    NcclGetErrorStringFn get_error_string_{nullptr};
};

void write_unique_id(const std::filesystem::path& path, const NcclUniqueId& id) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    const auto tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out)
            throw std::runtime_error("Failed to write NCCL rendezvous file: " + tmp);
        out.write(id.internal, sizeof(id.internal));
    }
    std::filesystem::rename(tmp, path);
}

NcclUniqueId read_unique_id(const std::filesystem::path& path) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!std::filesystem::exists(path)) {
        if (std::chrono::steady_clock::now() > deadline) {
            throw std::runtime_error("Timed out waiting for NCCL rendezvous file: " +
                                     path.string());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    NcclUniqueId id{};
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Failed to read NCCL rendezvous file: " + path.string());
    in.read(id.internal, sizeof(id.internal));
    if (in.gcount() != static_cast<std::streamsize>(sizeof(id.internal)))
        throw std::runtime_error("Short NCCL rendezvous file: " + path.string());
    return id;
}

void bind_local_cuda_device(int local_rank) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
        return;
    const int device = local_rank % count;
    auto status = cudaSetDevice(device);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("cudaSetDevice failed for tensor parallel rank: ") +
                                 cudaGetErrorString(status));
}

std::size_t skip_json_ws(const std::string& text, std::size_t pos) {
    while (pos < text.size()) {
        const char c = text[pos];
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n')
            break;
        ++pos;
    }
    return pos;
}

std::string extract_json_object_for_key(const std::string& text, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos)
        return {};
    const std::size_t colon = text.find(':', key_pos + needle.size());
    if (colon == std::string::npos)
        return {};
    std::size_t pos = skip_json_ws(text, colon + 1);
    if (pos >= text.size() || text[pos] != '{')
        return {};

    const std::size_t start = pos;
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (; pos < text.size(); ++pos) {
        const char c = text[pos];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '{') {
            ++depth;
        } else if (c == '}') {
            --depth;
            if (depth == 0)
                return text.substr(start, pos - start + 1);
        }
    }
    return {};
}

} // namespace

DistributedPlanRuntimeConfig parse_distributed_plan_runtime_config(
    const std::string& plan_json, const std::string& component) {
    if (plan_json.empty())
        throw std::runtime_error("distributed_plan.json section is empty");

    const std::string schema = extract_json_string(plan_json, "schema_version", "");
    if (schema != "1.0")
        throw std::runtime_error("Unsupported distributed_plan.json schema_version: " + schema);

    DistributedPlanRuntimeConfig cfg;
    cfg.component = component;

    const std::string mesh = extract_json_object_for_key(plan_json, "mesh");
    if (mesh.empty())
        throw std::runtime_error("distributed_plan.json is missing mesh");
    const std::string axes = extract_json_object_for_key(mesh, "axes");
    if (axes.empty())
        throw std::runtime_error("distributed_plan.json mesh is missing axes");

    cfg.world_size = extract_json_int(mesh, "world_size", 1);
    cfg.tp_size = extract_json_int(axes, "tp", 1);
    cfg.pp_size = extract_json_int(axes, "pp", 1);
    cfg.cp_size = extract_json_int(axes, "cp", 1);
    cfg.dp_size = extract_json_int(axes, "dp", 1);
    cfg.ep_size = extract_json_int(axes, "ep", 1);
    if (cfg.world_size < 1 || cfg.tp_size < 1 || cfg.pp_size < 1 || cfg.cp_size < 1 ||
        cfg.dp_size < 1 || cfg.ep_size < 1) {
        throw std::runtime_error("distributed_plan.json mesh sizes must be >= 1");
    }
    const int product = cfg.tp_size * cfg.pp_size * cfg.cp_size * cfg.dp_size * cfg.ep_size;
    if (product != cfg.world_size) {
        throw std::runtime_error("distributed_plan.json mesh axis product does not match world_size");
    }
    if (cfg.pp_size != 1 || cfg.cp_size != 1 || cfg.dp_size != 1 || cfg.ep_size != 1) {
        throw std::runtime_error(
            "This runtime currently supports only tensor-parallel distributed plans");
    }

    const std::string bundle_sections = extract_json_object_for_key(plan_json, "bundle_sections");
    if (bundle_sections.empty())
        throw std::runtime_error("distributed_plan.json is missing bundle_sections");
    const std::string component_section = extract_json_object_for_key(bundle_sections, component);
    if (component_section.empty()) {
        throw std::runtime_error("distributed_plan.json is missing bundle section for component '" +
                                 component + "'");
    }

    cfg.rank_section_pattern =
        extract_json_string(component_section, "rank_section_pattern", "");
    if (cfg.rank_section_pattern.empty()) {
        cfg.rank_section_pattern = extract_json_string(component_section, "section", "");
    }
    if (cfg.rank_section_pattern.empty()) {
        throw std::runtime_error("distributed_plan.json component '" + component +
                                 "' does not name a rank_section_pattern or section");
    }

    cfg.enabled = cfg.world_size > 1;
    return cfg;
}

std::string distributed_rank_section_name(const std::string& pattern, int rank) {
    std::string out = pattern;
    const std::string token = "{rank}";
    const std::string value = std::to_string(rank);
    std::size_t pos = 0;
    while ((pos = out.find(token, pos)) != std::string::npos) {
        out.replace(pos, token.size(), value);
        pos += value.size();
    }
    return out;
}

DistributedRuntimeGroup initialize_tensor_parallel_group(int tp_size) {
    DistributedRuntimeGroup group;
    group.tp_size = tp_size;
    group.world_size = detect_world_size();
    group.rank = detect_rank();
    group.local_rank = detect_local_rank();

    if (tp_size <= 1)
        return group;
    if (group.world_size != tp_size) {
        throw std::runtime_error("Tensor-parallel runtime requires launched world size to equal "
                                 "the distributed plan tp size for this initial implementation");
    }
    if (group.rank < 0 || group.rank >= tp_size)
        throw std::runtime_error("Tensor-parallel rank is outside [0, tp_size)");

    bind_local_cuda_device(group.local_rank);
    auto runtime = std::make_shared<NcclRuntime>();
    const auto path = rendezvous_path();
    NcclUniqueId id{};
    if (group.rank == 0) {
        id = runtime->unique_id();
        write_unique_id(path, id);
    } else {
        id = read_unique_id(path);
    }
    runtime->init(tp_size, group.rank, id);
    group.communicator = runtime->communicator();
    group.owner = std::move(runtime);
    return group;
}

MeshRuntimeGroup initialize_mesh_runtime_group(const MeshRuntimeConfig& config) {
    if (!config.enabled)
        return MeshRuntimeGroup{};
    return initialize_tensor_parallel_group(config.tp_size);
}

DistributedRuntimeGroup initialize_distributed_group(const DistributedPlanRuntimeConfig& config) {
    return initialize_mesh_runtime_group(config);
}

} // namespace trtmc
