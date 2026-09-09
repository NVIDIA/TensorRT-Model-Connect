/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "api_internal.h"
#include "trtmc/bundle.h"
#include "trtmc/runtime/family_loader.h"

#include <algorithm>
#include <cstddef>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace trtmc::api {

void require(bool condition, const char* message) {
    if (!condition)
        throw ApiFailure{TRTMC_INVALID_ARGUMENT, message};
}

std::size_t checked_size(std::uint64_t count, std::size_t element_size) {
    const auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
    require(count <= limit / element_size, "array size overflows the host address range");
    return static_cast<std::size_t>(count);
}

std::string_view string_view(trtmc_string_view value) {
    const auto bytes = checked_span(value.data, value.size);
    return bytes.empty() ? std::string_view{} : std::string_view(bytes.data(), bytes.size());
}

std::string owned_string(std::string_view value) {
    return value.empty() ? std::string{} : std::string(value);
}

std::string path_string(trtmc_string_view value) {
    const auto view = string_view(value);
    require(view.find('\0') == std::string_view::npos, "path contains an embedded NUL");
    return owned_string(view);
}

trtmc_string_view borrowed_string(std::string_view value) noexcept {
    return {value.data(), value.size()};
}

using OwnedConfigValue =
    std::variant<std::int64_t, double, bool, std::string, std::vector<std::int64_t>,
                 std::vector<double>, std::vector<std::string>>;

std::uint32_t wire_kind(internal::ConfigKind kind) {
    switch (kind) {
    case internal::ConfigKind::I64:
        return TRTMC_CONFIG_I64;
    case internal::ConfigKind::F64:
        return TRTMC_CONFIG_F64;
    case internal::ConfigKind::Bool:
        return TRTMC_CONFIG_BOOL;
    case internal::ConfigKind::String:
        return TRTMC_CONFIG_STRING;
    case internal::ConfigKind::I64List:
        return TRTMC_CONFIG_I64_LIST;
    case internal::ConfigKind::F64List:
        return TRTMC_CONFIG_F64_LIST;
    case internal::ConfigKind::StringList:
        return TRTMC_CONFIG_STRING_LIST;
    }
    throw ApiFailure{TRTMC_INTERNAL_ERROR, "family returned an unknown config kind"};
}

template <class T>
std::vector<T> copy_array(trtmc::Span<const T> value) {
    const auto checked = checked_span(value.data(), value.size());
    return checked.empty() ? std::vector<T>{} : std::vector<T>(checked.begin(), checked.end());
}

OwnedConfigValue own_value(const internal::ConfigValue& value) {
    switch (internal::config_kind(value)) {
    case internal::ConfigKind::I64:
        return std::get<std::int64_t>(value);
    case internal::ConfigKind::F64:
        return std::get<double>(value);
    case internal::ConfigKind::Bool:
        return std::get<bool>(value);
    case internal::ConfigKind::String:
        return owned_string(string_view(borrowed_string(std::get<std::string_view>(value))));
    case internal::ConfigKind::I64List:
        return copy_array(std::get<trtmc::Span<const std::int64_t>>(value));
    case internal::ConfigKind::F64List:
        return copy_array(std::get<trtmc::Span<const double>>(value));
    case internal::ConfigKind::StringList: {
        const auto strings = std::get<trtmc::Span<const std::string_view>>(value);
        const auto checked = checked_span(strings.data(), strings.size());
        std::vector<std::string> result;
        result.reserve(checked.size());
        for (const auto item : checked)
            result.push_back(owned_string(string_view(borrowed_string(item))));
        return result;
    }
    }
    throw ApiFailure{TRTMC_INTERNAL_ERROR, "family returned an unknown config value"};
}

struct ConfigFieldSnapshot {
    std::string name;
    std::string description;
    std::uint32_t kind;
    std::optional<OwnedConfigValue> value;
    std::vector<trtmc_string_view> strings;
    trtmc_config_field_v1 view{};

    explicit ConfigFieldSnapshot(const internal::ConfigField& field)
        : name(owned_string(string_view(borrowed_string(field.name)))),
          description(owned_string(string_view(borrowed_string(field.description)))),
          kind(wire_kind(field.kind)) {
        if (field.default_value) {
            if (internal::config_kind(*field.default_value) != field.kind)
                throw ApiFailure{TRTMC_INTERNAL_ERROR, "family config default has the wrong type"};
            value = own_value(*field.default_value);
        }
    }

    // Called after snapshots reach their final model-owned addresses.
    void bind_view() {
        view.name = borrowed_string(name);
        view.description = borrowed_string(description);
        view.kind = kind;
        view.has_fixed_default = value.has_value();
        if (!value)
            return;
        auto& output = view.default_value;
        output.kind = kind;
        switch (kind) {
        case TRTMC_CONFIG_I64:
            output.as.i64 = std::get<std::int64_t>(*value);
            break;
        case TRTMC_CONFIG_F64:
            output.as.f64 = std::get<double>(*value);
            break;
        case TRTMC_CONFIG_BOOL:
            output.as.boolean = std::get<bool>(*value);
            break;
        case TRTMC_CONFIG_STRING:
            output.as.string = borrowed_string(std::get<std::string>(*value));
            break;
        case TRTMC_CONFIG_I64_LIST: {
            const auto& values = std::get<std::vector<std::int64_t>>(*value);
            output.as.i64_list = {values.data(), values.size()};
            break;
        }
        case TRTMC_CONFIG_F64_LIST: {
            const auto& values = std::get<std::vector<double>>(*value);
            output.as.f64_list = {values.data(), values.size()};
            break;
        }
        case TRTMC_CONFIG_STRING_LIST:
            for (const auto& item : std::get<std::vector<std::string>>(*value))
                strings.push_back(borrowed_string(item));
            output.as.string_list = {strings.data(), strings.size()};
            break;
        }
    }
};

namespace {

struct ConfigSnapshotStorage final : ResultStorage {
    explicit ConfigSnapshotStorage(internal::ConfigView config) {
        const auto input = checked_span(config.data(), config.size());
        fields.reserve(input.size());
        entries.reserve(input.size());
        for (const auto& entry : input) {
            fields.emplace_back(internal::ConfigField{
                entry.name, internal::config_kind(entry.value), entry.value, {}});
        }
        for (auto& field : fields) {
            field.bind_view();
            entries.push_back({field.view.name, field.view.default_value});
        }
    }

    std::vector<ConfigFieldSnapshot> fields;
    std::vector<trtmc_config_entry_v1> entries;
};

} // namespace

trtmc_result* make_config_snapshot(internal::ConfigView config) {
    return make_result<ConfigSnapshotStorage>(config);
}

trtmc_status TRTMC_CALL config_snapshot_view(const trtmc_result* result, trtmc_config_view_v1* out,
                                             trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out != nullptr, "config view output is null");
        const auto& storage = require_result<ConfigSnapshotStorage>(result);
        *out = {storage.entries.data(), storage.entries.size()};
    });
}

internal::ConfigValue ConvertedConfig::convert(const trtmc_config_value_v1& value) {
    switch (value.kind) {
    case TRTMC_CONFIG_I64:
        return value.as.i64;
    case TRTMC_CONFIG_F64:
        return value.as.f64;
    case TRTMC_CONFIG_BOOL:
        require(value.as.boolean <= 1, "config bool must be zero or one");
        return value.as.boolean != 0;
    case TRTMC_CONFIG_STRING:
        return string_view(value.as.string);
    case TRTMC_CONFIG_I64_LIST:
        return checked_span(value.as.i64_list.data, value.as.i64_list.size);
    case TRTMC_CONFIG_F64_LIST:
        return checked_span(value.as.f64_list.data, value.as.f64_list.size);
    case TRTMC_CONFIG_STRING_LIST: {
        const auto input = checked_span(value.as.string_list.data, value.as.string_list.size);
        auto& strings = string_lists_.emplace_back();
        strings.reserve(input.size());
        for (const auto item : input)
            strings.push_back(string_view(item));
        return trtmc::Span<const std::string_view>{strings.data(), strings.size()};
    }
    default:
        throw ApiFailure{TRTMC_INVALID_ARGUMENT, "unknown config value kind"};
    }
}

ConvertedConfig::ConvertedConfig(const trtmc_config_view_v1* config) {
    if (config == nullptr)
        return;
    const auto input = checked_span(config->entries, config->count);
    entries_.reserve(input.size());
    string_lists_.reserve(input.size());
    for (const auto& entry : input)
        entries_.push_back({string_view(entry.name), convert(entry.value)});
}

std::string default_runtime_root() {
    static const unsigned char library_location = 0;
    Dl_info info{};
    if (dladdr(&library_location, &info) == 0 || info.dli_fname == nullptr)
        throw ApiFailure{TRTMC_INTERNAL_ERROR, "cannot locate the runtime library directory"};
    return std::filesystem::absolute(info.dli_fname).parent_path().string();
}

struct TaskSnapshot {
    const TaskBinding* binding;
    std::vector<ConfigFieldSnapshot> config_fields;
};

struct ModelState {
    std::unique_ptr<ITask> family;
    BundleInfo info;
    std::vector<TaskSnapshot> tasks;
    std::mutex mutex;
    bool session_active{false};
};

} // namespace trtmc::api

struct trtmc_error {
    trtmc_status code;
    std::string message;
};

struct trtmc_model {
    std::shared_ptr<trtmc::api::ModelState> state;
};

namespace trtmc::api {

trtmc_status store_error(trtmc_error** error, trtmc_status status,
                         std::string_view message) noexcept {
    try {
        *error = new trtmc_error{status, std::string(message)};
    } catch (...) {
        *error = nullptr;
    }
    return status;
}

std::shared_ptr<ModelState> model_owner(const trtmc_model* model) {
    require(model != nullptr, "model is null");
    return model->state;
}

std::mutex& model_mutex(const trtmc_model* model) {
    return model_mutex(model_owner(model));
}

std::mutex& model_mutex(const std::shared_ptr<ModelState>& state) {
    require(state != nullptr, "model state is null");
    return state->mutex;
}

void require_model_idle(const trtmc_model* model) {
    require_model_idle(model_owner(model));
}

void require_model_idle(const std::shared_ptr<ModelState>& state) {
    require(state != nullptr, "model state is null");
    if (state->session_active)
        throw ApiFailure{TRTMC_BUSY, "model has an active session"};
}

ModelSession::ModelSession(const trtmc_model* model) : state_(model_owner(model)) {
    require_model_idle(model);
    state_->session_active = true;
}

ModelSession::~ModelSession() {
    const std::lock_guard<std::mutex> lock(state_->mutex);
    state_->session_active = false;
}

ITask& model_family(const trtmc_model* model) {
    require(model != nullptr, "model is null");
    return *model->state->family;
}

const TaskSnapshot& require_task(const std::shared_ptr<ModelState>& state, std::string_view id,
                                 std::uint32_t major, std::uint32_t minor) {
    require(state != nullptr, "model state is null");
    bool known_id = false;
    for (const auto& task : state->tasks) {
        if (task.binding->id != id)
            continue;
        known_id = true;
        if (task.binding->major == major && task.binding->minor == minor)
            return task;
    }
    if (known_id)
        throw ApiFailure{TRTMC_VERSION_MISMATCH, "requested Task version is not available"};
    throw ApiFailure{TRTMC_UNSUPPORTED, "model does not support the requested Task"};
}

const TaskSnapshot& require_task(const trtmc_model* model, std::string_view id, std::uint32_t major,
                                 std::uint32_t minor) {
    return require_task(model_owner(model), id, major, minor);
}

ITask& require_family(const trtmc_model* model, std::string_view task, std::uint32_t major,
                      std::uint32_t minor) {
    return require_family(model_owner(model), task, major, minor);
}

ITask& require_family(const std::shared_ptr<ModelState>& state, std::string_view task,
                      std::uint32_t major, std::uint32_t minor) {
    require_task(state, task, major, minor);
    return *state->family;
}

void snapshot_tasks(ModelState& state, const internal::IModel& metadata) {
    const auto declarations = metadata.task_info();
    for (const auto bindings :
         {text_continuation_bindings(), text_task_bindings(), image_task_bindings(),
          stream_task_bindings(), features_task_bindings(), audio_task_bindings(),
          numeric_task_bindings(), video_task_bindings(), perception_task_bindings(),
          language_task_bindings(), tracking_task_bindings(), speech_task_bindings(),
          action_task_bindings(), recurrent_task_bindings()}) {
        for (const auto& binding : bindings) {
            const bool declared = std::any_of(
                declarations.begin(), declarations.end(), [&](const internal::TaskInfo& task) {
                    return task.id == binding.id && task.major == binding.major &&
                           task.minor == binding.minor;
                });
            if (!declared || !binding.implemented(*state.family))
                continue;
            TaskSnapshot snapshot{&binding, {}};
            for (const auto& field : metadata.config_fields(binding.id))
                snapshot.config_fields.emplace_back(field);
            state.tasks.push_back(std::move(snapshot));
        }
    }
    for (auto& task : state.tasks) {
        for (auto& field : task.config_fields)
            field.bind_view();
    }
}

trtmc_string_view TRTMC_CALL runtime_version() noexcept {
    return borrowed_string(TRTMC_VERSION_STRING);
}

trtmc_status TRTMC_CALL error_code(const trtmc_error* error) noexcept {
    return error == nullptr ? TRTMC_OK : error->code;
}

trtmc_string_view TRTMC_CALL error_message(const trtmc_error* error) noexcept {
    return error == nullptr ? trtmc_string_view{} : borrowed_string(error->message);
}

void TRTMC_CALL error_release(trtmc_error* error) noexcept {
    delete error;
}
void TRTMC_CALL result_release(trtmc_result* result) noexcept {
    delete result;
}
void TRTMC_CALL model_release(trtmc_model* model) noexcept {
    delete model;
}

trtmc_status TRTMC_CALL model_load(trtmc_string_view bundle_path,
                                   const trtmc_load_options_v1* options, trtmc_model** out,
                                   trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(out != nullptr, "model output is null");
        const auto path = path_string(bundle_path);
        require(!path.empty(), "bundle path is empty");
        std::string runtime_root;
        std::string runtime_cache;
        std::uint64_t kv_bytes = 0;
        bool cuda_graphs = false;
        if (options) {
            require(options->struct_size >= sizeof(trtmc_load_options_v1),
                    "load options are smaller than v1");
            require(options->cuda_graphs <= 1, "cuda_graphs must be zero or one");
            runtime_root = path_string(options->runtime_root);
            runtime_cache = path_string(options->runtime_cache_path);
            kv_bytes = options->kv_cache_size_bytes;
            cuda_graphs = options->cuda_graphs != 0;
        }
        if (runtime_root.empty())
            runtime_root = default_runtime_root();
        auto state = std::make_shared<ModelState>();
        const trtmc::BundleReader reader(path);
        state->info = reader.info();
        state->family =
            trtmc::load_task(reader, runtime_root, kv_bytes, runtime_cache, cuda_graphs);
        const auto* metadata = dynamic_cast<const internal::IModel*>(state->family.get());
        if (metadata == nullptr)
            throw ApiFailure{TRTMC_UNSUPPORTED,
                             "family has not implemented the Task SDK model interface"};
        snapshot_tasks(*state, *metadata);
        *out = new trtmc_model{std::move(state)};
    });
}

trtmc_status TRTMC_CALL model_info(const trtmc_model* model, trtmc_model_info_v1* out,
                                   trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(model != nullptr && out != nullptr, "model or metadata output is null");
        const auto& info = model->state->info;
        *out = {borrowed_string(info.family), borrowed_string(info.backend),
                borrowed_string(info.task)};
    });
}

trtmc_status TRTMC_CALL model_task_count(const trtmc_model* model, std::uint64_t* out,
                                         trtmc_error** error) noexcept {
    if (out)
        *out = 0;
    return guarded(error, [&] {
        require(model != nullptr && out != nullptr, "model or Task count output is null");
        *out = model->state->tasks.size();
    });
}

trtmc_status TRTMC_CALL model_task_info(const trtmc_model* model, std::uint64_t index,
                                        trtmc_task_info_v1* out, trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(model != nullptr && out != nullptr, "model or Task info output is null");
        require(index < model->state->tasks.size(), "Task index is out of range");
        const auto& binding = *model->state->tasks[static_cast<std::size_t>(index)].binding;
        *out = {borrowed_string(binding.id), binding.major, binding.minor};
    });
}

trtmc_status TRTMC_CALL model_get_task_api(const trtmc_model* model, trtmc_string_view task,
                                           std::uint32_t major, std::uint32_t minor,
                                           const trtmc_api_header** out,
                                           trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(out != nullptr, "Task table output is null");
        *out = require_task(model, string_view(task), major, minor).binding->api;
    });
}

trtmc_status TRTMC_CALL config_field_count(const trtmc_model* model, trtmc_string_view task,
                                           std::uint32_t major, std::uint32_t minor,
                                           std::uint64_t* out, trtmc_error** error) noexcept {
    if (out)
        *out = 0;
    return guarded(error, [&] {
        require(out != nullptr, "config count output is null");
        *out = require_task(model, string_view(task), major, minor).config_fields.size();
    });
}

trtmc_status TRTMC_CALL config_field_info(const trtmc_model* model, trtmc_string_view task,
                                          std::uint32_t major, std::uint32_t minor,
                                          std::uint64_t index, trtmc_config_field_v1* out,
                                          trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out != nullptr, "config field output is null");
        const auto& fields = require_task(model, string_view(task), major, minor).config_fields;
        require(index < fields.size(), "config field index is out of range");
        *out = fields[static_cast<std::size_t>(index)].view;
    });
}

const trtmc_core_api_v1 core_api = {{1, 0, sizeof(trtmc_core_api_v1)},
                                    runtime_version,
                                    error_code,
                                    error_message,
                                    error_release,
                                    model_load,
                                    model_release,
                                    model_info,
                                    model_task_count,
                                    model_task_info,
                                    model_get_task_api,
                                    config_field_count,
                                    config_field_info,
                                    result_release,
                                    bundle_open,
                                    bundle_release,
                                    bundle_info,
                                    bundle_read_section,
                                    bytes_result_view,
                                    byok_load,
                                    model_get_lora_api};

static_assert(offsetof(trtmc_core_api_v1, header) == 0);
} // namespace trtmc::api

extern "C" TRTMC_EXPORT trtmc_status TRTMC_CALL trtmc_get_api(std::uint32_t major,
                                                              std::uint32_t minor,
                                                              const trtmc_core_api_v1** out_api) {
    if (out_api == nullptr)
        return TRTMC_INVALID_ARGUMENT;
    *out_api = nullptr;
    if (major != 1 || minor != 0)
        return TRTMC_VERSION_MISMATCH;
    *out_api = &trtmc::api::core_api;
    return TRTMC_OK;
}
