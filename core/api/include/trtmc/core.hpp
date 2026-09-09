/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/span.h"
#include "trtmc/trtmc.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace trtmc {

class Error : public std::runtime_error {
  public:
    Error(trtmc_status code, std::string message)
        : std::runtime_error(std::move(message)), code_(code) {}
    trtmc_status code() const noexcept { return code_; }

  private:
    trtmc_status code_;
};

enum class ConfigKind : std::uint32_t {
    I64 = TRTMC_CONFIG_I64,
    F64 = TRTMC_CONFIG_F64,
    Bool = TRTMC_CONFIG_BOOL,
    String = TRTMC_CONFIG_STRING,
    I64List = TRTMC_CONFIG_I64_LIST,
    F64List = TRTMC_CONFIG_F64_LIST,
    StringList = TRTMC_CONFIG_STRING_LIST,
};

class ConfigValue {
  public:
    using Storage = std::variant<std::int64_t, double, bool, std::string, std::vector<std::int64_t>,
                                 std::vector<double>, std::vector<std::string>>;

    template <typename T,
              std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, int> = 0>
    ConfigValue(T value) : storage_(integer(value)) {}
    ConfigValue(double value) : storage_(value) {}
    ConfigValue(bool value) : storage_(value) {}
    ConfigValue(std::string value) : storage_(std::move(value)) {}
    ConfigValue(std::string_view value) : storage_(std::string(value)) {}
    ConfigValue(const char* value) : storage_(checked_string(value)) {}
    ConfigValue(std::vector<std::int64_t> value) : storage_(std::move(value)) {}
    ConfigValue(std::vector<double> value) : storage_(std::move(value)) {}
    ConfigValue(std::vector<std::string> value) : storage_(std::move(value)) {}
    ConfigValue(const ConfigValue&) = default;
    ConfigValue(ConfigValue&&) noexcept = default;
    ConfigValue& operator=(ConfigValue other) noexcept {
        storage_.swap(other.storage_);
        return *this;
    }

    template <typename T>
    const T& get() const {
        return std::get<T>(storage_);
    }

    ConfigKind kind() const noexcept {
        constexpr ConfigKind kinds[] = {
            ConfigKind::I64,     ConfigKind::F64,     ConfigKind::Bool,      ConfigKind::String,
            ConfigKind::I64List, ConfigKind::F64List, ConfigKind::StringList};
        return kinds[storage_.index()];
    }

  private:
    template <typename T>
    static std::int64_t integer(T value) {
        if constexpr (std::is_unsigned_v<T>) {
            if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
                throw std::out_of_range("config integer does not fit int64");
        }
        return static_cast<std::int64_t>(value);
    }

    static std::string checked_string(const char* value) {
        if (value == nullptr)
            throw std::invalid_argument("config string must not be null");
        return std::string(value);
    }

    Storage storage_;
};

struct ConfigEntry {
    std::string name;
    ConfigValue value;
};

class Config {
  public:
    Config() = default;
    Config(std::initializer_list<ConfigEntry> entries) : entries_(entries) {}
    explicit Config(std::vector<ConfigEntry> entries) : entries_(std::move(entries)) {}

    const std::vector<ConfigEntry>& entries() const noexcept { return entries_; }
    void add(std::string name, ConfigValue value) {
        entries_.push_back({std::move(name), std::move(value)});
    }

    // This object owns the C descriptor arrays, but borrows their values from
    // Config. Keep Config alive and unchanged for the duration of the C call.
    class CEntries {
      public:
        CEntries(const CEntries&) = delete;
        CEntries& operator=(const CEntries&) = delete;
        CEntries(CEntries&&) noexcept = default;
        CEntries& operator=(CEntries&&) noexcept = default;
        const trtmc_config_entry_v1* data() const noexcept { return entries_.data(); }
        std::size_t size() const noexcept { return entries_.size(); }
        trtmc_config_view_v1 view() const noexcept { return {data(), size()}; }

      private:
        friend class Config;
        explicit CEntries(const Config& config)
            : entries_(config.entries_.size()), strings_(config.entries_.size()) {
            for (std::size_t i = 0; i < entries_.size(); ++i) {
                const auto& source = config.entries_[i];
                auto& entry = entries_[i];
                entry.name = {source.name.data(), source.name.size()};
                entry.value.kind = static_cast<std::uint32_t>(source.value.kind());
                switch (source.value.kind()) {
                case ConfigKind::I64:
                    entry.value.as.i64 = source.value.get<std::int64_t>();
                    break;
                case ConfigKind::F64:
                    entry.value.as.f64 = source.value.get<double>();
                    break;
                case ConfigKind::Bool:
                    entry.value.as.boolean = source.value.get<bool>() ? 1U : 0U;
                    break;
                case ConfigKind::String: {
                    const auto& value = source.value.get<std::string>();
                    entry.value.as.string = {value.data(), value.size()};
                    break;
                }
                case ConfigKind::I64List: {
                    const auto& value = source.value.get<std::vector<std::int64_t>>();
                    entry.value.as.i64_list = {value.data(), value.size()};
                    break;
                }
                case ConfigKind::F64List: {
                    const auto& value = source.value.get<std::vector<double>>();
                    entry.value.as.f64_list = {value.data(), value.size()};
                    break;
                }
                case ConfigKind::StringList:
                    for (const auto& value : source.value.get<std::vector<std::string>>())
                        strings_[i].push_back({value.data(), value.size()});
                    entry.value.as.string_list = {strings_[i].data(), strings_[i].size()};
                    break;
                }
            }
        }
        std::vector<trtmc_config_entry_v1> entries_;
        std::vector<std::vector<trtmc_string_view>> strings_;
    };

    CEntries c_entries() const { return CEntries(*this); }

  private:
    std::vector<ConfigEntry> entries_;
};

struct LoadOptions {
    // Empty lets the runtime use its installed library directory.
    std::string runtime_root;
    std::uint64_t kv_cache_size_bytes{0};
    std::string runtime_cache_path;
    bool cuda_graphs{false};
};

struct ModelInfo {
    std::string family;
    std::string backend;
    std::string bundle_task;
};

struct TaskInfo {
    std::string id;
    std::uint32_t major;
    std::uint32_t minor;
};

struct ConfigField {
    std::string name;
    ConfigKind kind;
    std::optional<ConfigValue> default_value;
    std::string description;
};

namespace detail {

inline trtmc_string_view c_string(std::string_view value) noexcept {
    return {value.data(), value.size()};
}

inline std::string_view string_view(trtmc_string_view value) {
    if (value.size == 0)
        return {};
    if (value.data == nullptr || value.size > std::numeric_limits<std::size_t>::max())
        throw Error(TRTMC_INTERNAL_ERROR, "runtime returned an invalid string view");
    return {value.data, static_cast<std::size_t>(value.size)};
}

inline void check(const trtmc_core_api_v1& api, trtmc_status status, trtmc_error* error) {
    const auto release = [&api](trtmc_error* value) { api.error_release(value); };
    std::unique_ptr<trtmc_error, decltype(release)> owner(error, release);
    if (status != TRTMC_OK) {
        const auto message =
            error == nullptr ? std::string_view{} : string_view(api.error_message(error));
        throw Error(status, message.empty() ? "Model Connect call failed" : std::string(message));
    }
}

inline const trtmc_core_api_v1& core_api() {
    static const trtmc_core_api_v1* api = [] {
        const trtmc_core_api_v1* result = nullptr;
        const auto status = trtmc_get_api(1, 0, &result);
        if (status != TRTMC_OK)
            throw Error(status, "Model Connect core API v1.0 is unavailable");
        if (result == nullptr || result->header.major != 1 || result->header.minor != 0 ||
            result->header.byte_size < sizeof(trtmc_core_api_v1))
            throw Error(TRTMC_VERSION_MISMATCH,
                        "Model Connect returned an incompatible core table");
        return result;
    }();
    return *api;
}

struct ModelState {
    explicit ModelState(const trtmc_core_api_v1& table) noexcept : api(table) {}
    ~ModelState() { api.model_release(handle); }
    ModelState(const ModelState&) = delete;
    ModelState& operator=(const ModelState&) = delete;
    const trtmc_core_api_v1& api;
    trtmc_model* handle{nullptr};
};

class ResultOwner {
  public:
    ResultOwner(std::shared_ptr<ModelState> state, trtmc_result* result) noexcept
        : api_(&state->api), state_(std::move(state)), result_(result) {}
    ResultOwner(const trtmc_core_api_v1& api, trtmc_result* result) noexcept
        : api_(&api), result_(result) {}
    ~ResultOwner() { reset(); }
    ResultOwner(const ResultOwner&) = delete;
    ResultOwner& operator=(const ResultOwner&) = delete;
    ResultOwner(ResultOwner&& other) noexcept
        : api_(other.api_), state_(std::move(other.state_)),
          result_(std::exchange(other.result_, nullptr)) {}
    ResultOwner& operator=(ResultOwner&& other) noexcept {
        if (this != &other) {
            reset();
            api_ = other.api_;
            state_ = std::move(other.state_);
            result_ = std::exchange(other.result_, nullptr);
        }
        return *this;
    }
    trtmc_result* get() const noexcept { return result_; }
    const trtmc_core_api_v1& api() const noexcept { return *api_; }
    const std::shared_ptr<ModelState>& state() const noexcept { return state_; }
    void reset(trtmc_result* result = nullptr) noexcept {
        if (result_ != nullptr)
            api_->result_release(result_);
        result_ = result;
    }

  private:
    const trtmc_core_api_v1* api_;
    std::shared_ptr<ModelState> state_;
    trtmc_result* result_;
};

struct ViewResultAccess;

template <class View>
class ViewResult {
  public:
    using ReadView = trtmc_status(TRTMC_CALL*)(const trtmc_result*, View*, trtmc_error**);
    ViewResult(ResultOwner owner, ReadView read) noexcept : owner_(std::move(owner)), read_(read) {}
    ViewResult(const ViewResult&) = delete;
    ViewResult& operator=(const ViewResult&) = delete;
    ViewResult(ViewResult&&) noexcept = default;
    ViewResult& operator=(ViewResult&&) noexcept = default;
    // All pointers in the typed view borrow this result, not the original model.
    View view() const {
        View out{};
        if (!owner_.get())
            return out;
        trtmc_error* error = nullptr;
        const auto status = read_(owner_.get(), &out, &error);
        check(owner_.api(), status, error);
        return out;
    }

  private:
    friend struct ViewResultAccess;
    ResultOwner owner_;
    ReadView read_;
};

struct ViewResultAccess {
    template <class View>
    static trtmc_result* get(const ViewResult<View>& result) noexcept {
        return result.owner_.get();
    }
};

inline ConfigValue copy_value(const trtmc_config_value_v1& value) {
    switch (value.kind) {
    case TRTMC_CONFIG_I64:
        return ConfigValue(value.as.i64);
    case TRTMC_CONFIG_F64:
        return ConfigValue(value.as.f64);
    case TRTMC_CONFIG_BOOL:
        if (value.as.boolean > 1)
            throw Error(TRTMC_INTERNAL_ERROR, "runtime returned an invalid config bool");
        return ConfigValue(value.as.boolean != 0);
    case TRTMC_CONFIG_STRING:
        return ConfigValue(std::string(string_view(value.as.string)));
    case TRTMC_CONFIG_I64_LIST: {
        const auto list = value.as.i64_list;
        if (list.size == 0)
            return ConfigValue(std::vector<std::int64_t>{});
        if (list.data == nullptr || list.size > std::numeric_limits<std::size_t>::max())
            throw Error(TRTMC_INTERNAL_ERROR, "runtime returned an invalid integer list");
        return ConfigValue(std::vector<std::int64_t>(list.data, list.data + list.size));
    }
    case TRTMC_CONFIG_F64_LIST: {
        const auto list = value.as.f64_list;
        if (list.size == 0)
            return ConfigValue(std::vector<double>{});
        if (list.data == nullptr || list.size > std::numeric_limits<std::size_t>::max())
            throw Error(TRTMC_INTERNAL_ERROR, "runtime returned an invalid floating-point list");
        return ConfigValue(std::vector<double>(list.data, list.data + list.size));
    }
    case TRTMC_CONFIG_STRING_LIST: {
        const auto list = value.as.string_list;
        if (list.size != 0 && list.data == nullptr)
            throw Error(TRTMC_INTERNAL_ERROR, "runtime returned an invalid string list");
        std::vector<std::string> values;
        for (std::uint64_t i = 0; i < list.size; ++i)
            values.emplace_back(string_view(list.data[i]));
        return ConfigValue(std::move(values));
    }
    default:
        throw Error(TRTMC_INTERNAL_ERROR, "runtime returned an unknown config kind");
    }
}

inline std::vector<ConfigField> config_fields(const std::shared_ptr<ModelState>& model,
                                              std::string_view task, std::uint32_t major,
                                              std::uint32_t minor) {
    std::uint64_t count = 0;
    trtmc_error* error = nullptr;
    auto status =
        model->api.config_field_count(model->handle, c_string(task), major, minor, &count, &error);
    check(model->api, status, error);
    std::vector<ConfigField> fields;
    for (std::uint64_t i = 0; i < count; ++i) {
        trtmc_config_field_v1 field{};
        error = nullptr;
        status = model->api.config_field_info(model->handle, c_string(task), major, minor, i,
                                              &field, &error);
        check(model->api, status, error);
        std::optional<ConfigValue> default_value;
        if (field.has_fixed_default)
            default_value = copy_value(field.default_value);
        fields.push_back({std::string(string_view(field.name)), static_cast<ConfigKind>(field.kind),
                          std::move(default_value), std::string(string_view(field.description))});
    }
    return fields;
}

} // namespace detail

inline std::string runtime_version() {
    return std::string(detail::string_view(detail::core_api().runtime_version()));
}

class LoraManager;

class Model {
  public:
    static Model load(std::string_view bundle_path, const LoadOptions& options = {}) {
        const auto& api = detail::core_api();
        auto state = std::make_shared<detail::ModelState>(api);
        trtmc_load_options_v1 wire{
            sizeof(trtmc_load_options_v1), detail::c_string(options.runtime_root),
            options.kv_cache_size_bytes, detail::c_string(options.runtime_cache_path),
            options.cuda_graphs ? 1U : 0U};
        trtmc_error* error = nullptr;
        const auto status =
            api.model_load(detail::c_string(bundle_path), &wire, &state->handle, &error);
        detail::check(api, status, error);
        return Model(std::move(state));
    }

    ModelInfo info() const {
        trtmc_model_info_v1 info{};
        trtmc_error* error = nullptr;
        const auto status = state_->api.model_info(state_->handle, &info, &error);
        detail::check(state_->api, status, error);
        return {std::string(detail::string_view(info.family)),
                std::string(detail::string_view(info.backend)),
                std::string(detail::string_view(info.bundle_task))};
    }

    std::vector<TaskInfo> tasks() const {
        std::uint64_t count = 0;
        trtmc_error* error = nullptr;
        auto status = state_->api.model_task_count(state_->handle, &count, &error);
        detail::check(state_->api, status, error);
        std::vector<TaskInfo> tasks;
        for (std::uint64_t i = 0; i < count; ++i) {
            trtmc_task_info_v1 info{};
            error = nullptr;
            status = state_->api.model_task_info(state_->handle, i, &info, &error);
            detail::check(state_->api, status, error);
            tasks.push_back({std::string(detail::string_view(info.id)), info.major, info.minor});
        }
        return tasks;
    }

    template <typename Task>
    bool supports() const {
        const trtmc_api_header* table = nullptr;
        trtmc_error* error = nullptr;
        const auto status =
            state_->api.model_get_task_api(state_->handle, detail::c_string(Task::kTask),
                                           Task::kMajor, Task::kMinor, &table, &error);
        if (status == TRTMC_UNSUPPORTED || status == TRTMC_VERSION_MISMATCH) {
            state_->api.error_release(error);
            return false;
        }
        detail::check(state_->api, status, error);
        Task::validate_table(table);
        return true;
    }

    template <typename Task>
    Task task() const {
        const trtmc_api_header* table = nullptr;
        trtmc_error* error = nullptr;
        const auto status =
            state_->api.model_get_task_api(state_->handle, detail::c_string(Task::kTask),
                                           Task::kMajor, Task::kMinor, &table, &error);
        detail::check(state_->api, status, error);
        Task::validate_table(table);
        return Task(state_, table);
    }

    LoraManager lora_adapters() const;

  private:
    friend class LoraManager;
    explicit Model(std::shared_ptr<detail::ModelState> state) noexcept : state_(std::move(state)) {}
    std::shared_ptr<detail::ModelState> state_;
};

struct TextContinuationRequest {
    std::variant<std::string, std::vector<std::int32_t>> prefix;
};

struct TranscriptionSegmentView {
    double start_seconds;
    double end_seconds;
    std::string_view text;
    Span<const std::int32_t> token_ids;
};

// Borrowed from its containing result or stream event owner.
struct TextResultView {
    std::string_view text;
    Span<const std::int32_t> token_ids;
    double setup_ms;
    double prefill_ms;
    double decode_ms;
    std::vector<TranscriptionSegmentView> segments;
};

namespace detail {
struct TextResultAccess;
inline TextResultView text_result_view(const trtmc_text_result_view_v1& view) {
    TextResultView output{string_view(view.text),
                          {view.token_ids.data, static_cast<std::size_t>(view.token_ids.size)},
                          view.setup_ms,
                          view.prefill_ms,
                          view.decode_ms,
                          {}};
    for (std::uint64_t i = 0; i < view.segment_count; ++i) {
        const auto& segment = view.segments[i];
        output.segments.push_back(
            {segment.start_seconds,
             segment.end_seconds,
             string_view(segment.text),
             {segment.token_ids.data, static_cast<std::size_t>(segment.token_ids.size)}});
    }
    return output;
}
} // namespace detail

class TextContinuationResult {
  public:
    TextContinuationResult(const TextContinuationResult&) = delete;
    TextContinuationResult& operator=(const TextContinuationResult&) = delete;
    TextContinuationResult(TextContinuationResult&& other) noexcept
        : owner_(std::move(other.owner_)), view_(std::exchange(other.view_, {})) {}
    TextContinuationResult& operator=(TextContinuationResult&& other) noexcept {
        if (this != &other) {
            owner_ = std::move(other.owner_);
            view_ = std::exchange(other.view_, {});
        }
        return *this;
    }

    // These views borrow the result. Keep this owner alive while using them.
    std::string_view text() const { return detail::string_view(view_.text); }
    Span<const std::int32_t> token_ids() const noexcept {
        return {view_.token_ids.data, static_cast<std::size_t>(view_.token_ids.size)};
    }
    double setup_ms() const noexcept { return view_.setup_ms; }
    double prefill_ms() const noexcept { return view_.prefill_ms; }
    double decode_ms() const noexcept { return view_.decode_ms; }
    std::vector<TranscriptionSegmentView> segments() const {
        std::vector<TranscriptionSegmentView> segments;
        for (std::uint64_t i = 0; i < view_.segment_count; ++i) {
            const auto& item = view_.segments[i];
            segments.push_back(
                {item.start_seconds,
                 item.end_seconds,
                 detail::string_view(item.text),
                 {item.token_ids.data, static_cast<std::size_t>(item.token_ids.size)}});
        }
        return segments;
    }

  private:
    friend class TextContinuation;
    friend struct detail::TextResultAccess;
    TextContinuationResult(std::shared_ptr<detail::ModelState> state, trtmc_result* result) noexcept
        : owner_(std::move(state), result) {}
    detail::ResultOwner owner_;
    trtmc_text_result_view_v1 view_{};
};

namespace detail {
struct TextResultAccess {
    static TextContinuationResult adopt(std::shared_ptr<ModelState> state,
                                        trtmc_result* result) noexcept {
        return TextContinuationResult(std::move(state), result);
    }
    static trtmc_text_result_view_v1& view(TextContinuationResult& result) noexcept {
        return result.view_;
    }
};
} // namespace detail

class TextContinuation {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_CONTINUATION;
    static constexpr std::uint32_t kMajor = 1;
    static constexpr std::uint32_t kMinor = 0;

    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }

    TextContinuationResult run(const TextContinuationRequest& request,
                               const Config& config = {}) const {
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_text_continuation_request_v1 input{};
        if (const auto* text = std::get_if<std::string>(&request.prefix)) {
            input.prefix.kind = TRTMC_TEXT_UTF8;
            input.prefix.as.text = detail::c_string(*text);
        } else {
            const auto& ids = std::get<std::vector<std::int32_t>>(request.prefix);
            input.prefix.kind = TRTMC_TEXT_TOKEN_IDS;
            input.prefix.as.token_ids = {ids.data(), ids.size()};
        }
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        auto status = api_->run(state_->handle, &input, &options, &raw, &error);
        TextContinuationResult result(state_, raw);
        detail::check(state_->api, status, error);
        error = nullptr;
        status = api_->result_view(raw, &result.view_, &error);
        detail::check(state_->api, status, error);
        return result;
    }

    static void validate_table(const trtmc_api_header* table) {
        if (table == nullptr || table->major != kMajor || table->minor != kMinor ||
            table->byte_size < sizeof(trtmc_text_continuation_api_v1))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible text-continuation API table");
    }

  private:
    friend class Model;
    TextContinuation(std::shared_ptr<detail::ModelState> state,
                     const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_text_continuation_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_text_continuation_api_v1* api_;
};

} // namespace trtmc
