/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#if TRTMC_HAS_TVM_FFI

#include "plugins/tvm_ffi_runtime_bindings.h"

#include "utils/sha256.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <tvm/ffi/c_api.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

constexpr std::string_view kSlotPrefix = "trtmc.slot.";
constexpr std::string_view kSlotIdCharacters =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-@";
constexpr std::uintmax_t kMaxManifestBytes = 1024U * 1024U;

class OwnedTvmObject {
  public:
    OwnedTvmObject() = default;
    explicit OwnedTvmObject(TVMFFIObjectHandle handle) noexcept : handle_(handle) {}
    ~OwnedTvmObject() {
        if (handle_ != nullptr)
            TVMFFIObjectDecRef(handle_);
    }

    OwnedTvmObject(const OwnedTvmObject&) = delete;
    OwnedTvmObject& operator=(const OwnedTvmObject&) = delete;

    OwnedTvmObject(OwnedTvmObject&& other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)) {}
    OwnedTvmObject& operator=(OwnedTvmObject&& other) noexcept {
        if (this != &other) {
            if (handle_ != nullptr)
                TVMFFIObjectDecRef(handle_);
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    TVMFFIObjectHandle get() const noexcept { return handle_; }
    TVMFFIObjectHandle release() noexcept { return std::exchange(handle_, nullptr); }

  private:
    TVMFFIObjectHandle handle_{nullptr};
};

std::recursive_mutex& binding_load_mutex() {
    static std::recursive_mutex mutex;
    return mutex;
}

std::shared_ptr<const TvmFfiBindingSet>& active_binding_set() {
    static std::shared_ptr<const TvmFfiBindingSet> bindings;
    return bindings;
}

void discard_raised_error() noexcept {
    TVMFFIObjectHandle error = nullptr;
    TVMFFIErrorMoveFromRaised(&error);
    if (error != nullptr)
        TVMFFIObjectDecRef(error);
}

OwnedTvmObject get_global_function(std::string_view name) {
    TVMFFIByteArray name_array{name.data(), name.size()};
    TVMFFIObjectHandle function = nullptr;
    if (TVMFFIFunctionGetGlobal(&name_array, &function) != 0) {
        discard_raised_error();
        throw std::runtime_error("TVM-FFI global function lookup failed: " + std::string(name));
    }
    if (function == nullptr)
        throw std::runtime_error("TVM-FFI global function not found: " + std::string(name));
    return OwnedTvmObject(function);
}

OwnedTvmObject call_for_object(TVMFFIObjectHandle function, TVMFFIAny* args, int32_t num_args,
                               int32_t expected_type, const std::string& operation) {
    TVMFFIAny result{};
    result.type_index = kTVMFFINone;
    if (TVMFFIFunctionCall(function, args, num_args, &result) != 0) {
        discard_raised_error();
        throw std::runtime_error("TVM-FFI " + operation + " failed");
    }
    if (result.type_index != expected_type || result.v_obj == nullptr) {
        if (result.type_index >= kTVMFFIStaticObjectBegin && result.v_obj != nullptr)
            TVMFFIObjectDecRef(result.v_obj);
        throw std::runtime_error("TVM-FFI " + operation + " returned the wrong object type");
    }
    return OwnedTvmObject(reinterpret_cast<TVMFFIObjectHandle>(result.v_obj));
}

TvmFfiBoundFunctionPtr load_module_function(const fs::path& library,
                                            const std::string& function_name) {
    auto load_function = get_global_function("ffi.ModuleLoadFromFile");
    const std::string library_string = library.string();
    TVMFFIAny load_arg{};
    load_arg.type_index = kTVMFFIRawStr;
    load_arg.v_c_str = library_string.c_str();
    auto module = call_for_object(load_function.get(), &load_arg, 1, kTVMFFIModule, "module load");

    auto get_function = get_global_function("ffi.ModuleGetFunction");
    TVMFFIAny get_args[3]{};
    get_args[0].type_index = kTVMFFIModule;
    get_args[0].v_obj = reinterpret_cast<TVMFFIObject*>(module.get());
    get_args[1].type_index = kTVMFFIRawStr;
    get_args[1].v_c_str = function_name.c_str();
    get_args[2].type_index = kTVMFFIBool;
    get_args[2].v_int64 = 0;
    auto function = call_for_object(get_function.get(), get_args, 3, kTVMFFIFunction,
                                    "module function lookup for '" + function_name + "'");

    return std::shared_ptr<const TvmFfiBoundFunction>(
        new TvmFfiBoundFunction(module.release(), function.release()));
}

void require_exact_fields(const json& object, std::initializer_list<std::string_view> allowed,
                          const std::string& where) {
    if (!object.is_object())
        throw std::runtime_error(where + " must be an object");

    std::unordered_set<std::string> allowed_fields;
    for (std::string_view field : allowed)
        allowed_fields.emplace(field);
    for (auto it = object.begin(); it != object.end(); ++it) {
        if (allowed_fields.count(it.key()) == 0)
            throw std::runtime_error(where + " contains unknown field '" + it.key() + "'");
    }
    for (std::string_view field : allowed) {
        if (!object.contains(field))
            throw std::runtime_error(where + " is missing field '" + std::string(field) + "'");
    }
}

const std::string& require_string(const json& object, std::string_view field,
                                  const std::string& where) {
    const auto& value = object.at(field);
    if (!value.is_string() || value.get_ref<const std::string&>().empty())
        throw std::runtime_error(where + "." + std::string(field) + " must be a non-empty string");
    return value.get_ref<const std::string&>();
}

void require_schema_version(const json& root, const std::string& where) {
    if (!root.at("schema_version").is_number_integer() ||
        root.at("schema_version").get<int>() != 1) {
        throw std::runtime_error(where + ".schema_version must be 1");
    }
}

bool is_lower_sha256(std::string_view digest) {
    if (digest.size() != 64)
        return false;
    for (char character : digest) {
        if (!((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return true;
}

void require_sha256(std::string_view digest, const std::string& where) {
    if (!is_lower_sha256(digest))
        throw std::runtime_error(where + " must be a lowercase 64-character SHA-256");
}

void require_slot_id(std::string_view id, const std::string& where) {
    if (id.empty())
        throw std::runtime_error(where + " must be non-empty");
    for (char character : id) {
        if (kSlotIdCharacters.find(character) == std::string_view::npos)
            throw std::runtime_error(where + " contains an invalid character");
    }
}

json parse_json(std::string_view source, const std::string& where) {
    try {
        return json::parse(source.begin(), source.end());
    } catch (const json::exception& error) {
        throw std::runtime_error("Invalid " + where + ": " + error.what());
    }
}

std::string read_small_file(const fs::path& path, const std::string& where) {
    std::error_code error;
    const auto size = fs::file_size(path, error);
    if (error)
        throw std::runtime_error("Cannot read " + where + " '" + path.string() +
                                 "': " + error.message());
    if (size == 0 || size > kMaxManifestBytes)
        throw std::runtime_error(where + " must contain between 1 byte and 1 MiB");

    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot open " + where + " '" + path.string() + "'");
    std::string contents(static_cast<std::size_t>(size), '\0');
    input.read(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!input)
        throw std::runtime_error("Cannot read " + where + " '" + path.string() + "'");
    return contents;
}

std::string sha256_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot open kernel library '" + path.string() + "'");

    internal::Sha256 digest;
    std::array<char, 1024U * 1024U> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0)
            digest.update(buffer.data(), static_cast<std::size_t>(count));
    }
    if (!input.eof())
        throw std::runtime_error("Cannot read kernel library '" + path.string() + "'");
    return digest.hex_digest();
}

struct ExpectedSlot {
    std::string id;
    std::string kernel_name;
    std::string abi_sha256;
};

struct ExternalBinding {
    std::string id;
    std::string abi_sha256;
    fs::path library;
    std::string sha256;
    std::string function;
};

struct ValidatedBinding {
    ExpectedSlot expected;
    fs::path library;
    std::string function;
};

std::vector<ExpectedSlot> parse_slot_descriptor(std::string_view source) {
    const json root = parse_json(source, "kernel_slots.json");
    require_exact_fields(root, {"schema_version", "slots"}, "kernel_slots.json");
    require_schema_version(root, "kernel_slots.json");
    if (!root.at("slots").is_array() || root.at("slots").size() != 1)
        throw std::runtime_error("kernel_slots.json.slots must contain exactly one slot");

    std::vector<ExpectedSlot> slots;
    for (std::size_t index = 0; index < root.at("slots").size(); ++index) {
        const auto& entry = root.at("slots").at(index);
        const std::string where = "kernel_slots.json.slots[" + std::to_string(index) + "]";
        require_exact_fields(entry, {"id", "abi_sha256"}, where);
        ExpectedSlot slot{require_string(entry, "id", where), "",
                          require_string(entry, "abi_sha256", where)};
        require_slot_id(slot.id, where + ".id");
        require_sha256(slot.abi_sha256, where + ".abi_sha256");
        slot.kernel_name = std::string(kSlotPrefix) + slot.id;
        slots.push_back(std::move(slot));
    }
    return slots;
}

std::unordered_map<std::string, ExternalBinding>
parse_external_bindings(const fs::path& manifest_path) {
    const json root = parse_json(read_small_file(manifest_path, "kernel bindings manifest"),
                                 "kernel bindings manifest");
    require_exact_fields(root, {"schema_version", "bindings"}, "kernel bindings manifest");
    require_schema_version(root, "kernel bindings manifest");
    if (!root.at("bindings").is_array() || root.at("bindings").empty())
        throw std::runtime_error("kernel bindings manifest.bindings must be a non-empty array");

    std::unordered_map<std::string, ExternalBinding> bindings;
    for (std::size_t index = 0; index < root.at("bindings").size(); ++index) {
        const auto& entry = root.at("bindings").at(index);
        const std::string where =
            "kernel bindings manifest.bindings[" + std::to_string(index) + "]";
        require_exact_fields(entry, {"id", "abi_sha256", "library", "sha256", "function"}, where);
        ExternalBinding binding{
            require_string(entry, "id", where), require_string(entry, "abi_sha256", where),
            fs::path(require_string(entry, "library", where)),
            require_string(entry, "sha256", where), require_string(entry, "function", where)};
        require_slot_id(binding.id, where + ".id");
        require_sha256(binding.abi_sha256, where + ".abi_sha256");
        require_sha256(binding.sha256, where + ".sha256");
        if (binding.library.is_absolute())
            throw std::runtime_error(where + ".library must be relative to the manifest");
        if (!bindings.emplace(binding.id, std::move(binding)).second) {
            throw std::runtime_error("kernel bindings manifest contains duplicate id '" +
                                     require_string(entry, "id", where) + "'");
        }
    }
    return bindings;
}

ValidatedBinding
validate_external_binding(const ExpectedSlot& expected,
                          const std::unordered_map<std::string, ExternalBinding>& bindings,
                          const fs::path& manifest_path) {
    const auto binding = bindings.find(expected.id);
    if (binding == bindings.end())
        throw std::runtime_error("Kernel bindings manifest is missing slot '" + expected.id + "'");
    if (binding->second.abi_sha256 != expected.abi_sha256)
        throw std::runtime_error("Kernel ABI SHA-256 mismatch for slot '" + expected.id + "'");

    std::error_code error;
    const fs::path library =
        fs::canonical(manifest_path.parent_path() / binding->second.library, error);
    if (error) {
        throw std::runtime_error("Cannot resolve kernel library for slot '" + expected.id +
                                 "': " + error.message());
    }
    if (!fs::is_regular_file(library, error) || error)
        throw std::runtime_error("Kernel library is not a regular file: " + library.string());
    if (sha256_file(library) != binding->second.sha256)
        throw std::runtime_error("Kernel library SHA-256 mismatch for slot '" + expected.id + "'");
    return ValidatedBinding{expected, library, binding->second.function};
}

} // namespace

struct TvmFfiBindingSet::Impl {
    std::unordered_map<std::string, TvmFfiBoundFunctionPtr> functions;
};

TvmFfiBoundFunction::~TvmFfiBoundFunction() {
    if (function_ != nullptr)
        TVMFFIObjectDecRef(function_);
    if (module_ != nullptr)
        TVMFFIObjectDecRef(module_);
}

TvmFfiBindingSet::TvmFfiBindingSet(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

TvmFfiBindingSet::~TvmFfiBindingSet() = default;

std::shared_ptr<const TvmFfiBindingSet>
TvmFfiBindingSet::Load(std::string_view slot_descriptor_json, const std::string& bindings_path) {
    const auto expected_slots = parse_slot_descriptor(slot_descriptor_json);
    if (bindings_path.empty())
        throw std::runtime_error("A slot-ready bundle requires a kernel bindings manifest");

    std::error_code error;
    fs::path manifest_path = fs::canonical(fs::path(bindings_path), error);
    if (error) {
        throw std::runtime_error("Cannot resolve kernel bindings manifest '" + bindings_path +
                                 "': " + error.message());
    }
    const auto external_bindings = parse_external_bindings(manifest_path);
    if (external_bindings.size() != expected_slots.size()) {
        throw std::runtime_error("Kernel bindings manifest must bind every slot exactly once");
    }

    // Validate the entire manifest, including every DSO digest, before loading a module.
    std::vector<ValidatedBinding> validated;
    validated.reserve(expected_slots.size());
    for (const auto& expected : expected_slots)
        validated.push_back(validate_external_binding(expected, external_bindings, manifest_path));

    auto impl = std::make_unique<Impl>();
    for (const auto& binding : validated) {
        impl->functions.emplace(binding.expected.kernel_name,
                                load_module_function(binding.library, binding.function));
    }
    return std::shared_ptr<const TvmFfiBindingSet>(new TvmFfiBindingSet(std::move(impl)));
}

TvmFfiBoundFunctionPtr TvmFfiBindingSet::find(std::string_view kernel_name) const noexcept {
    for (const auto& entry : impl_->functions) {
        if (entry.first == kernel_name) {
            captured_.store(true);
            return entry.second;
        }
    }
    return nullptr;
}

std::size_t TvmFfiBindingSet::size() const noexcept {
    return impl_->functions.size();
}

bool is_runtime_tvm_ffi_kernel_name(std::string_view kernel_name) noexcept {
    return kernel_name.size() > kSlotPrefix.size() &&
           kernel_name.compare(0, kSlotPrefix.size(), kSlotPrefix) == 0;
}

TvmFfiBoundFunctionPtr active_tvm_ffi_binding(std::string_view kernel_name) noexcept {
    const auto bindings = std::atomic_load(&active_binding_set());
    return bindings == nullptr ? nullptr : bindings->find(kernel_name);
}

TvmFfiBoundFunctionPtr resolve_global_tvm_ffi_function(std::string_view name) noexcept {
    try {
        auto function = get_global_function(name);
        return std::shared_ptr<const TvmFfiBoundFunction>(
            new TvmFfiBoundFunction(nullptr, function.release()));
    } catch (...) {
        return nullptr;
    }
}

ScopedTvmFfiBindings::ScopedTvmFfiBindings(std::shared_ptr<const TvmFfiBindingSet> bindings)
    : lock_(binding_load_mutex()) {
    if (bindings == nullptr)
        throw std::invalid_argument("Cannot activate an empty TVM-FFI binding set");
    if (std::atomic_load(&active_binding_set()) != nullptr)
        throw std::logic_error("TVM-FFI binding scopes cannot be nested");
    std::atomic_store(&active_binding_set(), std::move(bindings));
    active_ = true;
}

ScopedTvmFfiBindings::~ScopedTvmFfiBindings() noexcept {
    if (active_)
        std::atomic_store(&active_binding_set(), std::shared_ptr<const TvmFfiBindingSet>{});
}

} // namespace trtmc

#endif // TRTMC_HAS_TVM_FFI
