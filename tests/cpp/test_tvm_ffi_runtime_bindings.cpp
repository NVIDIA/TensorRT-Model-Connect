/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#if defined(TRTMC_TVM_FFI_BINDINGS_TEST_DSO)

#include <tvm/ffi/function.h>

namespace {

int run_a_impl(int value) {
    return value + 1;
}

int run_b_impl(int value) {
    return value + 2;
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_a, run_a_impl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_b, run_b_impl);

#else

#include "plugins/tvm_ffi_runtime_bindings.h"
#include "utils/sha256.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tvm/ffi/c_api.h>

namespace {

namespace fs = std::filesystem;

int failures = 0;

void check(bool condition, const std::string& name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

class TemporaryDirectory {
  public:
    TemporaryDirectory() {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ =
            fs::temp_directory_path() / ("trtmc-tvm-ffi-runtime-bindings-" + std::to_string(nonce));
        fs::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        fs::remove_all(path_, error);
    }

    const fs::path& path() const noexcept { return path_; }

  private:
    fs::path path_;
};

std::string sha256_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    trtmc::internal::Sha256 digest;
    std::array<char, 4096> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        if (input.gcount() > 0)
            digest.update(buffer.data(), static_cast<std::size_t>(input.gcount()));
    }
    return digest.hex_digest();
}

void write_text(const fs::path& path, std::string_view contents) {
    std::ofstream output(path, std::ios::binary);
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!output)
        throw std::runtime_error("Failed to write " + path.string());
}

std::string descriptor_json(std::string_view abi, bool unknown_field = false) {
    return std::string("{\"schema_version\":1,\"slots\":[{") +
           "\"id\":\"test.slot@1\",\"abi_sha256\":\"" + std::string(abi) + "\"" +
           (unknown_field ? ",\"unexpected\":true" : "") + "}]}";
}

std::string binding_entry(std::string_view id, std::string_view abi, std::string_view library,
                          std::string_view digest, std::string_view function) {
    return std::string("{\"id\":\"") + std::string(id) + "\",\"abi_sha256\":\"" + std::string(abi) +
           "\",\"library\":\"" + std::string(library) + "\",\"sha256\":\"" + std::string(digest) +
           "\",\"function\":\"" + std::string(function) + "\"}";
}

fs::path write_manifest(const fs::path& directory, std::string_view name,
                        std::string_view bindings) {
    const fs::path path = directory / name;
    write_text(path,
               std::string("{\"schema_version\":1,\"bindings\":[") + std::string(bindings) + "]}");
    return path;
}

int invoke(const trtmc::TvmFfiBoundFunctionPtr& function, int value) {
    TVMFFIAny argument{};
    argument.type_index = kTVMFFIInt;
    argument.v_int64 = value;
    TVMFFIAny result{};
    result.type_index = kTVMFFINone;
    if (TVMFFIFunctionCall(function->handle(), &argument, 1, &result) != 0)
        throw std::runtime_error("Bound test function call failed");
    if (result.type_index != kTVMFFIInt)
        throw std::runtime_error("Bound test function returned the wrong type");
    return static_cast<int>(result.v_int64);
}

void expect_failure(const std::function<void()>& operation, std::string_view message,
                    const std::string& name) {
    try {
        operation();
        check(false, name);
    } catch (const std::exception& error) {
        check(std::string_view(error.what()).find(message) != std::string_view::npos, name);
    }
}

void test_bindings() {
    TemporaryDirectory temporary;
    const fs::path kernel_a = temporary.path() / "kernel_a.so";
    const fs::path kernel_b = temporary.path() / "kernel_b.so";
    fs::copy_file(TRTMC_TEST_TVM_FFI_BINDINGS_DSO, kernel_a);
    fs::copy_file(TRTMC_TEST_TVM_FFI_BINDINGS_DSO, kernel_b);
    const std::string digest = sha256_file(kernel_a);
    const std::string abi(64, 'a');
    const std::string descriptor = descriptor_json(abi);

    const fs::path manifest_a =
        write_manifest(temporary.path(), "a.json",
                       binding_entry("test.slot@1", abi, "kernel_a.so", digest, "run_a"));
    const fs::path manifest_b =
        write_manifest(temporary.path(), "b.json",
                       binding_entry("test.slot@1", abi, "kernel_b.so", digest, "run_b"));

    auto bindings_a = trtmc::TvmFfiBindingSet::Load(descriptor, manifest_a.string());
    check(bindings_a->size() == 1, "one binding in manifest A");
    check(!bindings_a->was_captured(), "binding starts uncaptured");

    trtmc::TvmFfiBoundFunctionPtr function_a;
    {
        trtmc::ScopedTvmFfiBindings scope(bindings_a);
        check(trtmc::active_tvm_ffi_binding("trtmc.slot.unknown") == nullptr,
              "unknown lookup stays empty");
        check(!bindings_a->was_captured(), "unknown lookup does not mark capture");
        function_a = trtmc::active_tvm_ffi_binding("trtmc.slot.test.slot@1");
        check(function_a != nullptr && invoke(function_a, 40) == 41, "activate binding A");
        check(bindings_a->was_captured(), "successful lookup marks capture");
    }
    check(trtmc::active_tvm_ffi_binding("trtmc.slot.test.slot@1") == nullptr,
          "scope clears active bindings");
    bindings_a.reset();

    auto bindings_b = trtmc::TvmFfiBindingSet::Load(descriptor, manifest_b.string());
    check(bindings_b->size() == 1, "one binding in manifest B");
    trtmc::TvmFfiBoundFunctionPtr function_b;
    {
        trtmc::ScopedTvmFfiBindings scope(bindings_b);
        function_b = trtmc::active_tvm_ffi_binding("trtmc.slot.test.slot@1");
        check(function_b != nullptr && invoke(function_b, 40) == 42, "activate binding B");
    }
    bindings_b.reset();
    check(invoke(function_a, 40) == 41, "binding A remains valid after loading B");

    const fs::path missing = write_manifest(temporary.path(), "missing.json", "");
    expect_failure([&] { trtmc::TvmFfiBindingSet::Load(descriptor, missing.string()); },
                   "non-empty array", "missing binding fails");

    const fs::path extra =
        write_manifest(temporary.path(), "extra.json",
                       binding_entry("test.slot@1", abi, "kernel_a.so", digest, "run_a") + "," +
                           binding_entry("extra.slot@1", abi, "kernel_a.so", digest, "run_a"));
    expect_failure([&] { trtmc::TvmFfiBindingSet::Load(descriptor, extra.string()); },
                   "bind every slot exactly once", "extra binding fails");

    const std::string wrong_abi(64, 'b');
    const fs::path incompatible =
        write_manifest(temporary.path(), "incompatible.json",
                       binding_entry("test.slot@1", wrong_abi, "kernel_a.so", digest, "run_a"));
    expect_failure([&] { trtmc::TvmFfiBindingSet::Load(descriptor, incompatible.string()); },
                   "Kernel ABI SHA-256 mismatch", "incompatible ABI fails");

    const fs::path changed = write_manifest(
        temporary.path(), "changed.json",
        binding_entry("test.slot@1", abi, "kernel_a.so", std::string(64, '0'), "run_a"));
    expect_failure([&] { trtmc::TvmFfiBindingSet::Load(descriptor, changed.string()); },
                   "Kernel library SHA-256 mismatch", "changed DSO fails");

    expect_failure(
        [&] { trtmc::TvmFfiBindingSet::Load(descriptor_json(abi, true), manifest_a.string()); },
        "unknown field", "slot descriptor is strict");
}

} // namespace

int main() {
    try {
        test_bindings();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " FAILED\n";
        return 1;
    }
    std::cerr << "All TVM-FFI runtime binding tests passed.\n";
    return 0;
}

#endif
