/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/byok.h"

#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tvm/ffi/c_api.h>
#include <vector>

namespace trtmc {

namespace {

struct LoadedKernel {
    TVMFFIObjectHandle module;
    TVMFFIObjectHandle function;
};

class OwnedObject {
  public:
    explicit OwnedObject(TVMFFIObjectHandle handle = nullptr) noexcept : handle_(handle) {}
    ~OwnedObject() {
        if (handle_ != nullptr)
            TVMFFIObjectDecRef(handle_);
    }
    OwnedObject(const OwnedObject&) = delete;
    OwnedObject& operator=(const OwnedObject&) = delete;
    OwnedObject(OwnedObject&& other) noexcept : handle_(other.release()) {}

    TVMFFIObjectHandle get() const noexcept { return handle_; }
    TVMFFIObjectHandle release() noexcept {
        auto* result = handle_;
        handle_ = nullptr;
        return result;
    }

  private:
    TVMFFIObjectHandle handle_;
};

void discard_raised_error() noexcept {
    TVMFFIObjectHandle error = nullptr;
    TVMFFIErrorMoveFromRaised(&error);
    if (error != nullptr)
        TVMFFIObjectDecRef(error);
}

std::vector<LoadedKernel>& loaded_kernels() {
    static auto* kernels = new std::vector<LoadedKernel>;
    return *kernels;
}

std::mutex& loaded_kernels_mutex() {
    static auto* mutex = new std::mutex;
    return *mutex;
}

OwnedObject global_function(const char* name) {
    TVMFFIByteArray key{name, std::char_traits<char>::length(name)};
    TVMFFIObjectHandle function = nullptr;
    if (TVMFFIFunctionGetGlobal(&key, &function) != 0 || function == nullptr) {
        discard_raised_error();
        throw std::runtime_error("TVM-FFI global function is unavailable: " + std::string(name));
    }
    return OwnedObject(function);
}

OwnedObject call_for_object(TVMFFIObjectHandle function, TVMFFIAny* args, int32_t count,
                            int32_t expected, const char* operation) {
    TVMFFIAny result{};
    result.type_index = kTVMFFINone;
    if (TVMFFIFunctionCall(function, args, count, &result) != 0) {
        discard_raised_error();
        throw std::runtime_error(std::string("TVM-FFI ") + operation + " failed");
    }
    if (result.type_index != expected || result.v_obj == nullptr) {
        if (result.type_index >= kTVMFFIStaticObjectBegin && result.v_obj != nullptr)
            TVMFFIObjectDecRef(result.v_obj);
        throw std::runtime_error(std::string("TVM-FFI ") + operation +
                                 " returned the wrong object type");
    }
    return OwnedObject(reinterpret_cast<TVMFFIObjectHandle>(result.v_obj));
}

} // namespace

void load_kernel(const std::string& library, const std::string& function,
                 const std::string& kernel_name) {
    if (library.empty() || function.empty() || kernel_name.empty())
        throw std::invalid_argument("BYOK library, function, and kernel name must be non-empty");

    std::lock_guard<std::mutex> lock(loaded_kernels_mutex());
    auto load = global_function("ffi.ModuleLoadFromFile");
    TVMFFIAny load_arg{};
    load_arg.type_index = kTVMFFIRawStr;
    load_arg.v_c_str = library.c_str();
    auto module = call_for_object(load.get(), &load_arg, 1, kTVMFFIModule, "module load");

    auto get = global_function("ffi.ModuleGetFunction");
    TVMFFIAny get_args[3]{};
    get_args[0].type_index = kTVMFFIModule;
    get_args[0].v_obj = reinterpret_cast<TVMFFIObject*>(module.get());
    get_args[1].type_index = kTVMFFIRawStr;
    get_args[1].v_c_str = function.c_str();
    get_args[2].type_index = kTVMFFIBool;
    get_args[2].v_int64 = 0;
    auto callable = call_for_object(get.get(), get_args, 3, kTVMFFIFunction, "function lookup");

    TVMFFIByteArray global_name{kernel_name.data(), kernel_name.size()};
    if (TVMFFIFunctionSetGlobal(&global_name, callable.get(), 1) != 0) {
        discard_raised_error();
        throw std::runtime_error("TVM-FFI global registration failed: " + kernel_name);
    }
    loaded_kernels().push_back({module.release(), callable.release()});
}

} // namespace trtmc

extern "C" const char* trtmc_load_byok_kernel(const char* library, const char* function,
                                              const char* kernel_name) noexcept {
    static thread_local std::string error;
    try {
        trtmc::load_kernel(library != nullptr ? library : "", function != nullptr ? function : "",
                           kernel_name != nullptr ? kernel_name : "");
        return nullptr;
    } catch (const std::exception& exception) {
        error = exception.what();
    } catch (...) {
        error = "unknown BYOK loader error";
    }
    return error.c_str();
}
