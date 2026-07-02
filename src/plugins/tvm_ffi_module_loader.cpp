/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Load TVM-FFI modules from .so files — pure C++ via TVM-FFI C ABI.

#if TRTMC_HAS_TVM_FFI

#include "plugins/tvm_ffi_module_loader.h"

#include <cstring>
#include <iostream>
#include <string>
#include <tvm/ffi/c_api.h>

namespace trtmc {

namespace {

// Helper: look up a TVM-FFI global function by name.
TVMFFIObjectHandle get_global(const char* name) {
    TVMFFIByteArray name_arr;
    name_arr.data = name;
    name_arr.size = static_cast<int64_t>(std::strlen(name));
    TVMFFIObjectHandle handle = nullptr;
    if (TVMFFIFunctionGetGlobal(&name_arr, &handle) != 0) {
        return nullptr;
    }
    return handle;
}

// Helper: call a TVM-FFI function with string arg, return object result.
TVMFFIObjectHandle call_with_str(TVMFFIObjectHandle func, const std::string& arg) {
    TVMFFIAny args[1];
    // Pass as raw string (kTVMFFIRawStr)
    args[0].type_index = kTVMFFIRawStr;
    args[0].v_ptr = const_cast<char*>(arg.c_str());

    TVMFFIAny result;
    result.type_index = kTVMFFINone;
    if (TVMFFIFunctionCall(func, args, 1, &result) != 0) {
        return nullptr;
    }
    // Result is an Object (Module)
    if (result.type_index >= kTVMFFIStaticObjectBegin) {
        return static_cast<TVMFFIObjectHandle>(result.v_ptr);
    }
    return nullptr;
}

// Helper: call ModuleGetFunction(module, name, query_imports=false)
TVMFFIObjectHandle module_get_function(TVMFFIObjectHandle get_func_fn, TVMFFIObjectHandle module,
                                       const std::string& name) {
    TVMFFIAny args[3];
    // arg0: module (Object)
    args[0].type_index = kTVMFFIModule;
    args[0].v_ptr = module;
    // arg1: function name (raw string)
    args[1].type_index = kTVMFFIRawStr;
    args[1].v_ptr = const_cast<char*>(name.c_str());
    // arg2: query_imports = false
    args[2].type_index = kTVMFFIBool;
    args[2].v_int64 = 0;

    TVMFFIAny result;
    result.type_index = kTVMFFINone;
    if (TVMFFIFunctionCall(get_func_fn, args, 3, &result) != 0) {
        return nullptr;
    }
    if (result.type_index >= kTVMFFIStaticObjectBegin) {
        return static_cast<TVMFFIObjectHandle>(result.v_ptr);
    }
    return nullptr;
}

} // namespace

bool load_tvm_ffi_module_func(const std::string& so_path, const std::string& func_name,
                              const std::string& global_name) {
    // 1. Get ffi.ModuleLoadFromFile
    auto* load_fn = get_global("ffi.ModuleLoadFromFile");
    if (load_fn == nullptr) {
        std::cerr << "[tvm_ffi_module_loader] ffi.ModuleLoadFromFile not found\n";
        return false;
    }

    // 2. Load the .so module
    auto* module = call_with_str(load_fn, so_path);
    if (module == nullptr) {
        std::cerr << "[tvm_ffi_module_loader] Failed to load: " << so_path << '\n';
        return false;
    }

    // 3. Get the function from the module
    auto* get_func_fn = get_global("ffi.ModuleGetFunction");
    if (get_func_fn == nullptr) {
        std::cerr << "[tvm_ffi_module_loader] ffi.ModuleGetFunction not found\n";
        return false;
    }

    auto* func = module_get_function(get_func_fn, module, func_name);
    if (func == nullptr) {
        std::cerr << "[tvm_ffi_module_loader] Function '" << func_name << "' not found in module\n";
        return false;
    }

    // 4. Register as global
    TVMFFIByteArray gname;
    gname.data = global_name.c_str();
    gname.size = static_cast<int64_t>(global_name.size());
    if (TVMFFIFunctionSetGlobal(&gname, func, 1) != 0) {
        std::cerr << "[tvm_ffi_module_loader] Failed to register '" << global_name << "'\n";
        return false;
    }

    return true;
}

} // namespace trtmc

#endif // TRTMC_HAS_TVM_FFI
