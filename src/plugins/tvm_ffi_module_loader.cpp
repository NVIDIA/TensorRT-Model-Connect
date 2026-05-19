// Load TVM-FFI modules from .so files — pure C++ via TVM-FFI C ABI.

#if TRTMC_HAS_TVM_FFI

#include "plugins/tvm_ffi_module_loader.h"

#include <cstring>
#include <iostream>
#include <string>
#include <tvm/ffi/c_api.h>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

struct OwnedAny {
    TVMFFIAny value{};

    OwnedAny() {
        value.type_index = kTVMFFINone;
        value.zero_padding = 0;
        value.v_int64 = 0;
    }

    OwnedAny(const OwnedAny&) = delete;
    OwnedAny& operator=(const OwnedAny&) = delete;

    OwnedAny(OwnedAny&& other) noexcept : value(other.value) {
        other.value.type_index = kTVMFFINone;
        other.value.zero_padding = 0;
        other.value.v_int64 = 0;
    }

    OwnedAny& operator=(OwnedAny&& other) noexcept {
        if (this != &other) {
            reset();
            value = other.value;
            other.value.type_index = kTVMFFINone;
            other.value.zero_padding = 0;
            other.value.v_int64 = 0;
        }
        return *this;
    }

    ~OwnedAny() { reset(); }

    void reset() {
        if (value.type_index >= kTVMFFIStaticObjectBegin && value.v_obj != nullptr)
            TVMFFIObjectDecRef(value.v_obj);
        value.type_index = kTVMFFINone;
        value.zero_padding = 0;
        value.v_int64 = 0;
    }

    TVMFFIObjectHandle object() const {
        if (value.type_index >= kTVMFFIStaticObjectBegin)
            return value.v_obj;
        return nullptr;
    }
};

std::vector<OwnedAny>& loaded_objects() {
    static auto* objects = new std::vector<OwnedAny>();
    return *objects;
}

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

OwnedAny own_call_result(const TVMFFIAny& result, const char* context) {
    OwnedAny owned;
    if (result.type_index == kTVMFFINone)
        return owned;
    if (TVMFFIAnyViewToOwnedAny(&result, &owned.value) != 0) {
        std::cerr << "[tvm_ffi_module_loader] Failed to own result from " << context << '\n';
        return {};
    }
    return owned;
}

// Helper: call a TVM-FFI function with string arg, return object result.
OwnedAny call_with_str(TVMFFIObjectHandle func, const std::string& arg) {
    TVMFFIAny args[1];
    // Pass as raw string (kTVMFFIRawStr)
    args[0].type_index = kTVMFFIRawStr;
    args[0].v_ptr = const_cast<char*>(arg.c_str());

    TVMFFIAny result;
    result.type_index = kTVMFFINone;
    if (TVMFFIFunctionCall(func, args, 1, &result) != 0) {
        return {};
    }
    return own_call_result(result, "ffi.ModuleLoadFromFile");
}

// Helper: call ModuleGetFunction(module, name, query_imports=false)
OwnedAny module_get_function(TVMFFIObjectHandle get_func_fn, const TVMFFIAny& module,
                             const std::string& name) {
    TVMFFIAny args[3];
    // arg0: module (Object)
    args[0] = module;
    // arg1: function name (raw string)
    args[1].type_index = kTVMFFIRawStr;
    args[1].v_ptr = const_cast<char*>(name.c_str());
    // arg2: query_imports = false
    args[2].type_index = kTVMFFIBool;
    args[2].v_int64 = 0;

    TVMFFIAny result;
    result.type_index = kTVMFFINone;
    if (TVMFFIFunctionCall(get_func_fn, args, 3, &result) != 0) {
        return {};
    }
    return own_call_result(result, "ffi.ModuleGetFunction");
}

} // namespace

bool load_tvm_ffi_module_func(const std::string& so_path, const std::string& func_name,
                              const std::string& global_name) {
    if (get_global(global_name.c_str()) != nullptr)
        return true;

    // 1. Get ffi.ModuleLoadFromFile
    auto* load_fn = get_global("ffi.ModuleLoadFromFile");
    if (load_fn == nullptr) {
        std::cerr << "[tvm_ffi_module_loader] ffi.ModuleLoadFromFile not found\n";
        return false;
    }

    // 2. Load the .so module
    auto module = call_with_str(load_fn, so_path);
    if (module.object() == nullptr) {
        std::cerr << "[tvm_ffi_module_loader] Failed to load: " << so_path << '\n';
        return false;
    }

    // 3. Get the function from the module
    auto* get_func_fn = get_global("ffi.ModuleGetFunction");
    if (get_func_fn == nullptr) {
        std::cerr << "[tvm_ffi_module_loader] ffi.ModuleGetFunction not found\n";
        return false;
    }

    auto func = module_get_function(get_func_fn, module.value, func_name);
    if (func.object() == nullptr) {
        std::cerr << "[tvm_ffi_module_loader] Function '" << func_name << "' not found in module\n";
        return false;
    }

    // 4. Register as global
    TVMFFIByteArray gname;
    gname.data = global_name.c_str();
    gname.size = static_cast<int64_t>(global_name.size());
    if (TVMFFIFunctionSetGlobal(&gname, func.object(), 1) != 0) {
        std::cerr << "[tvm_ffi_module_loader] Failed to register '" << global_name << "'\n";
        return false;
    }

    loaded_objects().push_back(std::move(module));
    loaded_objects().push_back(std::move(func));
    return true;
}

} // namespace trtmc

#endif // TRTMC_HAS_TVM_FFI
