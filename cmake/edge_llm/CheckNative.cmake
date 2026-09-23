# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Read the complete TensorRT SDK version using the compiler, including aliased macros.
# include_dir: native SDK include directory; output: caller variable receiving x.y.z.build.
function(_edgellm_trt_version include_dir output)
  set(_version)
  set(_probe "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/edgellm-version.cpp")
  file(WRITE "${_probe}" "#include <NvInferVersion.h>\n")
  foreach(_part IN ITEMS MAJOR MINOR PATCH BUILD)
    file(APPEND "${_probe}" "TRTMC_EDGE_${_part}=NV_TENSORRT_${_part}\n")
  endforeach()
  execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -E -P -I "${include_dir}" "${_probe}"
    OUTPUT_VARIABLE _expanded COMMAND_ERROR_IS_FATAL ANY)
  foreach(_part IN ITEMS MAJOR MINOR PATCH BUILD)
    if(NOT _expanded MATCHES "TRTMC_EDGE_${_part}=[ \t]*([0-9]+)")
      message(FATAL_ERROR "Cannot determine TensorRT ${_part} from ${include_dir}")
    endif()
    list(APPEND _version "${CMAKE_MATCH_1}")
  endforeach()
  list(JOIN _version "." _version)
  set(${output} "${_version}" PARENT_SCOPE)
endfunction()

# Require the installed package GPU architecture to be present on this build host.
function(_edgellm_check_gpu architecture)
  execute_process(COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _sms RESULT_VARIABLE _result OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REPLACE "." "" _sms "${_sms}")
  string(REPLACE "\n" ";" _sms "${_sms}")
  if(NOT _result EQUAL 0 OR NOT architecture IN_LIST _sms)
    message(FATAL_ERROR "EdgeLLM requires a local GPU with architecture ${architecture}")
  endif()
endfunction()

# A version label alone is not an ABI guarantee: development headers can retain
# 3.12.0 while changing parser layouts inside the same C++ ABI namespace.
function(_edgellm_json_include output)
  get_target_property(_includes nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
  foreach(_include IN LISTS _includes)
    string(REGEX REPLACE "^\\$<BUILD_INTERFACE:(.*)>$" "\\1" _include "${_include}")
    if(EXISTS "${_include}/nlohmann/json.hpp")
      set(${output} "${_include}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  message(FATAL_ERROR "Cannot locate nlohmann_json headers for EdgeLLM ABI validation")
endfunction()

function(_edgellm_check_json_headers include_dir vendor_dir)
  set(_header "${include_dir}/nlohmann/json.hpp")
  set(_single "${vendor_dir}/single_include/nlohmann/json.hpp")
  set(_multiple "${vendor_dir}/include/nlohmann/json.hpp")
  if(NOT EXISTS "${_header}" OR NOT EXISTS "${_single}" OR NOT EXISTS "${_multiple}")
    message(FATAL_ERROR "Missing nlohmann_json headers for EdgeLLM ABI validation")
  endif()
  file(SHA256 "${_header}" _actual)
  file(SHA256 "${_single}" _expected_single)
  if(_actual STREQUAL _expected_single)
    return()
  endif()
  file(GLOB_RECURSE _headers RELATIVE "${vendor_dir}/include" "${vendor_dir}/include/nlohmann/*.hpp")
  foreach(_relative IN LISTS _headers)
    if(EXISTS "${include_dir}/${_relative}")
      file(SHA256 "${include_dir}/${_relative}" _actual)
      file(SHA256 "${vendor_dir}/include/${_relative}" _expected)
      if(_actual STREQUAL _expected)
        continue()
      endif()
    endif()
    message(FATAL_ERROR "EdgeLLM requires the pinned nlohmann_json headers, not only the same version label. Set nlohmann_json_DIR to an installation of the pinned Edge 3rdParty/nlohmannJson dependency. Mismatch: ${_relative}")
  endforeach()
endfunction()

# Verify the selected native library itself, not only its accompanying headers.
function(_edgellm_check_trt_library library expected)
  if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "TensorRT library validation requires native execution")
  endif()
  set(_probe "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/edgellm-library-version.cpp")
  file(WRITE "${_probe}" [=[
#include <dlfcn.h>
#include <iostream>
int main(int argc, char** argv) {
  if (argc != 2) return 1;
  void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!library) { std::cerr << dlerror(); return 2; }
  const char* names[] = {"getInferLibMajorVersion", "getInferLibMinorVersion",
                         "getInferLibPatchVersion", "getInferLibBuildVersion"};
  for (int i = 0; i < 4; ++i) {
    auto version = reinterpret_cast<int (*)()>(dlsym(library, names[i]));
    if (!version) { std::cerr << "Missing " << names[i]; dlclose(library); return 3; }
    if (i) std::cout << ".";
    std::cout << version();
  }
  dlclose(library);
  return 0;
}
]=])
  unset(_edge_version_run CACHE)
  unset(_edge_version_compiled CACHE)
  try_run(_edge_version_run _edge_version_compiled
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/edgellm-library-version" "${_probe}"
    LINK_LIBRARIES "${CMAKE_DL_LIBS}" ARGS "${library}"
    RUN_OUTPUT_VARIABLE _actual COMPILE_OUTPUT_VARIABLE _compile_output)
  if(NOT _edge_version_compiled OR NOT _edge_version_run STREQUAL "0")
    message(FATAL_ERROR "Cannot verify selected TensorRT library ${library}: ${_actual} ${_compile_output}")
  endif()
  if(NOT _actual STREQUAL expected)
    message(FATAL_ERROR "EdgeLLM requires TensorRT library ${expected}; selected ${library} reports ${_actual}")
  endif()
endfunction()
