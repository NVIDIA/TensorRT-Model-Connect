# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Upstream 0.12 installs the CMake targets, not a Python interpreter. Do not
# copy a caller's venv or record a build-machine interpreter path in a wheel.
_edgellm_check_trt_selection("${TensorRT_INCLUDE_DIR}" "${TensorRT_LIBRARY}")
_edgellm_trt_version("${TensorRT_INCLUDE_DIR}" _edge_trt_version)
# This is the native consumer architecture. An external SDK does not export
# a manifest of its compiled device code. Each family still validates execution.
set(EdgeLLM_CUDA_ARCHITECTURE "${TRTMC_EDGELLM_CUDA_ARCHITECTURE}")

# Keep the SDK exported target's transitive libraries, including device link.
# Do not reconstruct an installed static link interface from archive names.
if(NOT TARGET TensorRTEdgeLLM::Core OR NOT TARGET TensorRTEdgeLLM::Plugin)
  message(FATAL_ERROR "The installed TensorRTEdgeLLM package lacks Core or Plugin")
endif()
get_target_property(_edge_includes TensorRTEdgeLLM::Core INTERFACE_INCLUDE_DIRECTORIES)
foreach(_edge_dir IN LISTS _edge_includes)
  if(EXISTS "${_edge_dir}/nlohmann/json.hpp")
    set(_edge_sdk_include "${_edge_dir}")
    break()
  endif()
endforeach()
if(NOT _edge_sdk_include)
  message(FATAL_ERROR "The installed TensorRTEdgeLLM SDK is missing its public JSON headers")
endif()
_edgellm_json_include(_mc_json)
# Public C++ request objects contain nlohmann::json. Do not cross that ABI
# boundary with headers from a different development snapshot.
file(GLOB_RECURSE _json_headers RELATIVE "${_edge_sdk_include}" "${_edge_sdk_include}/nlohmann/*.hpp")
foreach(_header IN LISTS _json_headers)
  execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files
    "${_edge_sdk_include}/${_header}" "${_mc_json}/${_header}"
    RESULT_VARIABLE _different)
  if(_different)
    message(FATAL_ERROR "TensorRTEdgeLLM and Model Connect JSON headers differ: ${_header}")
  endif()
endforeach()

# Record the release selected by find_package. An external vendor package
# attests its release version, not an independently verified Git checkout.
if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}" OR IS_ABSOLUTE "${CMAKE_INSTALL_BINDIR}")
  message(FATAL_ERROR "Edge SDK receipt requires relocatable install directories")
endif()
set(_all false)
if(_edge_provisioned AND TRTMC_EDGELLM_ALL_KERNELS)
  set(_all true)
endif()
get_filename_component(_sdk_prefix "${TensorRTEdgeLLM_DIR}/../../.." ABSOLUTE)
set(_onnx false)
if(TRTMC_EDGELLM_ONNX)
  set(_onnx true)
  foreach(_tool IN ITEMS llm_build visual_build audio_build)
    if(NOT EXISTS "${_sdk_prefix}/bin/${_tool}")
      message(FATAL_ERROR "The installed Edge SDK lacks requested ONNX tool: ${_tool}")
    endif()
    foreach(_component IN ITEMS sdk EdgeLLM)
      install(PROGRAMS "${_sdk_prefix}/bin/${_tool}" DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT "${_component}")
    endforeach()
  endforeach()
endif()
set(_receipt "${CMAKE_CURRENT_BINARY_DIR}/edgellm-sdk/edge-llm.json")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/edgellm-sdk")
file(WRITE "${_receipt}" "{
  \"schema_version\": 2,
  \"version\": \"${TensorRTEdgeLLM_VERSION}\",
  \"release_revision\": \"${_edge_revision}\",
  \"arch\": \"${CMAKE_SYSTEM_PROCESSOR}\",
  \"architectures\": [${EdgeLLM_CUDA_ARCHITECTURE}],
  \"cuda_version\": \"${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}\",
  \"tensorrt_version\": \"${_edge_trt_version}\",
  \"all_native_kernels\": ${_all},
  \"onnx\": ${_onnx},
  \"onnx_builder\": \"${CMAKE_INSTALL_BINDIR}/llm_build\",
  \"visual_builder\": \"${CMAKE_INSTALL_BINDIR}/visual_build\",
  \"audio_builder\": \"${CMAKE_INSTALL_BINDIR}/audio_build\",
  \"plugin\": \"${CMAKE_INSTALL_LIBDIR}/libNvInfer_edgellm_plugin.so\"
}\n")
foreach(_component IN ITEMS sdk EdgeLLM)
  install(FILES "${CMAKE_CURRENT_LIST_DIR}/NOTICE"
    DESTINATION share/licenses/TensorRTEdgeLLM COMPONENT "${_component}")
  install(FILES "${_sdk_prefix}/share/licenses/TensorRTEdgeLLM/LICENSE"
    DESTINATION share/licenses/TensorRTEdgeLLM COMPONENT "${_component}")
  install(FILES "${_receipt}" DESTINATION share/trtmc COMPONENT "${_component}")
  install(FILES "$<TARGET_FILE:TensorRTEdgeLLM::Plugin>" DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RENAME libNvInfer_edgellm_plugin.so COMPONENT "${_component}")
endforeach()
