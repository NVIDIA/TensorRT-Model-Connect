# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The family DSO and its advertised capabilities must move together on reinstall.
# Conan stages only the sdk component in an empty prefix before copying the DSO.
function(_groot_install_capability enabled)
  set(_script [[
    set(_prefix "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
    set(_family "@CMAKE_SHARED_LIBRARY_PREFIX@trtmc_model_lerobot_groot@CMAKE_SHARED_LIBRARY_SUFFIX@")
    if(NOT CMAKE_INSTALL_COMPONENT STREQUAL "sdk" OR
        (NOT EXISTS "${_prefix}/@CMAKE_INSTALL_LIBDIR@/${_family}" AND
         NOT EXISTS "${_prefix}/@CMAKE_INSTALL_BINDIR@/${_family}"))
      if(@enabled@)
        file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/trtmc/lerobot_groot"
          TYPE FILE FILES "@CMAKE_CURRENT_BINARY_DIR@/edge-0.12.json")
      else()
        file(REMOVE "${_prefix}/share/trtmc/lerobot_groot/edge-0.12.json")
      endif()
    endif()
  ]])
  string(CONFIGURE "${_script}" _script @ONLY)
  install(CODE "${_script}")
  install(CODE "${_script}" COMPONENT sdk EXCLUDE_FROM_ALL)
endfunction()

if(NOT TRTMC_EDGELLM_RELEASE STREQUAL "0.12.0" OR NOT TARGET TensorRTEdgeLLM::Core)
  _groot_install_capability(0)
  return()
endif()
# The base SDK does not install the experimental GR00T library or headers. Never pick up a
# GR00T archive from a different prefix than the Edge core against which it was built.
get_filename_component(_groot_prefix "${TensorRTEdgeLLM_DIR}/../../.." ABSOLUTE)
set(_groot_include "${_groot_prefix}/include/tensorrt_edgellm/gr00t_n1d7")
get_filename_component(_groot_libdir "${TensorRTEdgeLLM_DIR}/../.." ABSOLUTE)
set(_groot_lib "${_groot_libdir}/libgr00tExperimental.a")
# A partial policy installation must leave the native family usable. Check the runtime
# API and the policy-specific headers it includes from the same SDK.
foreach(_groot_file IN ITEMS "${_groot_lib}" "${_groot_include}/runtime/gr00tRuntime.h"
    "${_groot_include}/action/gr00tPolicy.h" "${_groot_include}/action/gr00tPolicyRunner.h"
    "${_groot_include}/multimodal/gr00tVisualRunner.h")
  if(NOT EXISTS "${_groot_file}" OR IS_DIRECTORY "${_groot_file}")
    message(STATUS "GR00T: selected Edge SDK lacks ${_groot_file}; using native TensorRT")
    _groot_install_capability(0)
    return()
  endif()
endforeach()

target_sources(trtmc_model_lerobot_groot PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/adapter.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/device_link.cu"
)
target_compile_definitions(trtmc_model_lerobot_groot PRIVATE TRTMC_GROOT_EDGE_012=1)
target_include_directories(trtmc_model_lerobot_groot SYSTEM PRIVATE
  "${_groot_include}" "${_groot_prefix}/include/tensorrt_edgellm")
target_link_libraries(trtmc_model_lerobot_groot PRIVATE "${_groot_lib}" TensorRTEdgeLLM::Core)
# The archive includes relocatable device code; the device link must match the SDK architecture
# (including architecture-specific suffixes), not just the numeric compute capability.
set_target_properties(trtmc_model_lerobot_groot PROPERTIES
  CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
add_custom_command(TARGET trtmc_model_lerobot_groot POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:TensorRTEdgeLLM::Plugin> $<TARGET_FILE_DIR:trtmc_model_lerobot_groot>/libNvInfer_edgellm_plugin.so
  VERBATIM
)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/edge-0.12.json"
  "{\"edge_release\":\"0.12.0\",\"policy\":\"libero_sim\"}\n")
_groot_install_capability(1)
