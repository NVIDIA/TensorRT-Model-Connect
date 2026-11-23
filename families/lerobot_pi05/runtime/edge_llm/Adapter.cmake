# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The family DSO and its advertised capabilities must move together on reinstall.
# Conan stages only the sdk component in an empty prefix before copying the DSO.
function(_pi05_install_capability enabled)
  set(_script [[
    set(_prefix "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
    set(_family "@CMAKE_SHARED_LIBRARY_PREFIX@trtmc_model_lerobot_pi05@CMAKE_SHARED_LIBRARY_SUFFIX@")
    if(NOT CMAKE_INSTALL_COMPONENT STREQUAL "sdk" OR
        (NOT EXISTS "${_prefix}/@CMAKE_INSTALL_LIBDIR@/${_family}" AND
         NOT EXISTS "${_prefix}/@CMAKE_INSTALL_BINDIR@/${_family}"))
      if(@enabled@)
        file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/trtmc/lerobot_pi05"
          TYPE FILE FILES "@CMAKE_CURRENT_BINARY_DIR@/edge-0.12.json")
      else()
        file(REMOVE "${_prefix}/share/trtmc/lerobot_pi05/edge-0.12.json")
      endif()
    endif()
  ]])
  string(CONFIGURE "${_script}" _script @ONLY)
  install(CODE "${_script}")
  install(CODE "${_script}" COMPONENT sdk EXCLUDE_FROM_ALL)
endfunction()

if(NOT TRTMC_EDGELLM_RELEASE STREQUAL "0.12.0" OR NOT TARGET TensorRTEdgeLLM::Core)
  _pi05_install_capability(0)
  return()
endif()
# The base SDK does not install the experimental Pi0.5 library or headers. Never pick up a
# Pi0.5 archive from a different prefix than the Edge core against which it was built.
get_filename_component(_pi05_prefix "${TensorRTEdgeLLM_DIR}/../../.." ABSOLUTE)
set(_pi05_include "${_pi05_prefix}/include/tensorrt_edgellm/pi05")
get_filename_component(_pi05_libdir "${TensorRTEdgeLLM_DIR}/../.." ABSOLUTE)
set(_pi05_lib "${_pi05_libdir}/libpi05Experimental.a")
# Check the core interop header used by the adapter as well as the policy API.
# A partially installed external SDK must not make the native family unbuildable.
foreach(_pi05_file IN ITEMS "${_pi05_lib}"
    "${_pi05_prefix}/include/tensorrt_edgellm/common/trtInterop.h"
    "${_pi05_include}/runtime/pi05Policy.h" "${_pi05_include}/runtime/pi05Runtime.h"
    "${_pi05_include}/action/pi05ActionRunner.h" "${_pi05_include}/common/pi05Common.h"
    "${_pi05_include}/multimodal/pi05VisualRunner.h")
  if(NOT EXISTS "${_pi05_file}" OR IS_DIRECTORY "${_pi05_file}")
    message(STATUS "Pi0.5: selected Edge SDK lacks ${_pi05_file}; using native TensorRT")
    _pi05_install_capability(0)
    return()
  endif()
endforeach()

target_sources(trtmc_model_lerobot_pi05 PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/adapter.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/device_link.cu"
)
target_compile_definitions(trtmc_model_lerobot_pi05 PRIVATE TRTMC_PI05_EDGE_012=1)
target_include_directories(trtmc_model_lerobot_pi05 SYSTEM PRIVATE
  "${_pi05_prefix}/include" "${_pi05_include}" "${_pi05_prefix}/include/tensorrt_edgellm")
target_link_libraries(trtmc_model_lerobot_pi05 PRIVATE "${_pi05_lib}" TensorRTEdgeLLM::Core)
# The archive includes relocatable device code; the device link must match the SDK architecture
# (including architecture-specific suffixes), not just the numeric compute capability.
set_target_properties(trtmc_model_lerobot_pi05 PROPERTIES
  CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
add_custom_command(TARGET trtmc_model_lerobot_pi05 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:TensorRTEdgeLLM::Plugin> $<TARGET_FILE_DIR:trtmc_model_lerobot_pi05>/libNvInfer_edgellm_plugin.so
  VERBATIM
)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/edge-0.12.json"
  "{\"edge_release\":\"0.12.0\",\"policies\":[\"pi05_aloha\",\"pi05_droid\",\"pi05_libero\"]}\n")
_pi05_install_capability(1)
