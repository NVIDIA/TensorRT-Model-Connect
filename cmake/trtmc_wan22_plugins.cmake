# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Wan2.2 source-numeric TensorRT plugins are built once as a model-local,
# TensorRT-ABI-tagged companion DSO. The Python builder embeds this qualified
# image in .trtfb; the native runtime loads only the authenticated bundle
# section. Neither path compiles Model-Connect plugin source at runtime. The
# cuDNN frontend SDPA implementation remains an NVRTC runtime consumer while
# it constructs the platform execution plan.

set(TRTMC_BUILD_WAN22_PLUGIN_COMPANION "AUTO" CACHE STRING
  "Build the Wan2.2 TensorRT plugin companion DSO (AUTO, ON, or OFF).")
set_property(CACHE TRTMC_BUILD_WAN22_PLUGIN_COMPANION PROPERTY STRINGS AUTO ON OFF)

function(_trtmc_wan22_source_digest OUTPUT_VAR)
  set(_roots
    "${PROJECT_SOURCE_DIR}/python/tensorrt_model_connect/families/wan2_2_ti2v/umt5_cuda_plugins"
    "${PROJECT_SOURCE_DIR}/python/tensorrt_model_connect/families/wan2_2_ti2v/dit_cuda_plugins"
    "${PROJECT_SOURCE_DIR}/python/tensorrt_model_connect/families/wan2_2_ti2v/vae_cuda_plugins"
  )
  set(_files
    "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
    "${PROJECT_SOURCE_DIR}/src/runtime/models/wan2_2_ti2v/plugins/plugin_manifest.cpp.in"
  )
  foreach(_root IN LISTS _roots)
    file(GLOB_RECURSE _root_files CONFIGURE_DEPENDS
      "${_root}/CMakeLists.txt"
      "${_root}/*.cpp"
      "${_root}/*.cu"
      "${_root}/*.cuh"
      "${_root}/*.h"
      "${_root}/*.hpp"
      "${_root}/*.md"
      "${_root}/*.txt"
    )
    list(APPEND _files ${_root_files})
  endforeach()
  list(REMOVE_DUPLICATES _files)
  list(SORT _files)

  set(_identity "wan2_2_ti2v.plugins.v1\n")
  foreach(_file IN LISTS _files)
    file(RELATIVE_PATH _relative "${PROJECT_SOURCE_DIR}" "${_file}")
    file(SHA256 "${_file}" _sha256)
    string(APPEND _identity "${_relative}:${_sha256}\n")
  endforeach()
  string(SHA256 _digest "${_identity}")
  set(${OUTPUT_VAR} "${_digest}" PARENT_SCOPE)
endfunction()

function(trtmc_add_wan22_plugin_companion)
  # The official UniPC path evaluates CFG and scheduler expressions as
  # separate eager CUDA operations. Contracting those source-level FP32
  # boundaries into host FMA instructions changes the denoising trajectory.
  # Keep this precision constraint in the family-owned build definition.
  set_source_files_properties(
    "${PROJECT_SOURCE_DIR}/src/runtime/models/wan2_2_ti2v/pipeline.cpp"
    PROPERTIES
      COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-ffp-contract=off>"
  )

  if(TRTMC_BUILD_WAN22_PLUGIN_COMPANION STREQUAL "OFF")
    message(STATUS "Wan2.2 plugin companion: disabled")
    return()
  endif()
  if(NOT TRTMC_BUILD_WAN22_PLUGIN_COMPANION MATCHES "^(AUTO|ON)$")
    message(FATAL_ERROR
      "TRTMC_BUILD_WAN22_PLUGIN_COMPANION must be AUTO, ON, or OFF; got "
      "'${TRTMC_BUILD_WAN22_PLUGIN_COMPANION}'")
  endif()

  set(_missing)
  if(NOT CMAKE_CUDA_COMPILER)
    list(APPEND _missing "CUDA compiler")
  endif()
  if(NOT _trtmc_can_build_trt_backend)
    list(APPEND _missing "TensorRT headers/libnvinfer")
  endif()

  find_package(CUDAToolkit QUIET)
  if(NOT CUDAToolkit_FOUND)
    list(APPEND _missing "CUDA Toolkit CMake package")
  endif()

  set(_wan22_dependency_search_paths ${_trtmc_default_search_paths})
  if(_trtmc_python)
    execute_process(
      COMMAND "${_trtmc_python}" -c
        "import importlib.util,sys; s=importlib.util.find_spec('nvidia.cudnn'); sys.stdout.write(s.submodule_search_locations[0] if s else '')"
      OUTPUT_VARIABLE _wan22_cudnn_python_root
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(_wan22_cudnn_python_root)
      list(APPEND _wan22_dependency_search_paths "${_wan22_cudnn_python_root}")
    endif()
  endif()

  find_path(TRTMC_WAN22_CUDNN_INCLUDE_DIR cudnn.h
    PATHS ${_wan22_dependency_search_paths}
    PATH_SUFFIXES "" include nvidia/cudnn/include
  )
  find_library(TRTMC_WAN22_CUDNN_LIBRARY
    NAMES cudnn libcudnn.so.9 libcudnn.so
    PATHS ${_wan22_dependency_search_paths}
    PATH_SUFFIXES "" lib lib64 nvidia/cudnn/lib
  )
  if(NOT TRTMC_WAN22_CUDNN_INCLUDE_DIR OR NOT TRTMC_WAN22_CUDNN_LIBRARY)
    list(APPEND _missing "cuDNN headers/libcudnn")
  endif()
  if(NOT TARGET CUDA::cublasLt)
    list(APPEND _missing "CUDA::cublasLt")
  endif()
  if(NOT TARGET CUDA::nvrtc)
    list(APPEND _missing "CUDA::nvrtc")
  endif()

  if(_missing)
    string(JOIN ", " _missing_text ${_missing})
    if(TRTMC_BUILD_WAN22_PLUGIN_COMPANION STREQUAL "ON")
      message(FATAL_ERROR
        "Wan2.2 plugin companion was requested but dependencies are missing: "
        "${_missing_text}")
    endif()
    message(STATUS "Wan2.2 plugin companion: skipped (${_missing_text})")
    return()
  endif()

  set(_trt_major "")
  set(_trt_minor "")
  if(TRTMC_TRT_BACKEND_ABI MATCHES "^([0-9]+)_([0-9]+)$")
    set(_trt_major "${CMAKE_MATCH_1}")
    set(_trt_minor "${CMAKE_MATCH_2}")
  else()
    _trtmc_header_define_value(_trt_major NV_TENSORRT_MAJOR)
    _trtmc_header_define_value(_trt_minor NV_TENSORRT_MINOR)
  endif()
  if(_trt_major STREQUAL "" OR _trt_minor STREQUAL "")
    if(TRTMC_BUILD_WAN22_PLUGIN_COMPANION STREQUAL "ON")
      message(FATAL_ERROR
        "Wan2.2 plugin companion requires a detectable TensorRT major/minor ABI")
    endif()
    message(STATUS "Wan2.2 plugin companion: skipped (unknown TensorRT ABI)")
    return()
  endif()

  set(_umt5_dir
    "${PROJECT_SOURCE_DIR}/python/tensorrt_model_connect/families/wan2_2_ti2v/umt5_cuda_plugins")
  set(_dit_dir
    "${PROJECT_SOURCE_DIR}/python/tensorrt_model_connect/families/wan2_2_ti2v/dit_cuda_plugins")
  set(_vae_dir
    "${PROJECT_SOURCE_DIR}/python/tensorrt_model_connect/families/wan2_2_ti2v/vae_cuda_plugins")
  set(_cudnn_frontend_dir "${_dit_dir}/third_party/cudnn_frontend/include")
  if(NOT EXISTS "${_cudnn_frontend_dir}/cudnn_frontend.h")
    message(FATAL_ERROR
      "Wan2.2 plugin companion requires the vendored cuDNN frontend headers")
  endif()

  _trtmc_wan22_source_digest(TRTMC_WAN22_PLUGIN_SOURCE_DIGEST)
  set(TRTMC_WAN22_PLUGIN_SEMANTIC_ABI "wan2_2_ti2v.plugins.v1")
  set(TRTMC_WAN22_PLUGIN_CREATOR_SET
    "Wan22DitAdaptiveNormFp32:1:;Wan22DitBf16Linear:1:;Wan22DitCudnnSdpa:1:;Wan22DitFinalProjectionFp32:1:;Wan22DitFp32Barrier:1:;Wan22DitGatedResidualFp32:1:;Wan22DitGelu:1:;Wan22DitLayerNormFp32:1:;Wan22DitPatchEmbedding:1:;Wan22DitRmsNormFp32:1:;Wan22DitRotary:1:;Wan22DitSiluFp32:1:;Wan22DitTimeLinear1:1:;Wan22DitTimeLinear2:1:;Wan22DitTimeProjection:1:;Wan22Umt5Bf16Barrier:1:;Wan22Umt5SourceGelu:1:;Wan22Umt5SourceRmsNorm:1:;Wan22Umt5SourceSoftmax:1:;Wan22VaeConv3d:1:;Wan22VaeFp32Barrier:1:;Wan22VaeRmsNorm:1:")
  set(TRTMC_WAN22_PLUGIN_TRT_MAJOR "${_trt_major}")
  set(TRTMC_WAN22_PLUGIN_TRT_MINOR "${_trt_minor}")
  configure_file(
    "${PROJECT_SOURCE_DIR}/src/runtime/models/wan2_2_ti2v/plugins/plugin_manifest.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/wan2_2_ti2v/plugin_manifest.cpp"
    @ONLY
  )

  add_library(trtmc_model_wan2_2_ti2v_plugins SHARED
    "${CMAKE_CURRENT_BINARY_DIR}/generated/wan2_2_ti2v/plugin_manifest.cpp"
    "${_umt5_dir}/wan22_umt5_gelu_plugin.cu"
    "${_dit_dir}/wan22_adaptive_norm_fp32_plugin.cu"
    "${_dit_dir}/wan22_bf16_linear_plugin.cu"
    "${_dit_dir}/wan22_dit_numeric_plugins.cu"
    "${_dit_dir}/wan22_final_projection_fp32_plugin.cu"
    "${_dit_dir}/wan22_gated_residual_fp32_plugin.cu"
    "${_dit_dir}/wan22_cudnn_sdpa_plugin.cpp"
    "${_dit_dir}/wan22_layer_norm_fp32_plugin.cu"
    "${_dit_dir}/wan22_patch_embedding_plugin.cu"
    "${_dit_dir}/wan22_rms_norm_fp32_plugin.cu"
    "${_dit_dir}/wan22_silu_fp32_plugin.cu"
    "${_dit_dir}/wan22_time_linear1_plugin.cu"
    "${_dit_dir}/wan22_time_linear2_plugin.cu"
    "${_dit_dir}/wan22_time_projection_plugin.cu"
    "${_vae_dir}/wan22_vae_fp32_barrier_plugin.cu"
    "${_vae_dir}/wan22_vae_rms_norm_plugin.cu"
    "${_vae_dir}/wan22_vae_conv3d_plugin.cu"
  )
  target_compile_definitions(trtmc_model_wan2_2_ti2v_plugins PRIVATE
    CUDNN_FRONTEND_SKIP_JSON_LIB
  )
  target_include_directories(trtmc_model_wan2_2_ti2v_plugins PRIVATE
    "${_dit_dir}"
  )
  target_include_directories(trtmc_model_wan2_2_ti2v_plugins SYSTEM PRIVATE
    "${TRTMC_TRT_INCLUDE_DIR}"
    "${TRTMC_CUDA_INCLUDE_DIR}"
    "${TRTMC_WAN22_CUDNN_INCLUDE_DIR}"
    "${_cudnn_frontend_dir}"
  )
  target_link_libraries(trtmc_model_wan2_2_ti2v_plugins PRIVATE
    "${TRTMC_TRT_LIBRARY}"
    "${TRTMC_WAN22_CUDNN_LIBRARY}"
    CUDA::cublasLt
    CUDA::cudart
    CUDA::nvrtc
    ${CMAKE_DL_LIBS}
  )
  target_compile_options(trtmc_model_wan2_2_ti2v_plugins PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--ftz=false;--prec-div=true;--prec-sqrt=true;--fmad=true>"
  )
  # Package real cubins for both qualification platforms.  Deliberately omit
  # PTX so production startup never depends on driver JIT compilation.
  set_property(TARGET trtmc_model_wan2_2_ti2v_plugins PROPERTY
    CUDA_ARCHITECTURES "103-real;110-real")

  set_target_properties(trtmc_model_wan2_2_ti2v_plugins PROPERTIES
    OUTPUT_NAME
      "trtmc_model_wan2_2_ti2v_plugins_trt${_trt_major}_${_trt_minor}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/wan2_2_ti2v"
    # The exact build output is embedded into .trtfb.  Never record build-host
    # TensorRT/CUDA/cuDNN directories (or even an $ORIGIN fallback) in that
    # executable bundle section.  The builder and native runtime resolve the
    # ABI-qualified dependencies explicitly before loading this image.
    SKIP_BUILD_RPATH TRUE
    INSTALL_RPATH ""
  )
  # Keep the companion inside the Wan2.2 target closure.  Selective/model-proof
  # builds intentionally request only the owning model target, so depending on
  # the aggregate target alone would omit the plugin DSO from those artifacts.
  # The companion remains a build-only auxiliary DSO; `trtmc build` embeds it
  # into the Wan bundle and the native runtime needs no sibling library.
  if(TARGET trtmc_model_wan2_2_ti2v)
    add_dependencies(
      trtmc_model_wan2_2_ti2v
      trtmc_model_wan2_2_ti2v_plugins
    )
  else()
    message(FATAL_ERROR
      "Wan2.2 plugin companion requires target trtmc_model_wan2_2_ti2v")
  endif()
  add_dependencies(trtmc_model_plugins trtmc_model_wan2_2_ti2v_plugins)
  install(TARGETS trtmc_model_wan2_2_ti2v_plugins
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/trtmc/models/wan2_2_ti2v
  )
  message(STATUS
    "Wan2.2 plugin companion: libtrtmc_model_wan2_2_ti2v_plugins_trt${_trt_major}_${_trt_minor}.so")
  message(STATUS "Wan2.2 plugin source digest: ${TRTMC_WAN22_PLUGIN_SOURCE_DIGEST}")
endfunction()
