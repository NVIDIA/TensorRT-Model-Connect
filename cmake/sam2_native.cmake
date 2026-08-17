# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build-only SAM2 TensorRT-native attention support. The builder and benchmark
# consume only project-owned sources and TensorRT native attention.

# Stable install metadata for opt-in package lanes.  This component contains
# only the checkpoint-to-bundle builder; the diagnostic benchmark remains a
# build-tree qualification tool.
set(TRTMC_SAM2_NATIVE_BUILDER_INSTALL_COMPONENT "sam2_native_builder")
set(CPACK_COMPONENT_SAM2_NATIVE_BUILDER_DISPLAY_NAME
    "SAM2 TensorRT native-attention builder")
set(CPACK_COMPONENT_SAM2_NATIVE_BUILDER_DESCRIPTION
    "Unqualified, opt-in SAM2 checkpoint-to-bundle builder using TensorRT native attention")

function(_trtmc_install_sam2_native_builder TARGET_NAME)
  if(NOT TARGET "${TARGET_NAME}")
    message(FATAL_ERROR
      "Cannot install missing SAM2 native-attention builder target '${TARGET_NAME}'")
  endif()
  install(TARGETS "${TARGET_NAME}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT "${TRTMC_SAM2_NATIVE_BUILDER_INSTALL_COMPONENT}"
  )
endfunction()

function(_trtmc_sam2_native_require_exact_variable VARIABLE EXPECTED DESCRIPTION)
  if(NOT "${${VARIABLE}}" STREQUAL "${EXPECTED}")
    message(FATAL_ERROR
      "SAM2 native-attention ${DESCRIPTION} must be exactly '${EXPECTED}'; "
      "found '${${VARIABLE}}'")
  endif()
endfunction()

function(_trtmc_sam2_native_read_elf_soname OUTPUT LIBRARY DESCRIPTION READELF)
  if(NOT EXISTS "${LIBRARY}" OR IS_DIRECTORY "${LIBRARY}")
    message(FATAL_ERROR "SAM2 native-attention ${DESCRIPTION} is missing: ${LIBRARY}")
  endif()
  execute_process(
    COMMAND "${READELF}" --dynamic --wide "${LIBRARY}"
    RESULT_VARIABLE _trtmc_sam2_native_readelf_status
    OUTPUT_VARIABLE _trtmc_sam2_native_readelf_stdout
    ERROR_VARIABLE _trtmc_sam2_native_readelf_stderr
  )
  if(NOT _trtmc_sam2_native_readelf_status EQUAL 0)
    message(FATAL_ERROR
      "Cannot inspect SAM2 native-attention ${DESCRIPTION}: "
      "${_trtmc_sam2_native_readelf_stderr}")
  endif()
  string(REGEX MATCH
    "\\(SONAME\\)[^\n]*\\[([^]]+)\\]"
    _trtmc_sam2_native_soname_match
    "${_trtmc_sam2_native_readelf_stdout}")
  if(NOT _trtmc_sam2_native_soname_match)
    message(FATAL_ERROR "SAM2 native-attention ${DESCRIPTION} has no ELF SONAME")
  endif()
  set(${OUTPUT} "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()

function(trtmc_add_sam2_native_builder)
  if(CMAKE_VERSION VERSION_LESS 3.21)
    message(FATAL_ERROR
      "TRTMC_BUILD_SAM2_NATIVE_BUILDER=ON requires CMake 3.21 or newer "
      "for closed C++ compile/link launchers")
  endif()
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "TRTMC_BUILD_SAM2_NATIVE_BUILDER=ON requires nvcc")
  endif()
  if(NOT _trtmc_can_build_trt_backend)
    message(FATAL_ERROR
      "TRTMC_BUILD_SAM2_NATIVE_BUILDER=ON requires TensorRT headers and libnvinfer")
  endif()

  _trtmc_header_define_value(_trtmc_sam2_native_trt_major NV_TENSORRT_MAJOR)
  if(NOT _trtmc_sam2_native_trt_major STREQUAL "11")
    message(FATAL_ERROR
      "The SAM2 native-attention builder requires TensorRT 11; "
      "found major '${_trtmc_sam2_native_trt_major}'")
  endif()

  if(CMAKE_CONFIGURATION_TYPES OR NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR
      "The SAM2 native-attention reference builder requires a single-config Release build")
  endif()
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_CXX_FLAGS "" "ambient C++ flags")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" "Release C++ flags")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_CUDA_FLAGS "" "ambient CUDA flags")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG" "Release CUDA flags")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_SHARED_LINKER_FLAGS "" "ambient shared-link flags")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_SHARED_LINKER_FLAGS_RELEASE "" "Release shared-link flags")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_EXE_LINKER_FLAGS "" "ambient executable-link flags")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_EXE_LINKER_FLAGS_RELEASE "" "Release executable-link flags")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_CXX_COMPILER_LAUNCHER "" "C++ compiler launcher")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_CUDA_COMPILER_LAUNCHER "" "CUDA compiler launcher")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_CXX_LINKER_LAUNCHER "" "C++ linker launcher")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_CUDA_LINKER_LAUNCHER "" "CUDA linker launcher")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_INTERPROCEDURAL_OPTIMIZATION "" "interprocedural optimization")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE "" "Release interprocedural optimization")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_CXX_STANDARD_LIBRARIES "" "ambient C++ standard libraries")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_CUDA_STANDARD_LIBRARIES "" "ambient CUDA standard libraries")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_LINK_WHAT_YOU_USE "" "link-what-you-use instrumentation")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_POSITION_INDEPENDENT_CODE "" "global position-independent-code initializer")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_EXECUTABLE_ENABLE_EXPORTS "" "global executable-export initializer")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_SHARED_LIBRARY_ENABLE_EXPORTS "" "global shared-library export initializer")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_CXX_VISIBILITY_PRESET "" "global C++ visibility initializer")
  _trtmc_sam2_native_require_exact_variable(
    CMAKE_VISIBILITY_INLINES_HIDDEN "" "global inline-visibility initializer")
  if(DEFINED ENV{LD_PRELOAD} AND NOT "$ENV{LD_PRELOAD}" STREQUAL "")
    message(FATAL_ERROR
      "SAM2 native-attention opt-in builds reject an ambient LD_PRELOAD")
  endif()

  find_program(_trtmc_sam2_native_readelf NAMES readelf llvm-readelf)
  find_program(_trtmc_sam2_native_env NAMES env)
  if(NOT _trtmc_sam2_native_readelf OR NOT _trtmc_sam2_native_env)
    message(FATAL_ERROR "SAM2 native-attention requires readelf and env")
  endif()

  _trtmc_sam2_native_read_elf_soname(
    _trtmc_sam2_native_trt_soname "${TRTMC_TRT_LIBRARY}" "TensorRT library"
    "${_trtmc_sam2_native_readelf}")
  if(NOT _trtmc_sam2_native_trt_soname STREQUAL "libnvinfer.so.11")
    message(FATAL_ERROR
      "TensorRT 11 headers require selected library SONAME libnvinfer.so.11; "
      "found ${_trtmc_sam2_native_trt_soname}")
  endif()
  string(REGEX MATCH "^([0-9]+)" _trtmc_sam2_native_cuda_major_match
    "${CMAKE_CUDA_COMPILER_VERSION}")
  if(NOT _trtmc_sam2_native_cuda_major_match)
    message(FATAL_ERROR "Cannot derive the CUDA toolkit major version")
  endif()
  set(_trtmc_sam2_native_cuda_major "${CMAKE_MATCH_1}")
  _trtmc_sam2_native_read_elf_soname(
    _trtmc_sam2_native_cudart_soname "${TRTMC_CUDART_LIBRARY}" "CUDA runtime library"
    "${_trtmc_sam2_native_readelf}")
  if(NOT _trtmc_sam2_native_cudart_soname STREQUAL
     "libcudart.so.${_trtmc_sam2_native_cuda_major}")
    message(FATAL_ERROR
      "CUDA ${_trtmc_sam2_native_cuda_major} compiler requires selected runtime SONAME "
      "libcudart.so.${_trtmc_sam2_native_cuda_major}; found "
      "${_trtmc_sam2_native_cudart_soname}")
  endif()

  set(_trtmc_sam2_native_host_path "/usr/bin:/bin")
  set(_trtmc_sam2_native_host_launcher
    "${_trtmc_sam2_native_env}" -i "PATH=${_trtmc_sam2_native_host_path}" LC_ALL=C)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/CMakeLists.txt"
    "${CMAKE_CURRENT_FUNCTION_LIST_FILE}")

  set(_trtmc_sam2_builder_sources
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder/checkpoint_reader.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder/bundle_writer.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder/durable_file_writer.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder/sam2_engine_builder.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder/sam2_engine_builder_trt.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder/sam2_image_network.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder/sam2_tracker_contract.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder/sam2_tracker_network.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder/sam2_trt_layers.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder/main.cpp"
  )
  add_executable(trtmc_sam2_native_builder ${_trtmc_sam2_builder_sources})
  target_include_directories(trtmc_sam2_native_builder PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder"
  )
  target_include_directories(trtmc_sam2_native_builder SYSTEM PRIVATE
    "${TRTMC_TRT_INCLUDE_DIR}"
    "${TRTMC_CUDA_INCLUDE_DIR}"
  )
  target_compile_definitions(trtmc_sam2_native_builder PRIVATE TRTMC_HAS_TRT=1)
  target_compile_options(trtmc_sam2_native_builder PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
  target_link_libraries(trtmc_sam2_native_builder PRIVATE
    "${TRTMC_TRT_LIBRARY}"
    "${TRTMC_CUDART_LIBRARY}"
    Threads::Threads
  )
  set_target_properties(trtmc_sam2_native_builder PROPERTIES
    OUTPUT_NAME "sam2_native_builder"
    INTERPROCEDURAL_OPTIMIZATION FALSE
    INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE
    POSITION_INDEPENDENT_CODE FALSE
    ENABLE_EXPORTS FALSE
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN FALSE
    UNITY_BUILD FALSE
    LINK_WHAT_YOU_USE FALSE
    CXX_COMPILER_LAUNCHER "${_trtmc_sam2_native_host_launcher}"
    CXX_LINKER_LAUNCHER "${_trtmc_sam2_native_host_launcher}"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH FALSE
  )
  _trtmc_install_sam2_native_builder(trtmc_sam2_native_builder)

  find_package(JPEG REQUIRED)
  set(_trtmc_sam2_benchmark_sources
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_benchmark/main.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_benchmark/sam2_benchmark_accuracy.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_benchmark/sam2_benchmark_protocol.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_benchmark/sam2_checked_plan_module.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder/checkpoint_reader.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder/bundle_writer.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder/durable_file_writer.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder/sam2_engine_builder.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder/sam2_engine_builder_trt.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder/sam2_image_network.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder/sam2_tracker_contract.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder/sam2_tracker_network.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder/sam2_trt_layers.cpp"
    "${PROJECT_SOURCE_DIR}/src/bundle/bundle_format.cpp"
    "${PROJECT_SOURCE_DIR}/src/utils/json_helpers.cpp"
    "${PROJECT_SOURCE_DIR}/src/runtime/models/sam2/sam2_jpeg_decoder.cpp"
    "${PROJECT_SOURCE_DIR}/src/runtime/models/sam2/sam2_native_bundle_loader.cpp"
    "${PROJECT_SOURCE_DIR}/src/runtime/models/sam2/sam2_qualification_authority.cpp"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_benchmark/sam2_empty_qualification_pin_provider.cpp"
    "${PROJECT_SOURCE_DIR}/src/runtime/models/sam2/sam2_native_video_processor.cpp"
    "${PROJECT_SOURCE_DIR}/src/runtime/models/sam2/sam2_video_session.cpp"
    "${PROJECT_SOURCE_DIR}/src/runtime/models/sam2/sam2_bbox_postprocess.cpp"
    "${PROJECT_SOURCE_DIR}/src/runtime/models/sam2/sam2_preprocess.cpp"
    "${PROJECT_SOURCE_DIR}/src/runtime/models/sam2/sam2_preprocess_cuda.cu"
    "${PROJECT_SOURCE_DIR}/src/runtime/models/sam2/sam2_mask_postprocess.cpp"
    "${PROJECT_SOURCE_DIR}/src/runtime/models/sam2/sam2_device_workspace.cpp"
    "${PROJECT_SOURCE_DIR}/src/runtime/models/sam2/sam2_mask_postprocess_cuda.cu"
  )
  add_executable(trtmc_sam2_native_benchmark ${_trtmc_sam2_benchmark_sources})
  target_include_directories(trtmc_sam2_native_benchmark PRIVATE
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder"
    "${PROJECT_SOURCE_DIR}/tools/sam2_native_benchmark"
  )
  target_include_directories(trtmc_sam2_native_benchmark SYSTEM PRIVATE
    "${TRTMC_TRT_INCLUDE_DIR}"
    "${TRTMC_CUDA_INCLUDE_DIR}"
  )
  target_compile_definitions(trtmc_sam2_native_benchmark PRIVATE TRTMC_HAS_TRT=1)
  target_compile_options(trtmc_sam2_native_benchmark PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>"
  )
  target_link_options(trtmc_sam2_native_benchmark PRIVATE "-Wl,--no-undefined")
  target_link_libraries(trtmc_sam2_native_benchmark PRIVATE
    "${TRTMC_TRT_LIBRARY}"
    "${TRTMC_CUDART_LIBRARY}"
    JPEG::JPEG
    nlohmann_json::nlohmann_json
    Threads::Threads
    "${CMAKE_DL_LIBS}"
  )
  set_target_properties(trtmc_sam2_native_benchmark PROPERTIES
    OUTPUT_NAME "sam2_native_benchmark"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED TRUE
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED TRUE
    CUDA_ARCHITECTURES "89-real"
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION FALSE
    CUDA_RESOLVE_DEVICE_SYMBOLS FALSE
    LINKER_LANGUAGE CXX
    INTERPROCEDURAL_OPTIMIZATION FALSE
    INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE
    POSITION_INDEPENDENT_CODE FALSE
    ENABLE_EXPORTS FALSE
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN FALSE
    UNITY_BUILD FALSE
    LINK_WHAT_YOU_USE FALSE
    CXX_COMPILER_LAUNCHER "${_trtmc_sam2_native_host_launcher}"
    CUDA_COMPILER_LAUNCHER "${_trtmc_sam2_native_host_launcher}"
    CXX_LINKER_LAUNCHER "${_trtmc_sam2_native_host_launcher}"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH FALSE
  )

  message(STATUS
    "SAM2 native-attention builder and diagnostic benchmark enabled "
    "(unqualified; TensorRT IAttentionV2, padded BHND, BF16 Q scale, "
    "softmax, noncausal, non-decomposable)")
endfunction()
