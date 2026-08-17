# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Model-owned SAM2 validation. Runtime registration is owned separately by
# src/runtime/models/sam2/MODEL.toml, while these targets exercise the native
# implementation and its fail-closed qualification boundaries.

function(_trtmc_add_sam2_test TEST_NAME)
  cmake_parse_arguments(
    ARG
    "REQUIRES_TRT;REQUIRES_GPU;NO_CTEST"
    ""
    "SOURCES;LINK_LIBRARIES;EXTRA_INCLUDES"
    ${ARGN}
  )
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "${TEST_NAME} requires at least one source")
  endif()

  set(_trtmc_sam2_test_sources ${ARG_SOURCES})
  list(POP_FRONT _trtmc_sam2_test_sources _trtmc_sam2_test_main)
  set(_trtmc_sam2_test_options MODEL_OWNED)
  if(ARG_REQUIRES_TRT)
    list(APPEND _trtmc_sam2_test_options REQUIRES_TRT)
  endif()
  if(ARG_REQUIRES_GPU)
    list(APPEND _trtmc_sam2_test_options REQUIRES_GPU)
  endif()
  if(ARG_NO_CTEST)
    list(APPEND _trtmc_sam2_test_options NO_CTEST)
  endif()

  trtmc_add_test(
    ${TEST_NAME}
    SOURCE "${_trtmc_sam2_test_main}"
    EXTRA_INCLUDES
      "${PROJECT_SOURCE_DIR}"
      "${PROJECT_SOURCE_DIR}/tests/cpp/models/sam2"
      "${PROJECT_SOURCE_DIR}/tools/sam2_native_builder"
      ${ARG_EXTRA_INCLUDES}
    ${_trtmc_sam2_test_options}
  )
  if(NOT TARGET ${TEST_NAME})
    return()
  endif()

  foreach(_trtmc_sam2_test_source IN LISTS _trtmc_sam2_test_sources)
    if(IS_ABSOLUTE "${_trtmc_sam2_test_source}")
      target_sources(${TEST_NAME} PRIVATE "${_trtmc_sam2_test_source}")
    else()
      target_sources(
        ${TEST_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/${_trtmc_sam2_test_source}")
    endif()
  endforeach()
  if(ARG_LINK_LIBRARIES)
    target_link_libraries(${TEST_NAME} PRIVATE ${ARG_LINK_LIBRARIES})
  endif()
  if(ARG_REQUIRES_GPU AND NOT ARG_NO_CTEST)
    set_tests_properties(${TEST_NAME} PROPERTIES LABELS "model;gpu")
  endif()
endfunction()

function(trtmc_add_sam2_tests)
  add_test(
    NAME test_runtime_model_data_manifest_paths
    COMMAND "${CMAKE_COMMAND}"
      "-DTRTMC_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
      -P "${PROJECT_SOURCE_DIR}/tests/cmake/test_runtime_model_data_manifest_paths.cmake"
  )
  set_tests_properties(test_runtime_model_data_manifest_paths PROPERTIES LABELS "model")

  add_test(
    NAME test_sam2_public_c_api_install_tree
    COMMAND "${CMAKE_COMMAND}"
      "-DTRTMC_BUILD_DIR=${PROJECT_BINARY_DIR}"
      "-DTRTMC_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
      "-DTRTMC_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}"
      -P "${PROJECT_SOURCE_DIR}/tests/cmake/test_sam2_public_c_api_install_tree.cmake"
  )
  set_tests_properties(test_sam2_public_c_api_install_tree PROPERTIES LABELS "model")

  add_test(
    NAME test_sam2_qualification_data_install_tree
    COMMAND "${CMAKE_COMMAND}"
      "-DTRTMC_BUILD_DIR=${PROJECT_BINARY_DIR}"
      "-DTRTMC_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
      "-DTRTMC_INSTALL_DATADIR=${CMAKE_INSTALL_DATADIR}"
      -P "${PROJECT_SOURCE_DIR}/tests/cmake/test_sam2_qualification_data_install_tree.cmake"
  )
  set_tests_properties(test_sam2_qualification_data_install_tree PROPERTIES LABELS "model")

  set(_trtmc_sam2_runtime_device_sources
    src/runtime/models/sam2/sam2_native_video_processor.cpp
    src/runtime/models/sam2/sam2_video_session.cpp
    src/runtime/models/sam2/sam2_bbox_postprocess.cpp
    src/runtime/models/sam2/sam2_preprocess.cpp
    src/runtime/models/sam2/sam2_preprocess_cuda.cu
    src/runtime/models/sam2/sam2_mask_postprocess.cpp
    src/runtime/models/sam2/sam2_device_workspace.cpp
    src/runtime/models/sam2/sam2_mask_postprocess_cuda.cu
  )

  _trtmc_add_sam2_test(test_sam2_engine_contract
    SOURCES tests/cpp/models/sam2/test_sam2_engine_contract.cpp)
  _trtmc_add_sam2_test(test_sam2_benchmark_protocol
    SOURCES
      tests/cpp/models/sam2/test_sam2_benchmark_protocol.cpp
      tools/sam2_native_benchmark/sam2_benchmark_accuracy.cpp
      tools/sam2_native_benchmark/sam2_benchmark_protocol.cpp
      tools/sam2_native_builder/durable_file_writer.cpp
      src/runtime/models/sam2/sam2_video_session.cpp
    LINK_LIBRARIES nlohmann_json::nlohmann_json)
  _trtmc_add_sam2_test(test_sam2_receipt_writer
    SOURCES
      tests/cpp/models/sam2/test_sam2_receipt_writer.cpp
      tools/sam2_native_benchmark/sam2_benchmark_protocol.cpp
      tools/sam2_native_builder/durable_file_writer.cpp
    LINK_LIBRARIES nlohmann_json::nlohmann_json)
  if(TARGET test_sam2_receipt_writer AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(test_sam2_receipt_writer PRIVATE
      "LINKER:--wrap=write"
      "LINKER:--wrap=fsync"
      "LINKER:--wrap=linkat"
      "LINKER:--wrap=unlinkat")
  endif()
  _trtmc_add_sam2_test(test_sam2_bbox_postprocess
    SOURCES
      tests/cpp/models/sam2/test_sam2_bbox_postprocess.cpp
      src/runtime/models/sam2/sam2_bbox_postprocess.cpp)
  _trtmc_add_sam2_test(test_sam2_checkpoint_reader
    SOURCES
      tests/cpp/models/sam2/test_sam2_checkpoint_reader.cpp
      tools/sam2_native_builder/checkpoint_reader.cpp)
  _trtmc_add_sam2_test(test_sam2_bundle_writer
    SOURCES
      tests/cpp/models/sam2/test_sam2_bundle_writer.cpp
      tools/sam2_native_builder/bundle_writer.cpp
      tools/sam2_native_builder/durable_file_writer.cpp)
  if(TARGET test_sam2_bundle_writer AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(test_sam2_bundle_writer PRIVATE
      "LINKER:--wrap=fsync"
      "LINKER:--wrap=unlinkat")
  endif()
  _trtmc_add_sam2_test(test_sam2_engine_builder
    SOURCES
      tests/cpp/models/sam2/test_sam2_engine_builder.cpp
      tools/sam2_native_builder/sam2_engine_builder.cpp
      tools/sam2_native_builder/bundle_writer.cpp
      tools/sam2_native_builder/durable_file_writer.cpp
      tools/sam2_native_builder/checkpoint_reader.cpp)
  _trtmc_add_sam2_test(test_sam2_golden_fixture
    SOURCES
      tests/cpp/models/sam2/test_sam2_golden_fixture.cpp
      tests/cpp/models/sam2/sam2_golden_fixture.cpp
    LINK_LIBRARIES nlohmann_json::nlohmann_json)
  set_tests_properties(test_sam2_golden_fixture PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
  _trtmc_add_sam2_test(test_sam2_native_bundle_loader
    SOURCES
      tests/cpp/models/sam2/test_sam2_native_bundle_loader.cpp
      src/runtime/models/sam2/sam2_native_bundle_loader.cpp
      src/runtime/models/sam2/sam2_qualification_authority.cpp
      tools/sam2_native_benchmark/sam2_empty_qualification_pin_provider.cpp
    LINK_LIBRARIES nlohmann_json::nlohmann_json)
  target_compile_definitions(test_sam2_native_bundle_loader PRIVATE
    TRTMC_SAM2_TEST_QUALIFICATION_AUTHORITY=1)
  _trtmc_add_sam2_test(test_sam2_preprocess_postprocess
    SOURCES
      tests/cpp/models/sam2/test_sam2_preprocess_postprocess.cpp
      src/runtime/models/sam2/sam2_preprocess.cpp
      src/runtime/models/sam2/sam2_mask_postprocess.cpp)
  _trtmc_add_sam2_test(test_sam2_tracker_contract
    SOURCES
      tests/cpp/models/sam2/test_sam2_tracker_contract.cpp
      tools/sam2_native_builder/sam2_tracker_contract.cpp
      tools/sam2_native_builder/checkpoint_reader.cpp)
  _trtmc_add_sam2_test(test_sam2_video_session
    SOURCES
      tests/cpp/models/sam2/test_sam2_video_session.cpp
      src/runtime/models/sam2/sam2_video_session.cpp)

  find_package(JPEG QUIET)
  if(TARGET JPEG::JPEG)
    _trtmc_add_sam2_test(test_sam2_jpeg_decoder
      SOURCES
        tests/cpp/models/sam2/test_sam2_jpeg_decoder.cpp
        src/runtime/models/sam2/sam2_jpeg_decoder.cpp
      LINK_LIBRARIES JPEG::JPEG)
  else()
    message(STATUS "Skipping test_sam2_jpeg_decoder: libjpeg development files not available")
  endif()

  _trtmc_add_sam2_test(test_sam2_image_network_contract
    SOURCES
      tests/cpp/models/sam2/test_sam2_image_network_contract.cpp
      tools/sam2_native_builder/checkpoint_reader.cpp
      tools/sam2_native_builder/sam2_trt_layers.cpp
      tools/sam2_native_builder/sam2_image_network.cpp
    REQUIRES_TRT)

  # These diagnostics need delivered inputs. Keep compile coverage in
  # trtmc_cpp_tests, but do not create fixture-free CTest entries that would
  # only return a usage error.
  _trtmc_add_sam2_test(test_sam2_image_network_inference
    SOURCES
      tests/cpp/models/sam2/test_sam2_image_network_inference.cpp
      tools/sam2_native_builder/checkpoint_reader.cpp
      tools/sam2_native_builder/sam2_trt_layers.cpp
      tools/sam2_native_builder/sam2_image_network.cpp
      src/runtime/models/sam2/sam2_preprocess.cpp
      src/runtime/models/sam2/sam2_bbox_postprocess.cpp
    REQUIRES_TRT REQUIRES_GPU NO_CTEST)
  _trtmc_add_sam2_test(test_sam2_tracker_network_build
    SOURCES
      tests/cpp/models/sam2/test_sam2_tracker_network_build.cpp
      tools/sam2_native_builder/checkpoint_reader.cpp
      tools/sam2_native_builder/sam2_tracker_contract.cpp
      tools/sam2_native_builder/sam2_tracker_network.cpp
      tools/sam2_native_builder/sam2_trt_layers.cpp
    REQUIRES_TRT NO_CTEST)

  if(CMAKE_CUDA_COMPILER)
    _trtmc_add_sam2_test(test_sam2_preprocess_normalization_cuda
      SOURCES
        tests/cpp/models/sam2/test_sam2_preprocess_normalization_cuda.cpp
        src/runtime/models/sam2/sam2_preprocess.cpp
        src/runtime/models/sam2/sam2_preprocess_cuda.cu
      REQUIRES_GPU)
    _trtmc_add_sam2_test(test_sam2_mask_postprocess_cuda
      SOURCES
        tests/cpp/models/sam2/test_sam2_mask_postprocess_cuda.cpp
        src/runtime/models/sam2/sam2_mask_postprocess.cpp
        src/runtime/models/sam2/sam2_mask_postprocess_cuda.cu
      REQUIRES_GPU)
    _trtmc_add_sam2_test(test_sam2_native_video_processor
      SOURCES
        tests/cpp/models/sam2/test_sam2_native_video_processor.cpp
        ${_trtmc_sam2_runtime_device_sources}
      LINK_LIBRARIES ${CMAKE_DL_LIBS})
    _trtmc_add_sam2_test(test_sam2_native_device_video_processor
      SOURCES
        tests/cpp/models/sam2/test_sam2_native_device_video_processor.cpp
        ${_trtmc_sam2_runtime_device_sources}
      LINK_LIBRARIES ${CMAKE_DL_LIBS}
      REQUIRES_GPU)
    _trtmc_add_sam2_test(test_sam2_native_five_frame_inference
      SOURCES
        tests/cpp/models/sam2/test_sam2_native_five_frame_inference.cpp
        tools/sam2_native_builder/checkpoint_reader.cpp
        tools/sam2_native_builder/sam2_trt_layers.cpp
        tools/sam2_native_builder/sam2_image_network.cpp
        tools/sam2_native_builder/sam2_tracker_contract.cpp
        tools/sam2_native_builder/sam2_tracker_network.cpp
        ${_trtmc_sam2_runtime_device_sources}
        tests/cpp/models/sam2/sam2_golden_fixture.cpp
      LINK_LIBRARIES nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS}
      REQUIRES_TRT REQUIRES_GPU NO_CTEST)
  else()
    message(STATUS "Skipping CUDA-backed SAM2 tests: CUDA compiler not available")
  endif()
endfunction()
