# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_sam2 SHARED
  plugin.cpp
  sam2_bbox_postprocess.cpp
  sam2_device_workspace.cpp
  sam2_mask_postprocess_cuda.cu
  sam2_native_video_processor.cpp
  sam2_pipeline.cpp
  sam2_preprocess.cpp
  sam2_preprocess_cuda.cu
  sam2_video_session.cpp
  ${TRTMC_MODEL_sam2_REGISTRATION_SOURCE}
)
target_include_directories(trtmc_model_sam2
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_sam2 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_sam2 PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_definitions(trtmc_model_sam2 PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_sam2 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_sam2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/sam2"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_sam2_bbox_postprocess
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_sam2_bbox_postprocess.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR};${CMAKE_CURRENT_LIST_DIR}/include"
    MODEL_OWNED
  )
  trtmc_add_test(test_sam2_preprocess_postprocess
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_sam2_preprocess_postprocess.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR};${CMAKE_CURRENT_LIST_DIR}/include"
    MODEL_OWNED
  )
  trtmc_add_test(test_sam2_runtime_registration
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_sam2_runtime_registration.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR};${CMAKE_CURRENT_LIST_DIR}/include"
    MODEL_OWNED
  )
  trtmc_add_test(test_sam2_video_session
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_sam2_video_session.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR};${CMAKE_CURRENT_LIST_DIR}/include"
    MODEL_OWNED
  )
  foreach(_sam2_test IN ITEMS
      test_sam2_bbox_postprocess
      test_sam2_preprocess_postprocess
      test_sam2_runtime_registration
      test_sam2_video_session)
    if(TARGET ${_sam2_test})
      target_link_libraries(${_sam2_test} PRIVATE trtmc_model_sam2)
    endif()
  endforeach()
endif()
