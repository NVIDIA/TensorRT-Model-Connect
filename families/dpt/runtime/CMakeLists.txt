# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_dpt SHARED
  image_ops.cpp
  pipeline.cpp
  plugin.cpp
)
# The image operations reproduce reference arithmetic operation by operation;
# fusing a multiply and an add would round differently.
set_source_files_properties(image_ops.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

target_include_directories(trtmc_model_dpt
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_dpt SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_dpt PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_dpt PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_dpt PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_dpt
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
# The library carries code adapted from Pillow and PyTorch; their notices ship with it.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../NOTICE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/trtmc/families/dpt
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_dpt_image_ops
    ${PROJECT_SOURCE_DIR}/families/dpt/tests/cpp/test_image_ops.cpp
    image_ops.cpp
  )
  target_include_directories(test_dpt_image_ops PRIVATE
    ${PROJECT_SOURCE_DIR}
  )
  target_compile_options(test_dpt_image_ops PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(
    NAME dpt_image_ops
    COMMAND test_dpt_image_ops
  )

  add_executable(test_dpt_pipeline
    ${PROJECT_SOURCE_DIR}/families/dpt/tests/cpp/test_pipeline.cpp
    image_ops.cpp
    pipeline.cpp
  )
  target_include_directories(test_dpt_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_dpt_pipeline SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_dpt_pipeline PRIVATE
    trtmc_core
    nlohmann_json::nlohmann_json
  )
  target_compile_options(test_dpt_pipeline PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(
    NAME dpt_pipeline
    COMMAND test_dpt_pipeline
  )
endif()
