# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_owlv2 SHARED
  bpe_tokenizer.cpp
  image_preprocess_seam.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_owlv2
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_owlv2 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_owlv2 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_owlv2 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_owlv2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_owlv2
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_owlv2_image_preprocess_seam
    ${PROJECT_SOURCE_DIR}/families/owlv2/tests/cpp/test_image_preprocess_seam.cpp
  )
  target_include_directories(test_owlv2_image_preprocess_seam PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_owlv2_image_preprocess_seam PRIVATE
    trtmc_model_owlv2
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_owlv2_image_preprocess_seam PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(
    NAME owlv2_image_preprocess_seam
    COMMAND test_owlv2_image_preprocess_seam
  )
endif()
