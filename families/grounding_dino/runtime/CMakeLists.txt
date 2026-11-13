# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_grounding_dino SHARED
  image_preprocess_seam.cpp
  pipeline.cpp
  plugin.cpp
  text_inputs.cpp
  wordpiece_tokenizer.cpp
)

target_include_directories(trtmc_model_grounding_dino
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_grounding_dino SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_grounding_dino PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_grounding_dino PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_grounding_dino PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_grounding_dino
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_grounding_dino_text_inputs
    ${PROJECT_SOURCE_DIR}/families/grounding_dino/tests/cpp/test_text_inputs.cpp
  )
  target_include_directories(test_grounding_dino_text_inputs PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_grounding_dino_text_inputs PRIVATE
    trtmc_model_grounding_dino
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_grounding_dino_text_inputs PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(
    NAME grounding_dino_text_inputs
    COMMAND test_grounding_dino_text_inputs
  )
endif()
