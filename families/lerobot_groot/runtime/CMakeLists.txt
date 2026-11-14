# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_lerobot_groot SHARED
  bpe_tokenizer.cpp
  image.cpp
  pipeline.cpp
  plugin.cpp
  prompt.cpp
)

target_include_directories(trtmc_model_lerobot_groot
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_lerobot_groot SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_lerobot_groot PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
# LeRobot's pre/postprocessing rounds after every float32 operation; keep the compiler from
# fusing them into FMAs so the results match bit for bit.
target_compile_options(trtmc_model_lerobot_groot PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>"
)
set_target_properties(trtmc_model_lerobot_groot PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_lerobot_groot
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(trtmc_lerobot_groot_qualification
    ${PROJECT_SOURCE_DIR}/families/lerobot_groot/tests/cpp/qualification.cpp
  )
  target_include_directories(trtmc_lerobot_groot_qualification PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(trtmc_lerobot_groot_qualification PRIVATE trtmc_c)
  target_compile_options(trtmc_lerobot_groot_qualification PRIVATE -Wall -Wextra -Wpedantic)
  add_dependencies(trtmc_lerobot_groot_qualification
    trtmc_model_lerobot_groot trtmc_backend_trt)

  # Run by test_e2e.py against the pinned fixtures, so it is a target rather than a CTest case.
  add_executable(trtmc_lerobot_groot_preprocess_check
    ${PROJECT_SOURCE_DIR}/families/lerobot_groot/tests/cpp/preprocess_check.cpp
    ${PROJECT_SOURCE_DIR}/families/lerobot_groot/runtime/bpe_tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/families/lerobot_groot/runtime/image.cpp
    ${PROJECT_SOURCE_DIR}/families/lerobot_groot/runtime/prompt.cpp
  )
  target_include_directories(trtmc_lerobot_groot_preprocess_check PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(trtmc_lerobot_groot_preprocess_check PRIVATE nlohmann_json::nlohmann_json)
  target_compile_options(trtmc_lerobot_groot_preprocess_check PRIVATE
    -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)

  add_executable(test_lerobot_groot_prompt
    ${PROJECT_SOURCE_DIR}/families/lerobot_groot/tests/cpp/test_prompt.cpp
    ${PROJECT_SOURCE_DIR}/families/lerobot_groot/runtime/prompt.cpp
  )
  target_include_directories(test_lerobot_groot_prompt PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(test_lerobot_groot_prompt PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME lerobot_groot_prompt COMMAND test_lerobot_groot_prompt)
  set_tests_properties(lerobot_groot_prompt PROPERTIES LABELS "cpu")

  add_executable(test_lerobot_groot_image
    ${PROJECT_SOURCE_DIR}/families/lerobot_groot/tests/cpp/test_image.cpp
    ${PROJECT_SOURCE_DIR}/families/lerobot_groot/runtime/image.cpp
  )
  target_include_directories(test_lerobot_groot_image PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(test_lerobot_groot_image PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
  add_test(NAME lerobot_groot_image COMMAND test_lerobot_groot_image)
  set_tests_properties(lerobot_groot_image PROPERTIES LABELS "cpu")
endif()
