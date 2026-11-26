# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_glm_ocr SHARED
  image.cpp
  model.cpp
  plugin.cpp
  prompt.cpp
  tokenizer.cpp
)

target_include_directories(trtmc_model_glm_ocr
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_glm_ocr SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_glm_ocr PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_glm_ocr PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_glm_ocr PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_glm_ocr
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
# Third-party notices: adapted code (NOTICE) and the Unicode data in unicode_tables.h.
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/../NOTICE
  ${CMAKE_CURRENT_SOURCE_DIR}/../UNICODE_LICENSE.txt
  DESTINATION ${CMAKE_INSTALL_DATADIR}/trtmc/families/glm_ocr
)

if(TRTMC_BUILD_TESTS)
  # One test executable: run without arguments it is the unit test; with a command it is the
  # host-parity probe of tests/test_e2e.py (built by CI through this CTest registration).
  add_executable(test_glm_ocr_runtime
    ${PROJECT_SOURCE_DIR}/families/glm_ocr/tests/cpp/test_runtime.cpp
    ${PROJECT_SOURCE_DIR}/families/glm_ocr/tests/cpp/host_probe.cpp
  )
  target_include_directories(test_glm_ocr_runtime PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_glm_ocr_runtime SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_glm_ocr_runtime PRIVATE
    trtmc_model_glm_ocr
    trtmc_core
    nlohmann_json::nlohmann_json
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_glm_ocr_runtime PRIVATE
    -Wall -Wextra -Wpedantic
  )
  set_target_properties(test_glm_ocr_runtime PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "${CMAKE_BINARY_DIR}"
  )
  add_test(
    NAME glm_ocr_runtime
    COMMAND test_glm_ocr_runtime
  )
endif()
