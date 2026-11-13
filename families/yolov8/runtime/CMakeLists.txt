# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_yolov8 SHARED
  image_preprocess_seam.cpp
  pipeline.cpp
  plugin.cpp
)
set_source_files_properties(image_preprocess_seam.cpp PROPERTIES
  COMPILE_OPTIONS "-Wno-missing-field-initializers;-Wno-pedantic;-Wno-unused-function"
)
target_include_directories(trtmc_model_yolov8 PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_yolov8 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_yolov8 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_yolov8 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_yolov8 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_yolov8
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_yolov8_task_contract
    ${PROJECT_SOURCE_DIR}/families/yolov8/tests/cpp/test_task_contract.cpp
  )
  target_include_directories(test_yolov8_task_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_yolov8_task_contract SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_yolov8_task_contract PRIVATE
    trtmc_model_yolov8
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_yolov8_task_contract PRIVATE
    -Wall -Wextra -Wpedantic -Werror
  )
  add_test(NAME yolov8_task_contract COMMAND test_yolov8_task_contract)
  set_tests_properties(yolov8_task_contract PROPERTIES LABELS "cpu")

  # Public SDK consumers for the selected single-device E2E. They link only the
  # public C runtime, and building the CPU contract builds them too.
  foreach(_language IN ITEMS c cpp)
    add_executable(test_yolov8_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/yolov8/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_yolov8_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_yolov8_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_yolov8_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_yolov8_task_contract
    test_yolov8_sdk_c
    test_yolov8_sdk_cpp
  )

  add_executable(test_yolov8_image_preprocess
    ${PROJECT_SOURCE_DIR}/families/yolov8/tests/cpp/test_image_preprocess_seam.cpp
  )
  target_include_directories(test_yolov8_image_preprocess PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_yolov8_image_preprocess PRIVATE
    trtmc_model_yolov8
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_yolov8_image_preprocess PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(
    NAME yolov8_image_preprocess
    COMMAND test_yolov8_image_preprocess
  )
endif()
