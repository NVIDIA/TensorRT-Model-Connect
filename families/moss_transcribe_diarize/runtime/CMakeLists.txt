# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_moss_transcribe_diarize SHARED
  log_mel.cpp
  pipeline.cpp
  plugin.cpp
  prompt_plan.cpp
  runtime_config.cpp
  text_output.cpp
)

target_include_directories(trtmc_model_moss_transcribe_diarize
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_moss_transcribe_diarize SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_moss_transcribe_diarize PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  Threads::Threads
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_moss_transcribe_diarize PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_moss_transcribe_diarize PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_moss_transcribe_diarize
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_moss_transcribe_diarize_host
    ${PROJECT_SOURCE_DIR}/families/moss_transcribe_diarize/tests/cpp/test_host.cpp
  )
  target_include_directories(test_moss_transcribe_diarize_host PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_moss_transcribe_diarize_host SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_moss_transcribe_diarize_host PRIVATE
    trtmc_model_moss_transcribe_diarize
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_moss_transcribe_diarize_host PRIVATE
    -Wall -Wextra -Wpedantic -Werror
  )
  add_test(NAME moss_transcribe_diarize_host COMMAND test_moss_transcribe_diarize_host)
  set_tests_properties(moss_transcribe_diarize_host PROPERTIES LABELS "cpu")

  # Direct C and C++ SDK calls of the speech_transcription Task, run by the family E2E.
  foreach(_language IN ITEMS c cpp)
    add_executable(test_moss_transcribe_diarize_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/moss_transcribe_diarize/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_moss_transcribe_diarize_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_moss_transcribe_diarize_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_moss_transcribe_diarize_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  target_link_libraries(test_moss_transcribe_diarize_sdk_cpp PRIVATE nlohmann_json::nlohmann_json)
  add_dependencies(test_moss_transcribe_diarize_host
    test_moss_transcribe_diarize_sdk_c test_moss_transcribe_diarize_sdk_cpp
  )
endif()
