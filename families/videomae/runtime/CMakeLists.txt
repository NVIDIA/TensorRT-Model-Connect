# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_videomae SHARED
  bundle_config.cpp
  clip.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_videomae
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_videomae SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_videomae PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_videomae PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_videomae PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_videomae
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
# The family's NOTICE (the sources it follows, the checkpoint's license) ships with the library.
install(FILES ${PROJECT_SOURCE_DIR}/families/videomae/NOTICE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/trtmc/families/videomae
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_videomae_clip
    ${PROJECT_SOURCE_DIR}/families/videomae/tests/cpp/test_clip.cpp
  )
  target_include_directories(test_videomae_clip PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_videomae_clip SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_videomae_clip PRIVATE trtmc_model_videomae nlohmann_json::nlohmann_json)
  target_compile_options(test_videomae_clip PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME videomae_clip COMMAND test_videomae_clip)
  set_tests_properties(videomae_clip PROPERTIES LABELS "cpu")

  foreach(_language IN ITEMS c cpp)
    add_executable(test_videomae_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/videomae/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_videomae_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_videomae_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_videomae_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    # A clip whose byte count overflows is refused before any allocation or load.
    add_test(NAME videomae_sdk_${_language}_refuses_an_oversized_clip
      COMMAND test_videomae_sdk_${_language} bundle runtime frames u8 16 1073741824 1073741824 mask out
    )
    set_tests_properties(videomae_sdk_${_language}_refuses_an_oversized_clip PROPERTIES
      LABELS "cpu" PASS_REGULAR_EXPRESSION "clip size overflows host storage"
    )
  endforeach()
endif()
