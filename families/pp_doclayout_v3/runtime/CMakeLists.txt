# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Only this family needs OpenCV core/imgproc for contour postprocessing.
find_path(TRTMC_LAYOUT_OPENCV_INCLUDE opencv2/imgproc.hpp PATH_SUFFIXES opencv4)
find_library(TRTMC_LAYOUT_OPENCV_CORE NAMES opencv_core)
find_library(TRTMC_LAYOUT_OPENCV_IMGPROC NAMES opencv_imgproc)
if(NOT TRTMC_LAYOUT_OPENCV_INCLUDE OR NOT TRTMC_LAYOUT_OPENCV_CORE OR NOT TRTMC_LAYOUT_OPENCV_IMGPROC)
  message(STATUS "Skipping pp_doclayout_v3: install OpenCV core and imgproc development packages")
  return()
endif()

add_library(trtmc_model_pp_doclayout_v3 SHARED layout.cpp plugin.cpp)
target_include_directories(trtmc_model_pp_doclayout_v3 PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_pp_doclayout_v3 SYSTEM PRIVATE ${TRTMC_LAYOUT_OPENCV_INCLUDE})
target_link_libraries(trtmc_model_pp_doclayout_v3 PRIVATE trtmc_core nlohmann_json::nlohmann_json
  ${TRTMC_LAYOUT_OPENCV_CORE} ${TRTMC_LAYOUT_OPENCV_IMGPROC})
target_compile_options(trtmc_model_pp_doclayout_v3 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_pp_doclayout_v3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_pp_doclayout_v3
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime)

if(TRTMC_BUILD_TESTS)
  add_executable(test_pp_doclayout_v3_layout ../tests/cpp/test_layout.cpp layout.cpp)
  target_include_directories(test_pp_doclayout_v3_layout PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(test_pp_doclayout_v3_layout SYSTEM PRIVATE ${TRTMC_LAYOUT_OPENCV_INCLUDE})
  target_link_libraries(test_pp_doclayout_v3_layout PRIVATE
    ${TRTMC_LAYOUT_OPENCV_CORE} ${TRTMC_LAYOUT_OPENCV_IMGPROC})
  add_test(NAME pp_doclayout_v3_layout COMMAND test_pp_doclayout_v3_layout)
endif()
