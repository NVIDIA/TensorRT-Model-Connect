# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The pinned SDK declares these targets but does not install them. Run after
# its top-level CMakeLists so we use the original targets and native toolchain.
function(_trtmc_install_edge_policy_sdk)
  set_property(DIRECTORY "${CMAKE_SOURCE_DIR}/experimental_models" PROPERTY EXCLUDE_FROM_ALL TRUE)
  # Leave unrelated experimental libraries and examples out of the SDK build.
  add_custom_target(trtmc_edge_policy_sdk ALL DEPENDS gr00tExperimental pi05Experimental)
  foreach(_policy IN ITEMS gr00tExperimental pi05Experimental)
    get_target_property(_headers ${_policy} SOURCE_DIR)
    get_filename_component(_parent "${_headers}" DIRECTORY)
    get_filename_component(_name "${_parent}" NAME)
    install(TARGETS ${_policy} ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT EdgePolicySDK)
    install(DIRECTORY "${_headers}/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/tensorrt_edgellm/${_name}"
      COMPONENT EdgePolicySDK FILES_MATCHING PATTERN "*.h")
  endforeach()
endfunction()

cmake_language(DEFER CALL _trtmc_install_edge_policy_sdk)
