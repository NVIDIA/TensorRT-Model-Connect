# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(_trtmc_model_data_relative_path_is_safe value output_var)
  if("${value}" STREQUAL "" OR IS_ABSOLUTE "${value}" OR
     NOT "${value}" MATCHES
       "^[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)*$")
    set(${output_var} FALSE PARENT_SCOPE)
  else()
    set(${output_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

function(_trtmc_model_manifest_safe_relative_path_list
    manifest_text key manifest_path output_var)
  string(REGEX MATCH "(^|\n)[ \t]*${key}[ \t]*=[ \t]*\\[([^]]*)\\]" _trtmc_match
    "${manifest_text}")
  if(NOT _trtmc_match)
    set(${output_var} "" PARENT_SCOPE)
    return()
  endif()

  set(_trtmc_raw_values "${CMAKE_MATCH_2}")
  if(_trtmc_raw_values MATCHES "[;\\\\]")
    message(FATAL_ERROR
      "Invalid ${key} syntax in ${manifest_path}: semicolons and backslashes are forbidden")
  endif()
  string(REGEX REPLACE "\"[^\"]+\"" "" _trtmc_array_syntax "${_trtmc_raw_values}")
  string(REGEX REPLACE "[ \t\r\n,]" "" _trtmc_array_syntax "${_trtmc_array_syntax}")
  if(NOT _trtmc_array_syntax STREQUAL "")
    message(FATAL_ERROR
      "Invalid ${key} syntax in ${manifest_path}: expected a list of quoted relative paths")
  endif()

  string(REGEX MATCHALL "\"[^\"]+\"" _trtmc_tokens "${_trtmc_raw_values}")
  set(_trtmc_values)
  foreach(_trtmc_token IN LISTS _trtmc_tokens)
    string(REPLACE "\"" "" _trtmc_value "${_trtmc_token}")
    _trtmc_model_data_relative_path_is_safe("${_trtmc_value}" _trtmc_is_safe)
    if(NOT _trtmc_is_safe)
      message(FATAL_ERROR
        "Invalid ${key} entry '${_trtmc_value}' in ${manifest_path}: expected a safe relative path")
    endif()
    list(FIND _trtmc_values "${_trtmc_value}" _trtmc_duplicate_index)
    if(NOT _trtmc_duplicate_index EQUAL -1)
      message(FATAL_ERROR
        "Duplicate ${key} entry '${_trtmc_value}' in ${manifest_path}")
    endif()
    list(APPEND _trtmc_values "${_trtmc_value}")
  endforeach()
  set(${output_var} ${_trtmc_values} PARENT_SCOPE)
endfunction()
