# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared helper for declarative source manifests that also need generated
# registration sources.

function(trtmc_configure_registration_manifest manifest_var source_dir template_path generated_source
         sources_var decls_var calls_var call_indent registry_type)
  set(_trtmc_sources)
  set(_trtmc_decls)
  set(_trtmc_calls)

  foreach(_trtmc_entry IN LISTS ${manifest_var})
    string(REPLACE "|" ";" _trtmc_fields "${_trtmc_entry}")
    list(LENGTH _trtmc_fields _trtmc_field_count)
    if(NOT _trtmc_field_count EQUAL 2)
      message(FATAL_ERROR "Invalid TRTMC registration manifest entry: ${_trtmc_entry}")
    endif()

    list(GET _trtmc_fields 0 _trtmc_source)
    list(GET _trtmc_fields 1 _trtmc_symbol)

    set(_trtmc_source_path "${source_dir}/${_trtmc_source}")
    if(NOT EXISTS "${_trtmc_source_path}")
      message(FATAL_ERROR "TRTMC registration manifest source does not exist: ${_trtmc_source_path}")
    endif()

    list(APPEND _trtmc_sources "${_trtmc_source_path}")
    string(APPEND _trtmc_decls "void ${_trtmc_symbol}(${registry_type}& registry);\n")
    string(APPEND _trtmc_calls "${call_indent}${_trtmc_symbol}(registry);\n")
  endforeach()

  set(${sources_var} ${_trtmc_sources} PARENT_SCOPE)
  set(${decls_var} "${_trtmc_decls}" PARENT_SCOPE)
  set(${calls_var} "${_trtmc_calls}" PARENT_SCOPE)

  set(${decls_var} "${_trtmc_decls}")
  set(${calls_var} "${_trtmc_calls}")
  get_filename_component(_trtmc_generated_dir "${generated_source}" DIRECTORY)
  file(MAKE_DIRECTORY "${_trtmc_generated_dir}")
  configure_file("${template_path}" "${generated_source}" @ONLY)
endfunction()
