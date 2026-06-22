function(_trtmc_model_manifest_string manifest_text key output_var)
  string(REGEX MATCH "(^|\n)[ \t]*${key}[ \t]*=[ \t]*\"([^\"]+)\"" _trtmc_match
    "${manifest_text}")
  if(_trtmc_match)
    set(${output_var} "${CMAKE_MATCH_2}" PARENT_SCOPE)
  else()
    set(${output_var} "" PARENT_SCOPE)
  endif()
endfunction()

function(_trtmc_model_manifest_list manifest_text key output_var)
  string(REGEX MATCH "(^|\n)[ \t]*${key}[ \t]*=[ \t]*\\[([^]]*)\\]" _trtmc_match
    "${manifest_text}")
  if(_trtmc_match)
    string(REGEX MATCHALL "\"[^\"]+\"" _trtmc_tokens "${CMAKE_MATCH_2}")
    set(_trtmc_values)
    foreach(_trtmc_token IN LISTS _trtmc_tokens)
      string(REPLACE "\"" "" _trtmc_value "${_trtmc_token}")
      list(APPEND _trtmc_values "${_trtmc_value}")
    endforeach()
    set(${output_var} ${_trtmc_values} PARENT_SCOPE)
  else()
    set(${output_var} "" PARENT_SCOPE)
  endif()
endfunction()

file(GLOB TRTMC_RUNTIME_MODEL_MANIFESTS CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/src/runtime/models/*/MODEL.toml")
if(NOT TRTMC_RUNTIME_MODEL_MANIFESTS)
  message(FATAL_ERROR "No runtime model manifests found under src/runtime/models")
endif()

set(TRTMC_RUNTIME_MODEL_IDS)
foreach(_trtmc_model_manifest IN LISTS TRTMC_RUNTIME_MODEL_MANIFESTS)
  get_filename_component(_trtmc_model_dir "${_trtmc_model_manifest}" DIRECTORY)
  get_filename_component(_trtmc_model_folder "${_trtmc_model_dir}" NAME)
  file(READ "${_trtmc_model_manifest}" _trtmc_model_manifest_text)

  _trtmc_model_manifest_string("${_trtmc_model_manifest_text}" "id" _trtmc_model)
  if(NOT _trtmc_model)
    message(FATAL_ERROR "Missing id in runtime model manifest: ${_trtmc_model_manifest}")
  endif()
  if(NOT _trtmc_model STREQUAL _trtmc_model_folder)
    message(FATAL_ERROR
      "Runtime model manifest id '${_trtmc_model}' must match folder '${_trtmc_model_folder}': "
      "${_trtmc_model_manifest}")
  endif()

  _trtmc_model_manifest_string("${_trtmc_model_manifest_text}" "runtime_library"
    _trtmc_runtime_library)
  if(NOT _trtmc_runtime_library)
    set(_trtmc_runtime_library "libtrtmc_model_${_trtmc_model}.so")
  endif()

  _trtmc_model_manifest_list("${_trtmc_model_manifest_text}" "runtime_plugins"
    _trtmc_runtime_plugins)
  if(NOT _trtmc_runtime_plugins)
    message(FATAL_ERROR "No runtime_plugins in ${_trtmc_model_manifest}")
  endif()

  string(MAKE_C_IDENTIFIER "${_trtmc_model}" _trtmc_model_var)
  list(APPEND TRTMC_RUNTIME_MODEL_IDS "${_trtmc_model}")
  set(TRTMC_MODEL_${_trtmc_model_var}_LIBRARY "${_trtmc_runtime_library}")

  foreach(_trtmc_manifest_entry IN LISTS _trtmc_runtime_plugins)
    string(REPLACE "|" ";" _trtmc_fields "${_trtmc_manifest_entry}")
    list(LENGTH _trtmc_fields _trtmc_field_count)
    if(NOT _trtmc_field_count EQUAL 2)
      message(FATAL_ERROR
        "Invalid runtime_plugins entry '${_trtmc_manifest_entry}' in ${_trtmc_model_manifest}")
    endif()
    list(GET _trtmc_fields 0 _trtmc_source)
    list(GET _trtmc_fields 1 _trtmc_symbol)
    set(_trtmc_source_path "${PROJECT_SOURCE_DIR}/src/runtime/models/${_trtmc_model}/${_trtmc_source}")
    if(NOT EXISTS "${_trtmc_source_path}")
      message(FATAL_ERROR "Runtime plugin source does not exist: ${_trtmc_source_path}")
    endif()
    list(APPEND TRTMC_MODEL_${_trtmc_model_var}_PLUGINS "${_trtmc_source}|${_trtmc_symbol}")
  endforeach()
endforeach()
list(SORT TRTMC_RUNTIME_MODEL_IDS)

set(TRTMC_MODEL_PLUGIN_INDEX_ENTRIES)
set(TRTMC_RUNTIME_STRATEGY_IDS)
foreach(_trtmc_model IN LISTS TRTMC_RUNTIME_MODEL_IDS)
  set(_trtmc_model_manifest "${PROJECT_SOURCE_DIR}/src/runtime/models/${_trtmc_model}/MODEL.toml")
  if(NOT EXISTS "${_trtmc_model_manifest}")
    message(FATAL_ERROR "Missing runtime model manifest: ${_trtmc_model_manifest}")
  endif()
  file(READ "${_trtmc_model_manifest}" _trtmc_model_manifest_text)
  set(_trtmc_strategy_tokens)
  string(REGEX MATCH "runtime_strategies[ \t]*=[ \t]*\\[([^]]*)\\]" _trtmc_strategy_match
    "${_trtmc_model_manifest_text}")
  if(_trtmc_strategy_match)
    string(REGEX MATCHALL "\"[^\"]+\"" _trtmc_strategy_tokens "${CMAKE_MATCH_1}")
  else()
    string(REGEX MATCH "runtime_strategy[ \t]*=[ \t]*\"([^\"]+)\"" _trtmc_single_strategy_match
      "${_trtmc_model_manifest_text}")
    if(_trtmc_single_strategy_match)
      set(_trtmc_strategy_tokens "\"${CMAKE_MATCH_1}\"")
    endif()
  endif()
  if(NOT _trtmc_strategy_tokens)
    message(FATAL_ERROR "No runtime_strategies in ${_trtmc_model_manifest}")
  endif()
  string(MAKE_C_IDENTIFIER "${_trtmc_model}" _trtmc_model_var)
  foreach(_trtmc_quoted_strategy IN LISTS _trtmc_strategy_tokens)
    string(REPLACE "\"" "" _trtmc_strategy "${_trtmc_quoted_strategy}")
    list(FIND TRTMC_RUNTIME_STRATEGY_IDS "${_trtmc_strategy}" _trtmc_existing_strategy)
    string(MAKE_C_IDENTIFIER "${_trtmc_strategy}" _trtmc_strategy_var)
    set(_trtmc_strategy_owner_var "TRTMC_RUNTIME_STRATEGY_OWNER_${_trtmc_strategy_var}")
    if(NOT _trtmc_existing_strategy EQUAL -1)
      message(FATAL_ERROR
        "Duplicate runtime_strategy '${_trtmc_strategy}' in ${_trtmc_model_manifest}; "
        "already claimed by ${${_trtmc_strategy_owner_var}}")
    endif()
    list(APPEND TRTMC_RUNTIME_STRATEGY_IDS "${_trtmc_strategy}")
    set(${_trtmc_strategy_owner_var} "${_trtmc_model}")
    string(APPEND TRTMC_MODEL_PLUGIN_INDEX_ENTRIES
      "        {\"${_trtmc_model}\", \"${_trtmc_strategy}\", \"${TRTMC_MODEL_${_trtmc_model_var}_LIBRARY}\"},\n")
  endforeach()
endforeach()

set(TRTMC_MODEL_PLUGIN_INDEX_SOURCE
  "${PROJECT_BINARY_DIR}/generated/model_plugin_index.cpp")
configure_file("${CMAKE_CURRENT_LIST_DIR}/model_plugin_index.cpp.in"
  "${TRTMC_MODEL_PLUGIN_INDEX_SOURCE}" @ONLY)
set_source_files_properties("${TRTMC_MODEL_PLUGIN_INDEX_SOURCE}" PROPERTIES GENERATED TRUE)

set(TRTMC_MODEL_PLUGIN_REGISTRATION_SOURCES)
foreach(_trtmc_model IN LISTS TRTMC_RUNTIME_MODEL_IDS)
  string(MAKE_C_IDENTIFIER "${_trtmc_model}" _trtmc_model_var)
  set(TRTMC_MODEL_PLUGIN_ID "${_trtmc_model}")
  set(TRTMC_MODEL_PLUGIN_REGISTRATION_DECLS)
  set(TRTMC_MODEL_PLUGIN_REGISTRATION_CALLS)
  foreach(_trtmc_manifest IN LISTS TRTMC_MODEL_${_trtmc_model_var}_PLUGINS)
    string(REPLACE "|" ";" _trtmc_fields "${_trtmc_manifest}")
    list(GET _trtmc_fields 1 _trtmc_symbol)
    string(APPEND TRTMC_MODEL_PLUGIN_REGISTRATION_DECLS
      "void ${_trtmc_symbol}(::trtmc::PipelineRegistry& registry);\n")
    string(APPEND TRTMC_MODEL_PLUGIN_REGISTRATION_CALLS
      "    ::trtmc::${_trtmc_symbol}(*registry);\n")
  endforeach()
  set(_trtmc_generated_model_reg
    "${PROJECT_BINARY_DIR}/generated/models/${_trtmc_model}/register_model_plugin.cpp")
  get_filename_component(_trtmc_generated_model_reg_dir "${_trtmc_generated_model_reg}" DIRECTORY)
  file(MAKE_DIRECTORY "${_trtmc_generated_model_reg_dir}")
  configure_file("${CMAKE_CURRENT_LIST_DIR}/register_model_plugin.cpp.in"
    "${_trtmc_generated_model_reg}" @ONLY)
  set_source_files_properties("${_trtmc_generated_model_reg}" PROPERTIES GENERATED TRUE)
  set(TRTMC_MODEL_${_trtmc_model_var}_REGISTRATION_SOURCE "${_trtmc_generated_model_reg}")
  list(APPEND TRTMC_MODEL_PLUGIN_REGISTRATION_SOURCES "${_trtmc_generated_model_reg}")
endforeach()
