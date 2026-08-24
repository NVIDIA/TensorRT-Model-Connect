# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(_required
    TRTMC_CLI
    TRTMC_DATASET_BENCHMARK
    TRTMC_BENCHMARK_WORKER
    TRTMC_TEST_DIRECTORY)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

function(_trtmc_require_contains LABEL TEXT EXPECTED)
  string(FIND "${TEXT}" "${EXPECTED}" _position)
  if(_position EQUAL -1)
    message(FATAL_ERROR "${LABEL} did not contain '${EXPECTED}':\n${TEXT}")
  endif()
endfunction()

execute_process(
  COMMAND "${TRTMC_CLI}" --help
  RESULT_VARIABLE _cli_status
  OUTPUT_VARIABLE _cli_stdout
  ERROR_VARIABLE _cli_stderr
)
if(NOT "${_cli_status}" STREQUAL "0")
  message(FATAL_ERROR "trtmc --help failed (${_cli_status}):\n${_cli_stdout}\n${_cli_stderr}")
endif()
_trtmc_require_contains("trtmc --help" "${_cli_stdout}\n${_cli_stderr}" "Usage:")

execute_process(
  COMMAND "${TRTMC_BENCHMARK_WORKER}" --help
  RESULT_VARIABLE _worker_status
  OUTPUT_VARIABLE _worker_stdout
  ERROR_VARIABLE _worker_stderr
)
if(NOT "${_worker_status}" STREQUAL "0")
  message(FATAL_ERROR
    "trtmc_benchmark_worker --help failed (${_worker_status}):\n"
    "${_worker_stdout}\n${_worker_stderr}")
endif()
_trtmc_require_contains(
  "trtmc_benchmark_worker --help"
  "${_worker_stdout}\n${_worker_stderr}"
  "Usage:")

file(MAKE_DIRECTORY "${TRTMC_TEST_DIRECTORY}")
set(_dataset "${TRTMC_TEST_DIRECTORY}/malformed.jsonl")
set(_output "${TRTMC_TEST_DIRECTORY}/output.jsonl")
file(WRITE "${_dataset}" "{\"sample_id\":\"broken\"\n")
execute_process(
  COMMAND "${TRTMC_DATASET_BENCHMARK}"
    "${TRTMC_TEST_DIRECTORY}/missing.bundle"
    "${_dataset}"
    "${_output}"
  RESULT_VARIABLE _dataset_status
  OUTPUT_VARIABLE _dataset_stdout
  ERROR_VARIABLE _dataset_stderr
)
if("${_dataset_status}" STREQUAL "0")
  message(FATAL_ERROR "trtmc_dataset_benchmark accepted malformed JSONL")
endif()
_trtmc_require_contains(
  "trtmc_dataset_benchmark malformed JSONL"
  "${_dataset_stdout}\n${_dataset_stderr}"
  "Malformed JSON at line 1")
