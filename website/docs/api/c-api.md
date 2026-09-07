---
title: C Task API
---

`<trtmc/c_api.h>` is a C11-compatible wrapper over the current abstract Task
interfaces. It exposes no C++ types. Link `libtrtmc_c.so`; the adapter delegates
loading to `libtrtmc_runtime.so` and does not contain model behavior.

```cmake
find_package(trtmc 0.1.0 EXACT REQUIRED)
target_link_libraries(my_c_app PRIVATE trtmc::trtmc_c)
```

## Status and errors

Every operation that can fail returns one of these values:

| Status | Meaning |
| --- | --- |
| `TRTMC_STATUS_OK` | The call completed and cleared this thread's last error. |
| `TRTMC_STATUS_INVALID_ARGUMENT` | A required pointer, path, count, or numeric value is invalid. |
| `TRTMC_STATUS_WRONG_TASK` | The loaded task does not implement the requested typed operation. |
| `TRTMC_STATUS_RUNTIME_ERROR` | Loading or the family implementation failed. |
| `TRTMC_STATUS_OUT_OF_MEMORY` | The adapter could not copy a caller-owned result. |

`trtmc_last_error()` returns a borrowed thread-local string. It is valid until
the next status-returning C API call on that thread and must not be freed.
`trtmc_version()` returns a borrowed process-lifetime string.

## Load and destroy

```c
#include <trtmc/c_api.h>
#include <stdio.h>

trtmc_task_t* task = NULL;
trtmc_status_t status = trtmc_task_load(
    "model.bundle", "/opt/trtmc/lib", &task);
if (status != TRTMC_STATUS_OK) {
    fprintf(stderr, "%s\n", trtmc_last_error());
    return 1;
}

const char* name = NULL;
status = trtmc_task_name(task, &name);
/* name is borrowed until task is destroyed. */

trtmc_task_destroy(task);
```

`runtime_root` is always explicit. The loader opens exactly the family and
backend named by the bundle and does not try another implementation.

## Text generation

```c
trtmc_text_result_t result = {0};
status = trtmc_text_generate(task, "Hello", 32, &result);
if (status == TRTMC_STATUS_OK) {
    fwrite(result.text, 1, result.text_length, stdout);
}
trtmc_text_result_free(&result);
```

`max_new_tokens` must be positive. On success, `text` is null-terminated,
`text_length` excludes that terminator, and both `text` and `token_ids` belong
to the caller. Free the whole result only with `trtmc_text_result_free()`.

## Batch image generation

```c
const char* prompts[] = {"A red fox", "A blue bird"};
const uint32_t seeds[] = {42, 7};
trtmc_image_result_t results[2] = {{0}};

status = trtmc_image_generate_batch(
    task, prompts, seeds, 2, 20, 7.5f, results);
if (status == TRTMC_STATUS_OK) {
    /* Each pixels buffer is contiguous float THWC/HWC data. */
}
trtmc_image_result_free(&results[0]);
trtmc_image_result_free(&results[1]);
```

The task must implement `image_generation_batch`. Prompts and seeds have one
shared positive count; `num_steps` is positive and `guidance_scale` is finite
and non-negative. Each image result owns an independent pixel allocation and is
freed independently.

Result structs must be zero-initialized before first use or released before
reuse. When a valid output object or array is supplied, failed typed operations
leave its owned pointers and sizes zeroed.
