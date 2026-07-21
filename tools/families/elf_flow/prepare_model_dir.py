# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer entry point for preparing an ELF Flow model directory."""

from tensorrt_model_connect.families.elf_flow.model.components.prepare_model_dir import (
    main,
    prepare_model_dir,
    resolve_model_dir,
)

__all__ = ["main", "prepare_model_dir", "resolve_model_dir"]


if __name__ == "__main__":
    raise SystemExit(main())
