# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lance-owned checkpoint staging CLI."""

from importlib import import_module

main = import_module(
    "tensorrt_model_connect.families.lance.prepare_model"
).main

if __name__ == "__main__":
    raise SystemExit(main())
