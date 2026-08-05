# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize every declared Python profile during the CI image build."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tensorrt_model_connect.python_profiles import (
    load_python_profile_registry,
    prebuilt_python_profile_names,
    profile_root,
    resolve_profile_python,
)


def main() -> None:
    names = prebuilt_python_profile_names(load_python_profile_registry())
    if not names:
        raise SystemExit("no family-owned Python profiles were declared")

    base_python = os.environ.get("TRTMC_BASE_PYTHON", "/opt/venv/bin/python")
    resolved: dict[str, dict[str, str]] = {}
    for name in names:
        python = resolve_profile_python(name, base_python)
        ready = Path(python).parent.parent / ".ready"
        if not ready.is_file():
            raise SystemExit(f"profile {name!r} was not marked ready: {ready}")
        resolved[name] = {"python": python, "ready": str(ready)}

    manifest = {
        "schema_version": 1,
        "profiles": resolved,
    }
    root = profile_root()
    (root / ".image-ready.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("prebuilt_python_profiles=" + ",".join(names))


if __name__ == "__main__":
    main()
