# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-local atomic component publication."""

import json
import os
from pathlib import Path
import tempfile

def write_component(output, plan_name, plan, manifest):
    """Publish a complete component without updating an existing directory."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if Path(plan_name).name != plan_name or not plan_name.endswith(".plan"):
        raise ValueError("plan_name must be a .plan filename")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cosyvoice3-", dir=output.parent) as tmp:
        stage = Path(tmp) / "component"
        stage.mkdir()
        (stage / plan_name).write_bytes(plan)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
