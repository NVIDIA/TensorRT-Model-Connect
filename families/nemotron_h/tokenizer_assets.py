# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native bundle tokenizer assets with checkpoint-defined template precedence."""

import json
from pathlib import Path


def native_tokenizer_assets(model_dir: Path, filenames: tuple[str, ...]) -> dict[str, bytes]:
    """Embed the effective original template; do not mutate checkpoint assets.

    The C++ runtime has one canonical tokenizer_config contract. Newer HF
    checkpoints put the template in a standalone file, which takes precedence
    over the older JSON field. Normalize only the bundle copy of that field.
    """
    assets = {
        name: (model_dir / name).read_bytes() for name in filenames if (model_dir / name).is_file()
    }
    metadata = json.loads(assets["tokenizer_config.json"])
    if not isinstance(metadata, dict):
        raise ValueError("Nemotron-H tokenizer_config.json must be an object")
    standalone = model_dir / "chat_template.jinja"
    template = (
        standalone.read_text(encoding="utf-8")
        if standalone.is_file()
        else metadata.get("chat_template")
    )
    if not isinstance(template, str) or not template.strip():
        raise ValueError("Nemotron-H requires an embedded or standalone chat template")
    metadata["chat_template"] = template
    assets["tokenizer_config.json"] = json.dumps(
        metadata, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return assets
