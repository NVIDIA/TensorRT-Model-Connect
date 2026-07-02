# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative, namespaced, self-registering runtime configuration.

Mirrors the C++ registry in ``include/trtmc/config/`` and
``src/runtime/config/``. The design contract is documented in
``website/docs/context/config-registry-status.md``:

    - Static init / module import: registers *schemas* only (metadata +
      defaults). No values.
    - Session start: the CLI/profile loader assembles a
      ``ConfigBundle`` by merging layered contributions against the
      registered schemas. Highest priority wins per field; schema
      defaults fill gaps.
    - Features query their own namespace via
      ``bundle.get("triattention", "kv_budget")``.

The bundle provides defaults, not ground truth. The runtime never
"overrides" the bundle — that word is forbidden in any identifier
inside this package.
"""

from tensorrt_model_connect.runtime_config.schema_registry import (
    ConfigField,
    Layer,
    Schema,
    SchemaRegistry,
    clear_for_testing,
    lookup,
    register_schema,
    registered_namespaces,
)
from tensorrt_model_connect.runtime_config.config_bundle import (
    ConfigBundle,
    LayerContribution,
    ResolvedValue,
    bundle_defaults_contribution,
    write_effective_config,
)
from tensorrt_model_connect.runtime_config.cli_support import (
    build_cli_contribution,
    coerce_scalar,
    load_layered_file,
    parse_set_token,
    parse_set_tokens,
    resolve_cli_config,
    write_effective_config_next_to,
)

__all__ = [
    "ConfigBundle",
    "ConfigField",
    "Layer",
    "LayerContribution",
    "ResolvedValue",
    "Schema",
    "SchemaRegistry",
    "build_cli_contribution",
    "bundle_defaults_contribution",
    "clear_for_testing",
    "coerce_scalar",
    "load_layered_file",
    "lookup",
    "parse_set_token",
    "parse_set_tokens",
    "register_schema",
    "registered_namespaces",
    "resolve_cli_config",
    "write_effective_config",
    "write_effective_config_next_to",
]
